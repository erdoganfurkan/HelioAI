"""Build the speasy catalog ChromaDB index.

Usage:
    helioai index           # incremental (skip existing)
    helioai index --rebuild # wipe and rebuild
"""

from __future__ import annotations

import html
import re
import shutil
import time
from pathlib import Path

# SPASE Region vocabulary — maps mission/spacecraft name fragments to SPASE Region values.
# Covers both AMDA-style names and CDA/CSA-style spacecraft codes.
# Source: https://spase-group.org/data/model/spase-2.6.0 — Region simpleType
_SPACECRAFT_REGION: dict[str, str] = {
    # AMDA-style mission names
    "ace": "Heliosphere.NearEarth",
    "wind": "Heliosphere.NearEarth",
    "dscovr": "Heliosphere.NearEarth",
    "cluster": "Earth.Magnetosphere",
    "mms": "Earth.Magnetosphere",
    "themis": "Earth.Magnetosphere",
    "goes": "Earth.Magnetosphere",
    "geotail": "Earth.Magnetosphere",
    "polar": "Earth.Magnetosphere.Polar",
    "fast": "Earth.Magnetosphere.Polar",
    "image": "Earth.Magnetosphere",
    "van allen": "Earth.Magnetosphere.RadiationBelt",
    "rbsp": "Earth.Magnetosphere.RadiationBelt",
    "crres": "Earth.Magnetosphere.RadiationBelt",
    "eiscat": "Earth.NearSurface.Ionosphere",
    "swarm": "Earth.NearSurface.Ionosphere",
    "dmsp": "Earth.NearSurface.AuroralRegion",
    "solar orbiter": "Heliosphere.Inner",
    "solo": "Heliosphere.Inner",
    "parker": "Heliosphere.Inner",
    "psp": "Heliosphere.Inner",
    "helios": "Heliosphere.Inner",
    "stereo": "Heliosphere.Inner",
    "ulysses": "Heliosphere",
    "voyager": "Heliosphere.Outer",
    "new horizons": "Heliosphere.Outer",
    "cassini": "Saturn",
    "galileo": "Jupiter",
    "juno": "Jupiter",
    "bepicolombo": "Mercury",
    "bepi": "Mercury",
    "mex": "Mars",
    "maven": "Mars",
    "vex": "Venus",
    # CDA-style spacecraft codes
    "ac": "Heliosphere.NearEarth",  # ACE
    "wi": "Heliosphere.NearEarth",  # Wind
    "ds": "Heliosphere.NearEarth",  # DSCOVR
    "c1": "Earth.Magnetosphere",  # Cluster-1
    "c2": "Earth.Magnetosphere",  # Cluster-2
    "c3": "Earth.Magnetosphere",  # Cluster-3
    "c4": "Earth.Magnetosphere",  # Cluster-4
    "th": "Earth.Magnetosphere",  # THEMIS (THA/THB/THC/THD/THE)
    "mms1": "Earth.Magnetosphere",
    "mms2": "Earth.Magnetosphere",
    "mms3": "Earth.Magnetosphere",
    "mms4": "Earth.Magnetosphere",
    "ge": "Earth.Magnetosphere",  # Geotail
    "po": "Earth.Magnetosphere.Polar",
    "fa": "Earth.Magnetosphere.Polar",  # FAST
    "rbspa": "Earth.Magnetosphere.RadiationBelt",
    "rbspb": "Earth.Magnetosphere.RadiationBelt",
    "sta": "Heliosphere.Inner",  # STEREO-A
    "stb": "Heliosphere.Inner",  # STEREO-B
    "ul": "Heliosphere",  # Ulysses
    "vg1": "Heliosphere.Outer",  # Voyager-1
    "vg2": "Heliosphere.Outer",  # Voyager-2
    "cas": "Saturn",  # Cassini
    "jno": "Jupiter",  # Juno
    "mav": "Mars",  # MAVEN
}


_PROVIDER_PREFIXES = {
    "amda": "amda",
    "cda": "cda",
    "csa": "csa",
    "ssc": "ssc",
    "archive": "archive",
    "uiowaephtool": "uiowaephtool",
}

_STRIP_HTML_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return _STRIP_HTML_RE.sub(" ", html.unescape(text)).strip()


def _get_region(uid: str) -> str:
    """Map a speasy uid to a SPASE Region string."""
    uid_lower = uid.lower()
    # Strip provider prefix: "amda/ace_b_gse" → "ace_b_gse"
    dataset_part = uid_lower.split("/", 1)[-1]
    # Try first token (most specific)
    mission_key = dataset_part.split("_")[0].split("-")[0]
    region = _SPACECRAFT_REGION.get(mission_key, "")
    if not region:
        # Fallback: substring search for multi-word keys and edge cases
        searchable = uid_lower.replace("/", " ").replace("_", " ")
        for k, v in _SPACECRAFT_REGION.items():
            if k in searchable:
                region = v
                break
    return region


def _region_for(uid: str, parent_meta: dict | None) -> str:
    """The product's SPASE Region — published if the archive states one, guessed otherwise.

    `_get_region` matches a 40-entry table as a substring, and its two-letter keys
    collide: "ac" (ACE) matches inside "cce_mepa_ion_act", so AMPTE/CCE particle
    counts in the magnetosheath were labelled Heliosphere.NearEarth. Measured against
    AMDA's own `target`, the table disagrees on 1279 products and is silent on 1813
    more. The region is written into the indexed text, so a wrong one is wrong
    retrieval vocabulary, not just wrong metadata.

    The table stays for the 21% of AMDA datasets that publish no target, and for CDA
    and CSA, which publish none at all.
    """
    return (parent_meta or {}).get("region") or _get_region(uid)


_AMDA_SPASE_ROOT = "CDPP-AMDA/"


def _amda_mission(spase_id: str) -> str:
    """Spacecraft and instrument out of an AMDA SPASE id.

    The ids run `spase://CNES/NumericalData/CDPP-AMDA/<MISSION>/<INSTRUMENT>/<dataset>`,
    and all 1075 AMDA datasets carry one. Everything but the trailing dataset segment is
    kept, which handles the two- and four-segment variants without special-casing them.

    Example:
        >>> _amda_mission("spase://CNES/NumericalData/CDPP-AMDA/ACE/MFI/ace-imf-all")
        'ACE MFI'
    """
    if _AMDA_SPASE_ROOT not in spase_id:
        return ""
    segments = spase_id.split(_AMDA_SPASE_ROOT, 1)[1].strip("/").split("/")
    return " ".join(segments[:-1])


def _extract_dataset_meta(child_vars: dict, provider_prefix: str) -> dict:
    """Extract scientific metadata from a DatasetIndex node for propagation to its parameters."""
    meta: dict = {}
    if provider_prefix == "amda":
        mtype = child_vars.get("measurement_type") or ""
        if mtype:
            meta["measurement_type"] = mtype
        desc = child_vars.get("desc") or ""
        if desc:
            meta["dataset_description"] = _strip_html(desc)[:200]
        dataset_id = child_vars.get("xmlid") or ""
        if dataset_id:
            meta["dataset_id"] = dataset_id
        mission = _amda_mission(child_vars.get("spaseId") or "")
        if mission:
            meta["mission"] = mission
        level = child_vars.get("processing_level") or ""
        if level:
            meta["processing_level"] = level
        target = child_vars.get("target") or ""
        if target:
            meta["region"] = target
    elif provider_prefix == "cda":
        desc = child_vars.get("description") or ""
        if desc:
            meta["dataset_description"] = desc[:200]
    elif provider_prefix == "csa":
        mtypes = child_vars.get("measurement_types") or ""
        if mtypes:
            meta["measurement_type"] = mtypes if isinstance(mtypes, str) else ", ".join(mtypes)
        category = child_vars.get("category") or ""
        if category:
            meta["category"] = category
        title = child_vars.get("title") or ""
        if title:
            meta["dataset_description"] = title[:200]
        observatory = child_vars.get("observatory_name") or ""
        if observatory:
            meta["observatory"] = observatory
        experiments = child_vars.get("experiments") or ""
        if experiments:
            meta["experiments"] = (
                experiments if isinstance(experiments, str) else ", ".join(experiments)
            )
    return meta


def _is_dataset_node(child_vars: dict, provider_prefix: str) -> bool:
    """Detect a dataset-level container node (not a leaf parameter)."""
    spz_type = child_vars.get("__spz_type__") or ""
    if spz_type == "DatasetIndex":
        return True
    # AMDA doesn't set __spz_type__; detect by presence of measurement_type/desc without xmlid
    if provider_prefix == "amda" and not child_vars.get("xmlid"):
        return bool(child_vars.get("measurement_type") or child_vars.get("desc"))
    return False


def build_index(
    rebuild: bool = False, batch_size: int = 128, verbose: bool = True, classify: bool = False
) -> int:
    """Walk the speasy inventory and index all parameters into ChromaDB.

    Backs `helioai index` and must run once before `search_parameters` works;
    the index persists under `settings.rag.chroma_dir`.

    Args:
        rebuild: Drop and re-create the collection instead of appending.
        batch_size: Documents per ChromaDB insert.
        verbose: Print per-provider progress to stdout.
        classify: Ask the judgment backend for the SPASE measurement type of every
            product the archive leaves untyped, before embedding (`--classify`; needs
            `HELIOAI_JUDGMENT_BACKEND=jev`). See `classify_measurement_types`.

    Returns:
        Number of parameters indexed (0 when speasy or chromadb is missing).

    Example:
        >>> build_index(rebuild=True)   # equivalent to: helioai index --rebuild
        82433
    """
    try:
        import chromadb  # noqa: F401 — the index cannot be built without it; opened in open_collections
        import speasy as spz
        from sentence_transformers import SentenceTransformer
        from speasy.core.inventory.indexes import SpeasyIndex
    except ImportError as e:
        print(f"[indexer] Missing dependency: {e}")
        print("[indexer] Run: pip install speasy chromadb sentence-transformers")
        return 0

    from helioai.config import settings

    chroma_dir = settings.rag.chroma_dir
    collection_name = settings.rag.collection_name
    embed_model = settings.rag.embed_model

    if rebuild and chroma_dir.exists():
        if verbose:
            print(f"[indexer] wiping {chroma_dir}")
        shutil.rmtree(chroma_dir)

    chroma_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[indexer] loading embedding model {embed_model}…")
    model = SentenceTransformer(embed_model)

    _, (collection, catalog_collection) = open_collections(
        chroma_dir, [collection_name, settings.rag.catalogs_collection_name], verbose=verbose
    )

    existing_ids: set[str] = set()
    if not rebuild:
        try:
            existing_ids = set(collection.get(include=[])["ids"])
            if verbose and existing_ids:
                print(f"[indexer] {len(existing_ids)} existing entries — skipping")
        except Exception:
            pass

    if verbose:
        print("[indexer] walking speasy inventory…")

    docs: list[dict] = []
    tree = spz.inventories.tree

    for provider_attr, prefix in _PROVIDER_PREFIXES.items():
        provider_node = getattr(tree, provider_attr, None)
        if provider_node is None:
            continue
        before = len(docs)
        _walk(provider_node, prefix, docs, existing_ids, SpeasyIndex)
        if verbose:
            print(f"[indexer]   {prefix}: {len(docs) - before} new params")

    if verbose:
        print(f"[indexer] total new params to index: {len(docs)}")

    if not docs:
        if verbose:
            print("[indexer] up to date — nothing to index")
        return 0

    if classify:
        import asyncio

        docs = asyncio.run(classify_measurement_types(docs, chroma_dir, verbose=verbose))

    t0 = time.perf_counter()
    total = 0

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        ids = [d["id"] for d in batch]
        texts = [d["text"] for d in batch]
        metas = [d["meta"] for d in batch]

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metas)
        total += len(ids)
        if verbose:
            print(f"[indexer]   {total}/{len(docs)} indexed…", end="\r", flush=True)

    elapsed = time.perf_counter() - t0
    if verbose:
        print()
        print(f"[indexer] done: {total} params in {elapsed:.1f}s")
        print(f"[indexer] collection total: {collection.count()}")

    # Index catalogs + timetables into a separate collection
    cat_total = _build_catalog_index(
        model, catalog_collection, settings, rebuild=rebuild, verbose=verbose
    )

    return total + cat_total


MEASUREMENT_TYPES: tuple[str, ...] = (
    "MagneticField",
    "ElectricField",
    "ThermalPlasma",
    "EnergeticParticles",
    "IonComposition",
    "Ephemeris",
    "Waves",
    "Spectrum",
    "NeutralGas",
    "InstrumentStatus",
    "Irradiance",
    "Radiance",
)
"""The SPASE MeasurementType vocabulary the index already carries (AMDA, CSA), as the
closed set a classifier chooses from — so a filled field is usable as an exact filter."""

MEASUREMENT_TYPE_INSTRUCTIONS = (
    "Which SPASE MeasurementType best describes this archived data product? The type "
    "follows the instrument's population, as SPASE assigns it: ThermalPlasma covers "
    "everything a thermal plasma analyser produces — bulk moments (density, temperature, "
    "velocity), and also the electron or ion distributions, pitch-angle fluxes, phase-space "
    "densities and raw counts of instruments such as PEACE, CIS/HIA/CODIF, EAS, SWA, SPC, "
    "SWE, FPI, HPCA, even when binned in keV. EnergeticParticles is for dedicated "
    "energetic-particle detectors (tens of keV to GeV: EPD, RAPID, EIS, FEEPS, SIS, EPHIN, "
    "cosmic rays). Ephemeris is a spacecraft position, orbit, attitude or a geometric angle. "
    "IonComposition is per-species ion measurements of a mass spectrometer. Waves and "
    "Spectrum are wave or spectral products. InstrumentStatus is housekeeping: temperatures, "
    "voltages, modes, quality flags."
)
MEASUREMENT_TYPE_FLOOR = 0.9
_LABEL_SENTENCE = re.compile(r"\s*Measurement:\s*[^.]*\.\s*")


def _normalised_type(label: str | None) -> str:
    return "".join(ch for ch in str(label or "").split(",")[0].lower() if ch.isalnum())


async def classify_measurement_types(
    docs: list[dict], record_dir, *, verbose: bool = False
) -> list[dict]:
    """Fill `measurement_type` where the archive left it empty; flag it where the judge
    disagrees with the archive. Never overwrites a published label.

    The field is indexed on 15.6 % of the products — AMDA and CSA — and on none of CDA's
    68 000, so every ranking signal built on it (`_rerank_penalty`) and every filter reaches
    a sixth of the catalogue. Measured on 2026-09-22 against 200 products the archive had
    labelled, label stripped from the text before asking: 76.5 % agreement, **89 % where the
    judge's confidence is at least 0.9** (72 % of the items) — and the remaining confident
    disagreements were the archive's errors (MMS FPI plasma moments labelled MagneticField,
    a JADE density labelled EnergeticParticles, a Langmuir-probe density labelled
    ElectricField). So the judge is better than its ground truth, and the floor is 0.9:
    below it the field stays empty — abstention is a type — and a published label the judge
    contradicts at or above it is kept and flagged as `measurement_type_jev`, for a person
    to adjudicate, never replaced.

    A filled type is written into the metadata (with `measurement_type_source: "jev"` and
    the confidence) and into the text (`Measurement: X.`), where the dense and sparse
    channels read it. Every call is recorded to `judgment_index.jsonl` in the index
    directory. 37 ms per product at eight in flight: the whole catalogue in ~45 minutes.

    Args:
        docs: `{id, text, meta}` as `_walk` collects them.
        record_dir: Where the calls are recorded (the Chroma directory).
        verbose: Print the counts.

    Returns:
        The same docs, metadata and text amended in place.
    """
    from helioai.config import settings
    from helioai.core import judgment

    if settings.judgment.backend == "null":
        if verbose:
            print(
                "[indexer] --classify needs HELIOAI_JUDGMENT_BACKEND=jev and TYPESAFE_API_KEY; "
                "skipping classification"
            )
        return docs
    question = {
        "mtype": judgment.Choice(
            MEASUREMENT_TYPE_INSTRUCTIONS, MEASUREMENT_TYPES, floor=MEASUREMENT_TYPE_FLOOR
        )
    }
    states = [{"product": _LABEL_SENTENCE.sub(" ", d["text"]).strip()} for d in docs]
    answers = await judgment.batch(
        "index_measurement_type",
        states,
        question,
        record_to=Path(record_dir) / "judgment_index.jsonl",
    )
    filled = flagged = abstained = 0
    for doc, answer in zip(docs, answers, strict=True):
        label = answer["mtype"] if answer is not None else None
        if label is None:
            abstained += 1
            continue
        meta = doc["meta"]
        published = meta.get("measurement_type")
        confidence = float((answer.raw.get("mtype") or {}).get("confidence") or 0.0)
        if not published:
            meta["measurement_type"] = label
            meta["measurement_type_source"] = "jev"
            meta["measurement_type_confidence"] = round(confidence, 3)
            doc["text"] = f"{doc['text'].rstrip()} Measurement: {label}."
            filled += 1
        elif _normalised_type(published) != _normalised_type(label):
            meta["measurement_type_jev"] = label
            meta["measurement_type_jev_confidence"] = round(confidence, 3)
            flagged += 1
    if verbose:
        print(
            f"[indexer] measurement types: {filled} filled, {flagged} published labels flagged, "
            f"{abstained} abstained (floor {MEASUREMENT_TYPE_FLOOR}), of {len(docs)}"
        )
    return docs


HNSW_SYNC_THRESHOLD = 1
HNSW_EF_SEARCH = 400


def open_collections(chroma_dir, names: list[str], *, verbose: bool = False):
    """Open or create the index's collections so that every write is persisted at once,
    and so that the dense search looks wide enough to find a near-twin.

    Chroma's local HNSW segment persists to disk only every `sync_threshold` writes — 1000
    by default. Whatever follows the last persist stays in the write-ahead log and is
    replayed into the in-memory graph at every process start, in an order that varies, so
    the graph varies and the ranking with it. Measured on 2026-09-22: the 325 SSCWeb
    trajectories added after the last persist gave five different dense top-50 lists in five
    processes for one query embedding, `ssc/mms1` at rank 1 in four of them and absent from
    the fifth. The catalogue collection, 221 entries, had never been persisted at all. A
    threshold of one costs 0.11 s per batch on the full 82k index and leaves nothing to
    replay, so a search ranks the same in every process and a read-only process never
    writes to the index.

    `ef_search` is raised from Chroma's 100 to 400. The catalogue is full of near-twins —
    314 SSCWeb trajectories that differ by a spacecraft name, hundreds of housekeeping
    variables that differ by a suffix — and an approximate search with a narrow beam loses
    the exact twin: `ssc/mms1`, the true nearest neighbour of "MMS1 spacecraft position GSE
    2019", was absent from the dense top-50 at 100 and is rank 1 at 400, for 0.7 → 1.2 ms per
    query (measured 2026-09-22). On the 30 HelioBench n1 queries the change moved recall@1
    from 53.3 % to 56.7 %.

    A collection created before these settings keeps the ones it was loaded with, so it is
    modified and the client reopened: the replay on reopen persists its tail. Reopening
    clears Chroma's process-wide client cache — fine in `helioai index`, and the reason this
    is not done lazily by a process that also serves searches.

    Args:
        chroma_dir: The Chroma directory, created when absent.
        names: Collection names, opened in order.
        verbose: Print when a legacy collection is settled.

    Returns:
        `(client, collections)` — the client the collections belong to.
    """
    import chromadb
    from chromadb.api.client import SharedSystemClient

    wanted = {"sync_threshold": HNSW_SYNC_THRESHOLD, "ef_search": HNSW_EF_SEARCH}
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collections = [
        client.get_or_create_collection(
            name=n, configuration={"hnsw": {"space": "cosine", **wanted}}
        )
        for n in names
    ]
    legacy = [
        c
        for c in collections
        if any((c.configuration_json.get("hnsw") or {}).get(k) != v for k, v in wanted.items())
    ]
    if not legacy:
        return client, collections
    for c in legacy:
        c.modify(configuration={"hnsw": wanted})
    SharedSystemClient.clear_system_cache()
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collections = [client.get_collection(n) for n in names]
    for c in collections:
        c.count()
    if verbose:
        print(f"[indexer] settled {len(legacy)} collection(s): pending writes persisted")
    return client, collections


def _walk(
    node,
    provider_prefix: str,
    out: list[dict],
    skip_ids: set[str],
    SpeasyIndex,
    depth: int = 0,
    max_docs: int = 200_000,
    parent_meta: dict | None = None,
) -> None:
    """Recursively walk a SpeasyIndex node, collecting indexable parameters."""
    if len(out) >= max_docs or depth > 15:
        return

    for child in vars(node).values():
        if not isinstance(child, SpeasyIndex) or len(out) >= max_docs:
            continue

        child_vars = vars(child)
        spz_type = child_vars.get("__spz_type__") or ""

        # Dataset container: capture metadata and propagate to children
        if _is_dataset_node(child_vars, provider_prefix):
            dataset_meta = _extract_dataset_meta(child_vars, provider_prefix)
            _walk(
                child,
                provider_prefix,
                out,
                skip_ids,
                SpeasyIndex,
                depth + 1,
                max_docs,
                dataset_meta,
            )
            continue

        if _is_time_axis(child_vars):
            _walk(
                child, provider_prefix, out, skip_ids, SpeasyIndex, depth + 1, max_docs, parent_meta
            )
            continue

        if provider_prefix == "ssc" and spz_type == "ParameterIndex":
            doc = _ssc_trajectory_doc(child_vars, skip_ids)
            if doc is not None:
                out.append(doc)
            continue

        xmlid = child_vars.get("xmlid") or ""
        description = child_vars.get("description") or ""
        is_amda = bool(xmlid)

        if not (xmlid and description) and spz_type in ("ParameterIndex", "ComponentIndex"):
            if not is_amda:
                xmlid = child_vars.get("__spz_uid__") or ""
            description = (
                child_vars.get("CATDESC") or child_vars.get("cat_description") or description or ""
            )

        name = (
            child_vars.get("name")
            or child_vars.get("FIELDNAM")
            or child_vars.get("field_name")
            or child_vars.get("__spz_name__")
            or xmlid
        )
        units = child_vars.get("units") or child_vars.get("UNITS") or ""

        # CSA ParameterIndex carries entity/property directly (no parent needed)
        entity = child_vars.get("entity") or ""
        prop = child_vars.get("property") or ""

        if xmlid and (description or is_amda):
            uid = f"{provider_prefix}/{xmlid}"
            if uid not in skip_ids:
                skip_ids.add(uid)
                region = _region_for(uid, parent_meta)
                cov_start, cov_stop = _coverage(child_vars)
                text = _build_text(
                    name,
                    description,
                    units,
                    xmlid,
                    parent_meta=parent_meta,
                    entity=entity,
                    prop=prop,
                    region=region,
                    coverage=(cov_start, cov_stop),
                )
                if text.strip():
                    meta_entry: dict = {
                        "name": name,
                        "units": units,
                        "xmlid": xmlid,
                        "provider": provider_prefix,
                    }
                    mtype = (parent_meta or {}).get("measurement_type") or ""
                    if mtype:
                        meta_entry["measurement_type"] = mtype
                    if region:
                        meta_entry["region"] = region
                    if cov_start:
                        meta_entry["start_time"] = cov_start
                    if cov_stop:
                        meta_entry["stop_time"] = cov_stop
                    out.append({"id": uid, "text": text, "meta": meta_entry})

        _walk(child, provider_prefix, out, skip_ids, SpeasyIndex, depth + 1, max_docs, parent_meta)


_TIME_CDF_TYPES = ("CDF_EPOCH", "CDF_EPOCH16", "CDF_TIME_TT2000")


def _ssc_trajectory_doc(child_vars: dict, skip_ids: set[str]) -> dict | None:
    """The index entry for one SSC trajectory, or None when it is not one.

    SSC nodes have neither `xmlid` nor `description` — only `Id`, `Resolution`,
    `start_date`, `stop_date` and the display name — so the general walk skipped all 314
    of them and the index had no spacecraft position at all: a question "where was MMS1"
    could only be answered from an instrument's own ephemeris variable, when one existed,
    and `provider="ssc"` returned nothing while the prompt offered it. The text is written
    the way a position question is asked; the id is the one speasy downloads
    (`spz.get_data("ssc/mms1", …)` — GSE km by default, GSM/GEO/SM on request).
    """
    uid = str(child_vars.get("__spz_uid__") or child_vars.get("Id") or "")
    if not uid:
        return None
    pid = f"ssc/{uid}"
    if pid in skip_ids:
        return None
    skip_ids.add(pid)
    name = str(child_vars.get("__spz_name__") or uid)
    resolution = child_vars.get("Resolution")
    cadence = f" Cadence: {resolution} s." if resolution else ""
    region = _get_region(pid)
    cov_start, cov_stop = _coverage(child_vars)
    dates = _coverage_sentence((cov_start, cov_stop))
    text = (
        f"{name} spacecraft position (orbit, trajectory, ephemeris) from NASA SSCWeb. "
        f"Location of {name} as X, Y, Z in km — GSE by default; GSM, GEO, GEI, SM, GSM "
        f"on request. Where the spacecraft was at a given time.{cadence} Units: km."
        + (f" Region: {region}." if region else "")
        + (f" {dates}" if dates else "")
    )
    meta: dict = {"name": name, "units": "km", "xmlid": uid, "provider": "ssc"}
    if region:
        meta["region"] = region
    if cov_start:
        meta["start_time"] = cov_start
    if cov_stop:
        meta["stop_time"] = cov_stop
    return {"id": pid, "text": text, "meta": meta}


def _is_time_axis(child_vars: dict) -> bool:
    """True for a CDF time axis, which is never a plottable product.

    `cda/AC_OR_SSC/Epoch` was indexed as if it were the ACE position: its CDF metadata
    carries `FIELDNAM = XYZ_GSE` and `CATDESC = "ACE X/Y/Z GSE coordinates"`, so the
    search returned it for a position query and the download died inside speasy with
    `tuple index out of range`. `cdf_type` is the one attribute that says what it is.
    """
    cdf_type = str(child_vars.get("cdf_type") or "").upper()
    if cdf_type in _TIME_CDF_TYPES:
        return True
    return str(child_vars.get("__spz_name__") or "").lower() in {"epoch", "time", "time_tags"}


def _coverage(child_vars: dict) -> tuple[str, str]:
    """Published (start, stop) for a product, as plain ISO dates, or ("", "").

    speasy already carries these in the inventory, so indexing them costs nothing extra
    and lets a search drop products that cannot cover the window being asked about.
    """
    out = []
    for key in ("start_date", "stop_date"):
        v = child_vars.get(key)
        out.append(str(v)[:19] if v else "")
    return out[0], out[1]


def _coverage_sentence(coverage: tuple[str, str] | None) -> str:
    """ "Coverage: 1994-11-13 to 2026-09-01." — the dates a query names, as searchable text.

    Nearly every question names a year, and the indexed text never did: "2019" in
    "MMS1 position 2019" matched nothing in either channel and only diluted the rest,
    while a product ending in 1997 ranked as if it covered the date. The catalogue index
    has written `Survey: … to …` since it was built; this is the same sentence for the
    products.
    """
    if not coverage:
        return ""
    start, stop = (str(c or "")[:10] for c in coverage)
    return f"Coverage: {start} to {stop}." if start and stop else ""


def _build_text(
    name: str,
    description: str,
    units: str,
    xmlid: str,
    parent_meta: dict | None = None,
    entity: str = "",
    prop: str = "",
    region: str = "",
    coverage: tuple[str, str] | None = None,
) -> str:
    head = name if name != xmlid else xmlid.replace("_", " ")
    parts = [f"{head}."]
    if description:
        parts.append(f"{description}.")
    if entity and prop:
        parts.append(f"{entity} {prop}.")
    elif entity:
        parts.append(f"Particle: {entity}.")
    meta = parent_meta or {}
    mtype = meta.get("measurement_type") or ""
    if mtype:
        parts.append(f"Measurement: {mtype}.")
    category = meta.get("category") or ""
    if category:
        parts.append(f"Category: {category}.")
    # Which spacecraft and instrument this belongs to. Only the parent dataset knows:
    # `amda/imf` is the definitive ACE 16-second IMF vector and neither its id nor its
    # description says "ACE", so no query naming the mission could ever reach it. The
    # dataset was already resolved for its description and its identity thrown away —
    # 780 AMDA parameters were invisible to their own mission name because of it.
    for label, key in (
        ("Dataset", "dataset_id"),
        ("Mission", "mission"),
        ("Observatory", "observatory"),
        ("Instrument", "experiments"),
        ("Processing", "processing_level"),
    ):
        value = meta.get(key) or ""
        if value:
            parts.append(f"{label}: {value}.")
    dataset_desc = meta.get("dataset_description") or ""
    if dataset_desc:
        parts.append(f"{dataset_desc}.")
    if units:
        parts.append(f"Units: {units}.")
    if region:
        parts.append(f"Region: {region}.")
    dates = _coverage_sentence(coverage)
    if dates:
        parts.append(dates)
    return " ".join(parts)


def _build_catalog_index(model, collection, settings, rebuild: bool, verbose: bool) -> int:
    """Index AMDA catalogs + timetables into a dedicated ChromaDB collection."""
    try:
        import speasy as spz
    except ImportError:
        return 0

    if rebuild:
        existing = set()
    else:
        try:
            existing = set(collection.get(include=[])["ids"])
        except Exception:
            existing = set()

    docs: list[dict] = []
    try:
        flat = spz.inventories.flat_inventories.amda
        for product_type, src in (
            ("catalog", getattr(flat, "catalogs", None) or {}),
            ("timetable", getattr(flat, "timetables", None) or {}),
        ):
            for uid, idx in src.items():
                doc_id = f"amda/{uid}"
                if doc_id in existing:
                    continue
                name = str(getattr(idx, "__spz_name__", "") or getattr(idx, "name", "") or uid)
                desc = str(getattr(idx, "desc", "") or getattr(idx, "description", "") or "")[:300]
                nb = 0
                try:
                    nb = int(getattr(idx, "nbIntervals", 0) or 0)
                except (TypeError, ValueError):
                    pass
                s_start = str(getattr(idx, "surveyStart", "") or "")[:10]
                s_stop = str(getattr(idx, "surveyStop", "") or "")[:10]
                region = _get_region(doc_id)

                text_parts = [f"{name}.", f"{product_type}."]
                if region:
                    text_parts.append(f"Region: {region}.")
                if s_start and s_stop:
                    text_parts.append(f"Survey: {s_start} to {s_stop}.")
                if nb:
                    text_parts.append(f"Events: {nb}.")
                if desc:
                    text_parts.append(desc)
                text = " ".join(text_parts)

                meta: dict = {
                    "name": name,
                    "product_type": product_type,
                    "provider": "amda",
                    "nb_events": nb,
                }
                if region:
                    meta["region"] = region
                docs.append({"id": doc_id, "text": text, "meta": meta})
    except Exception as e:
        if verbose:
            print(f"[indexer] catalog walk error: {e}")
        return 0

    if not docs:
        if verbose:
            print("[indexer] catalogs: up to date — nothing to index")
        return 0

    if verbose:
        print(f"[indexer] indexing {len(docs)} catalogs/timetables…")

    embeddings = model.encode(
        [d["text"] for d in docs],
        batch_size=128,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    collection.upsert(
        ids=[d["id"] for d in docs],
        embeddings=embeddings,
        documents=[d["text"] for d in docs],
        metadatas=[d["meta"] for d in docs],
    )

    if verbose:
        print(
            f"[indexer] catalogs done: {len(docs)} entries (collection total: {collection.count()})"
        )

    return len(docs)
