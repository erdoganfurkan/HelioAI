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
from typing import Any

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


_SPASE_NUMERICAL = "/NumericalData/"
_ISO_DURATION = re.compile(
    r"^PT(?:(?P<h>\d+(?:\.\d+)?)H)?(?:(?P<m>\d+(?:\.\d+)?)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?$", re.I
)


def _cadence_label(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:g} h"
    if seconds >= 60:
        return f"{seconds / 60:g} min"
    if seconds >= 1:
        return f"{seconds:g} s"
    return f"{seconds * 1000:g} ms"


def _cda_spase(resource_id: str) -> tuple[str, str]:
    """The words of a CDAWeb dataset's SPASE resource id, and its cadence, for the text.

    A CDA product's text said what its variable was and nothing about whose it was:
    `cda/RBSP-A_MAGNETOMETER_4SEC-GSM_EMFISIS-L3/Mag` read "Mag. Magnetometer vector.
    Fluxgate magnetometer data - Craig Kletzing (University of Iowa)." — no RBSP, no Van
    Allen, no EMFISIS, no GSM, no 4 s. Those words lived only in the id, which the sparse
    channel tokenises and the dense channel never sees, so for "Van Allen Probe A EMFISIS
    4-second magnetic field GSM" every ACE, ISEE and IMP-8 vector outranked it (2026-09-22,
    rank 10–11 in the agent's third query). The same defect that hid 780 AMDA products
    from their mission name, fixed for AMDA by the `Dataset:`/`Mission:` sentences and
    never for the 68 000 CDA products. CDAWeb publishes the words: 64 % of its datasets
    carry `spase://NASA/NumericalData/RBSP/A/EMFISIS/MAGNETOMETER/L3/GSM/PT4S` — mission,
    spacecraft, instrument, level, frame and, last, the cadence as an ISO 8601 duration.

    Args:
        resource_id: The dataset's `spase_DatasetResourceID`, possibly blank.

    Returns:
        `(words, cadence)`: the path segments after `NumericalData/` joined by spaces,
        without the duration, and the duration rendered ("4 s", "1 min"); both empty
        when the id is absent or not of that shape.

    Example:
        >>> _cda_spase("spase://NASA/NumericalData/RBSP/A/EMFISIS/MAGNETOMETER/L3/GSM/PT4S")
        ('RBSP A EMFISIS MAGNETOMETER L3 GSM', '4 s')
    """
    text = str(resource_id or "").strip()
    if _SPASE_NUMERICAL not in text:
        return "", ""
    segments = [seg for seg in text.split(_SPASE_NUMERICAL, 1)[1].split("/") if seg]
    cadence = ""
    if segments and (m := _ISO_DURATION.match(segments[-1])):
        seconds = (
            float(m.group("h") or 0) * 3600
            + float(m.group("m") or 0) * 60
            + float(m.group("s") or 0)
        )
        if seconds > 0:
            cadence = _cadence_label(seconds)
        segments = segments[:-1]
    return " ".join(segments), cadence


def _cda_components(child_vars: dict) -> list[str]:
    """The component labels of a CDAWeb vector, for the text: `LABL_PTR_1` reads
    `['Bx_GSM', 'By_GSM', 'Bz_GSM']` on the RBSP field vector and says its frame where
    nothing else in the product does. A scalar's single label repeats its name and is
    left out."""
    labels = child_vars.get("LABL_PTR_1")
    if isinstance(labels, str):
        labels = [labels]
    if not isinstance(labels, list | tuple):
        return []
    cleaned = [str(x).strip() for x in labels if str(x).strip()]
    return cleaned if len(cleaned) >= 2 else []


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
        dataset_id = child_vars.get("serviceprovider_ID") or child_vars.get("__spz_uid__") or ""
        if dataset_id:
            meta["dataset_id"] = str(dataset_id)
        words, cadence = _cda_spase(child_vars.get("spase_DatasetResourceID") or "")
        if words:
            meta["spase"] = words
        if cadence:
            meta["cadence"] = cadence
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
        classify: Ask the judgment backend, before embedding, for the SPASE measurement
            type of every product the archive leaves untyped and for the SPASE region of
            every product whose region is the table's guess (`--classify`; needs
            `HELIOAI_JUDGMENT_BACKEND=jev`). See `classify_products`.

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
        records = chroma_dir / JUDGMENT_RECORDS
        kept = records.read_bytes() if records.exists() else None
        if verbose:
            print(f"[indexer] wiping {chroma_dir}")
        shutil.rmtree(chroma_dir)
        if kept is not None:
            chroma_dir.mkdir(parents=True, exist_ok=True)
            records.write_bytes(kept)

    chroma_dir.mkdir(parents=True, exist_ok=True)
    judged_meta, judged = load_judged(SHIPPED_JUDGED, local_judged_path())
    if verbose and judged:
        print(
            f"[indexer] {len(judged)} products already put to the judge "
            f"(snapshot {judged_meta.get('date', '?')}, {', '.join(judged_meta.get('models') or [])})"
        )

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

    if judged:
        applied = apply_judged(docs, judged)
        if verbose:
            print(
                f"[indexer] the judge's recorded answers typed {applied} products, no request made"
            )
    if classify:
        import asyncio

        docs = asyncio.run(
            classify_products(
                docs, chroma_dir, judged=judged, judged_meta=judged_meta, verbose=verbose
            )
        )

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
_REGION_SENTENCE = re.compile(r"\s*Region:\s*[A-Za-z0-9.]+\.(?=\s|$)")

REGIONS: tuple[str, ...] = (
    "Sun",
    "Sun.Corona",
    "Heliosphere",
    "Heliosphere.Inner",
    "Heliosphere.NearEarth",
    "Heliosphere.Remote1AU",
    "Heliosphere.Outer",
    "Earth",
    "Earth.Magnetosphere",
    "Earth.Magnetosheath",
    "Earth.Magnetosphere.Polar",
    "Earth.Magnetosphere.Magnetotail",
    "Earth.Magnetosphere.RadiationBelt",
    "Earth.NearSurface",
    "Earth.NearSurface.Ionosphere",
    "Earth.NearSurface.AuroralRegion",
    "Earth.NearSurface.EquatorialRegion",
    "Earth.NearSurface.PolarCap",
    "Mercury",
    "Venus",
    "Mars",
    "Jupiter",
    "Jupiter.Io",
    "Jupiter.Europa",
    "Jupiter.Ganymede",
    "Jupiter.Callisto",
    "Saturn",
    "Saturn.Enceladus",
    "Uranus",
    "Neptune",
    "Pluto",
    "Comet",
)
"""The SPASE Region vocabulary AMDA publishes as dataset targets (30 values on 8 435
products) plus the two the indexer's table uses and AMDA does not — the closed set a
classifier chooses from."""

REGION_INSTRUCTIONS = (
    "In which SPASE Region was this archived data product observed? The region is the body "
    "and domain the archive files the dataset under, decided by the mission and its orbit, "
    "not by the physical quantity. Spacecraft at L1 or upstream of Earth (Wind, ACE, DSCOVR, "
    "OMNI, SOHO) are Heliosphere.NearEarth; STEREO is Heliosphere.Remote1AU; Solar Orbiter, "
    "Parker Solar Probe, Helios, MESSENGER cruise and BepiColombo cruise are "
    "Heliosphere.Inner; Voyager and New Horizons beyond Saturn are Heliosphere.Outer; a "
    "cruise phase or an interplanetary monitor with no nearer body is Heliosphere. "
    "Earth-orbiting magnetospheric missions (Cluster, MMS, THEMIS, Geotail, Double Star, "
    "GOES) are Earth.Magnetosphere unless the dataset is explicitly a magnetosheath, "
    "magnetotail, polar or radiation-belt product; ground-based instruments (EISCAT, "
    "magnetometer stations, indices such as Kp, AE, Dst) are Earth or Earth.NearSurface.*. "
    "An orbiter of a planet is that planet (Juno, Galileo → Jupiter; Cassini → Saturn; "
    "MAVEN, Mars Express → Mars; Venus Express → Venus; MESSENGER in orbit → Mercury); a "
    "moon flyby product is Planet.Moon."
)
REGION_FLOOR = 0.9


JUDGMENT_RECORDS = "judgment_index.jsonl"
JUDGED_FILE = "judged_products.jsonl.gz"
SHIPPED_JUDGED = Path(__file__).parent / "data" / JUDGED_FILE
"""Every question the judge has been asked about a product, with its answer, shipped with
the package: one gzipped JSON line per product (`id`, `name`, then one `{choice,
confidence}` per question asked — `mtype`, `region`), after a first line of provenance
(`meta`: date, models, floors, count). It is the part of the index that cannot be rebuilt
from code — 82 266 requests, US$ 2.4 on 2026-09-22 — kept as the judge's raw answers, not
as decided fields, so the policy (floors, never overwriting a published label) lives in
code and can change without asking again. Abstentions are in it too: a question already
asked is not paid for twice. `helioai index` applies it to every product the archive left
untyped; `--classify` asks only what no record answers and appends to the local copy."""


def local_judged_path() -> Path:
    """Where `--classify` writes the answers it obtains: beside the data, not the index.

    The Chroma directory is wiped by `--rebuild`; the data root is not. A user with a key
    who classifies a new provider keeps those answers across every rebuild, and they take
    precedence over the shipped file for the same id.
    """
    from helioai.config import settings

    return Path(settings.data_dir) / JUDGED_FILE


def load_judged(*paths: Path) -> tuple[dict, dict[str, dict]]:
    """Read the judge's recorded answers from each file in turn, later files overriding
    earlier ones id by id; a missing or unreadable file contributes nothing.

    Returns:
        `(meta, records)` — the provenance of the last file read that had any, and
        `{id: {"name": …, "mtype": {"choice", "confidence"}, "region": {…}}}`.
    """
    import gzip
    import json

    meta: dict = {}
    records: dict[str, dict] = {}
    for path in paths:
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if "meta" in row:
                        meta = dict(row["meta"])
                        continue
                    pid = row.get("id")
                    if pid:
                        records[pid] = {k: v for k, v in row.items() if k != "id"}
        except (OSError, ValueError):
            continue
    return meta, records


def save_judged(path: Path, meta: dict, records: dict[str, dict]) -> None:
    """Write the answers as `load_judged` reads them, sorted by id, provenance first."""
    import gzip
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "asked": len(records)}
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for pid in sorted(records):
            f.write(json.dumps({"id": pid, **records[pid]}, ensure_ascii=False) + "\n")


def _answer(record: dict, question: str, floor: float) -> tuple[str | None, float]:
    """The recorded choice for a question when the judge was confident enough, else None."""
    raw = record.get(question)
    if not isinstance(raw, dict) or not raw.get("choice"):
        return None, 0.0
    confidence = float(raw.get("confidence") or 0.0)
    return (raw["choice"], confidence) if confidence >= floor else (None, confidence)


def _apply_record(doc: dict, record: dict) -> bool:
    """Write one product's recorded answers onto its freshly walked doc: fill the empty
    type, flag the contradicted label, replace the table's region — never touch what the
    archive publishes today. Returns whether anything was written."""
    meta, touched = doc["meta"], False
    label, confidence = _answer(record, "mtype", MEASUREMENT_TYPE_FLOOR)
    if label:
        published = meta.get("measurement_type")
        if not published:
            meta["measurement_type"] = label
            meta["measurement_type_source"] = "jev"
            meta["measurement_type_confidence"] = round(confidence, 3)
            doc["text"] = _before_coverage(doc["text"], f"Measurement: {label}.")
            touched = True
        elif _normalised_type(published) != _normalised_type(label):
            meta["measurement_type_jev"] = label
            meta["measurement_type_jev_confidence"] = round(confidence, 3)
            touched = True
    region, confidence = _answer(record, "region", REGION_FLOOR)
    if region and meta.get("region_source") != "archive":
        meta["region"] = region
        meta["region_source"] = "jev"
        meta["region_confidence"] = round(confidence, 3)
        doc["text"] = _with_region_sentence(doc["text"], region)
        touched = True
    return touched


def apply_judged(docs: list[dict], judged: dict[str, dict]) -> int:
    """Apply the recorded answers to every walked doc they name.

    A record whose `name` no longer matches the product's is skipped: the id was reused
    for something else, and a type decided about the old content is not evidence about
    the new. A record without a name (none of the first pass had one) is applied.

    Returns:
        How many docs received at least one field.
    """
    applied = 0
    for doc in docs:
        record = judged.get(doc["id"])
        if not record:
            continue
        name = record.get("name")
        if name and doc["meta"].get("name") and name != doc["meta"]["name"]:
            continue
        applied += _apply_record(doc, record)
    return applied


def _normalised_type(label: str | None) -> str:
    return "".join(ch for ch in str(label or "").split(",")[0].lower() if ch.isalnum())


def _before_coverage(text: str, sentence: str) -> str:
    """The text with one more sentence, placed before the coverage sentence — the dates
    close every indexed text, as `_build_text` writes it."""
    head, sep, tail = text.strip().partition(" Coverage: ")
    return f"{head} {sentence}{sep}{tail}" if sep else f"{head} {sentence}"


def _with_region_sentence(text: str, region: str) -> str:
    """The text with `Region: X.` said once, where the table's guess used to be."""
    return _before_coverage(_REGION_SENTENCE.sub("", text), f"Region: {region}.")


async def classify_products(
    docs: list[dict],
    record_dir: Path | str,
    *,
    judged: dict[str, dict] | None = None,
    judged_meta: dict | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Ask the judge what no record has answered yet, apply it, and keep the answers.

    Two closed questions per product: the SPASE measurement type where the archive left it
    empty, and the SPASE region where the indexer had only guessed. Neither answer ever
    overwrites anything the archive published.

    **Measurement type.** The field is indexed on 15.6 % of the products — AMDA and CSA —
    and on none of CDA's 68 000, so every ranking signal built on it (`_rerank_penalty`)
    and every filter reaches a sixth of the catalogue. Measured on 2026-09-22 against 200
    products the archive had labelled, label stripped from the text before asking: 76.5 %
    agreement, **89 % where the judge's confidence is at least 0.9** (72 % of the items) —
    and the remaining confident disagreements were the archive's errors (MMS FPI plasma
    moments labelled MagneticField, a JADE density labelled EnergeticParticles, a
    Langmuir-probe density labelled ElectricField). So the judge is better than its ground
    truth, and the floor is 0.9: below it the field stays empty — abstention is a type —
    and a published label the judge contradicts at or above it is kept and flagged as
    `measurement_type_jev`, for a person to adjudicate, never replaced.

    **Region.** `_get_region` guesses from a 40-entry table matched as a substring; against
    AMDA's 8 435 published targets it agrees on 26.9 %, is silent on 41 % and wrong on 32 %
    ("ac" inside "cce_mepa_ion_act" made AMPTE/CCE a near-Earth heliospheric product).
    Measured the same day on 200 of those products, target stripped: the judge agrees
    exactly on 70 %, **on the body (Earth, Jupiter, Heliosphere…) on 97.1 % at confidence
    ≥ 0.9**, and where judge and table differ the judge is right 75 times to the table's
    one. Its confident disagreements with the archive are granularity, in both directions
    (Helios filed as Heliosphere, a Galileo Io flyby read as Jupiter), so a published target
    is never flagged — it stands. The table's guess is not a publication: at or above the
    floor the judge's region replaces it, or fills the silence, with `region_source: "jev"`
    and the confidence; below it the guess stays, marked as the guess it is.

    **What is asked.** A product is asked only the questions no record in `judged` answers
    for it — the shipped file plus the local one — so a second pass over the same
    catalogue costs nothing, a new provider costs its own products, and adding a question
    costs one request per product for that question alone. Both sentences the text carried
    are stripped before asking, so the judge reads the product, not the labels. Every call
    is recorded to `judgment_index.jsonl` in the index directory with the product id as its
    key, and every answer — abstentions included — is appended to the local
    `judged_products.jsonl.gz` for the next rebuild. ~2.9 ¢ per 1 000 requests (metered
    2026-09-22: 838 tokens a request, the instruction being most of it).

    Args:
        docs: `{id, text, meta}` as `_walk` collects them.
        record_dir: Where the calls are recorded (the Chroma directory).
        judged: The answers already on record, updated in place.
        judged_meta: Their provenance, carried into the saved file.
        verbose: Print the counts.

    Returns:
        The same docs, metadata and text amended in place.
    """
    from helioai.config import settings
    from helioai.core import judgment

    judged = {} if judged is None else judged
    if settings.judgment.backend == "null":
        if verbose:
            print(
                "[indexer] --classify needs HELIOAI_JUDGMENT_BACKEND=jev and TYPESAFE_API_KEY; "
                "skipping classification"
            )
        return docs
    questions = {
        "mtype": judgment.Choice(
            MEASUREMENT_TYPE_INSTRUCTIONS, MEASUREMENT_TYPES, floor=MEASUREMENT_TYPE_FLOOR
        ),
        "region": judgment.Choice(REGION_INSTRUCTIONS, REGIONS, floor=REGION_FLOOR),
    }
    groups: dict[tuple[str, ...], list[dict]] = {}
    for doc in docs:
        record = judged.get(doc["id"]) or {}
        missing = tuple(q for q in questions if q not in record)
        if missing:
            groups.setdefault(missing, []).append(doc)
    n_requests = sum(len(g) for g in groups.values())
    if verbose:
        print(
            f"[indexer] {len(docs) - n_requests} products fully on record; "
            f"asking {n_requests} requests: "
            + ", ".join(f"{len(g)} × {'+'.join(k)}" for k, g in groups.items())
        )
    models: set[str] = set(judged_meta.get("models") or []) if judged_meta else set()
    filled = flagged = 0
    for missing, group in groups.items():
        asked = {q: questions[q] for q in missing}
        states = [
            {"product": _REGION_SENTENCE.sub("", _LABEL_SENTENCE.sub(" ", d["text"])).strip()}
            for d in group
        ]
        answers = await judgment.batch(
            "index_classify",
            states,
            asked,
            record_to=Path(record_dir) / JUDGMENT_RECORDS,
            keys=[d["id"] for d in group],
        )
        for doc, answer in zip(group, answers, strict=True):
            if answer is None:
                continue
            record = judged.setdefault(doc["id"], {})
            record["name"] = doc["meta"].get("name") or record.get("name")
            for q in missing:
                raw = answer.raw.get(q) or {}
                if raw.get("choice") is not None:
                    record[q] = {
                        "choice": raw["choice"],
                        "confidence": round(float(raw.get("confidence") or 0.0), 3),
                    }
            models.add(answer.model)
            before = doc["meta"].get("measurement_type")
            if _apply_record(doc, record):
                if doc["meta"].get("measurement_type_source") == "jev" and not before:
                    filled += 1
                if "measurement_type_jev" in doc["meta"]:
                    flagged += 1
    if n_requests:
        meta = {
            **(judged_meta or {}),
            "date": time.strftime("%Y-%m-%d"),
            "models": sorted(models),
            "floors": {"mtype": MEASUREMENT_TYPE_FLOOR, "region": REGION_FLOOR},
        }
        save_judged(local_judged_path(), meta, judged)
    if verbose:
        print(
            f"[indexer] this pass: {filled} types filled, {flagged} published labels flagged, "
            f"{n_requests} requests, answers saved to {local_judged_path()}"
        )
    return docs


HNSW_SYNC_THRESHOLD = 1
HNSW_EF_SEARCH = 400


def open_collections(
    chroma_dir: Path | str, names: list[str], *, verbose: bool = False
) -> tuple[Any, list[Any]]:
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
        components = _cda_components(child_vars) if provider_prefix == "cda" else []

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
                    components=components,
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
                        meta_entry["region_source"] = (
                            "archive" if (parent_meta or {}).get("region") else "table"
                        )
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
        meta["region_source"] = "table"
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
    components: list[str] | None = None,
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
        ("SPASE", "spase"),
        ("Observatory", "observatory"),
        ("Instrument", "experiments"),
        ("Processing", "processing_level"),
        ("Cadence", "cadence"),
    ):
        value = meta.get(key) or ""
        if value:
            parts.append(f"{label}: {value}.")
    dataset_desc = meta.get("dataset_description") or ""
    if dataset_desc:
        parts.append(f"{dataset_desc}.")
    if components:
        parts.append(f"Components: {', '.join(components)}.")
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
