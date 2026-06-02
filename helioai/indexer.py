"""Build the speasy catalog ChromaDB index.

Usage:
    helioai index           # incremental (skip existing)
    helioai index --rebuild # wipe and rebuild
    python indexer.py       # direct call from repo root
"""

from __future__ import annotations

import html
import re
import shutil
import time


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


def build_index(rebuild: bool = False, batch_size: int = 128, verbose: bool = True) -> int:
    """Walk the speasy inventory and index all parameters into ChromaDB."""
    try:
        import chromadb
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

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
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

    return total


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
                region = _get_region(uid)
                text = _build_text(
                    name,
                    description,
                    units,
                    xmlid,
                    parent_meta=parent_meta,
                    entity=entity,
                    prop=prop,
                    region=region,
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
                    out.append({"id": uid, "text": text, "meta": meta_entry})

        _walk(child, provider_prefix, out, skip_ids, SpeasyIndex, depth + 1, max_docs, parent_meta)


def _build_text(
    name: str,
    description: str,
    units: str,
    xmlid: str,
    parent_meta: dict | None = None,
    entity: str = "",
    prop: str = "",
    region: str = "",
) -> str:
    head = name if name != xmlid else xmlid.replace("_", " ")
    parts = [f"{head}."]
    if description:
        parts.append(f"{description}.")
    if entity and prop:
        parts.append(f"{entity} {prop}.")
    elif entity:
        parts.append(f"Particle: {entity}.")
    mtype = (parent_meta or {}).get("measurement_type") or ""
    if mtype:
        parts.append(f"Measurement: {mtype}.")
    category = (parent_meta or {}).get("category") or ""
    if category:
        parts.append(f"Category: {category}.")
    dataset_desc = (parent_meta or {}).get("dataset_description") or ""
    if dataset_desc:
        parts.append(f"{dataset_desc}.")
    if units:
        parts.append(f"Units: {units}.")
    if region:
        parts.append(f"Region: {region}.")
    return " ".join(parts)
