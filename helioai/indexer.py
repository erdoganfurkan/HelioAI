"""Build the speasy catalog ChromaDB index.

Usage:
    helioai index           # incremental (skip existing)
    helioai index --rebuild # wipe and rebuild
    python indexer.py       # direct call from repo root
"""

from __future__ import annotations

import shutil
import time


def build_index(rebuild: bool = False, batch_size: int = 128, verbose: bool = True) -> int:
    """Walk the speasy inventory and index all parameters into ChromaDB.

    Returns the number of documents indexed.
    """
    try:
        import chromadb
        import speasy as spz
        from sentence_transformers import SentenceTransformer
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
                print(f"[indexer] {len(existing_ids)} existing — skipping")
        except Exception:
            pass

    if verbose:
        print("[indexer] walking speasy inventory…")

    docs: list[dict] = []
    _walk_inventory(spz.inventories.tree, docs, existing_ids, max_docs=100_000)

    if verbose:
        print(f"[indexer] found {len(docs)} new parameters to index")

    if not docs:
        if verbose:
            print("[indexer] nothing to index — up to date")
        return 0

    t0 = time.perf_counter()
    total = 0

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
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


def _walk_inventory(node, out: list[dict], skip_ids: set[str], depth: int = 0, max_docs: int = 100_000) -> None:
    if len(out) >= max_docs or depth > 10:
        return

    for attr in dir(node):
        if attr.startswith("_") or len(out) >= max_docs:
            continue
        try:
            child = getattr(node, attr, None)
        except Exception:
            continue
        if child is None:
            continue

        uid = getattr(child, "uid", None) or getattr(child, "spz_uid", None)
        name = getattr(child, "name", attr) or attr
        desc = getattr(child, "desc", "") or ""
        units = getattr(child, "units", "") or ""

        if uid and str(uid) not in skip_ids:
            text = _build_text(str(name), str(desc), str(units), attr)
            if text.strip():
                out.append({
                    "id": str(uid),
                    "text": text,
                    "meta": {"name": str(name), "units": str(units), "attr": attr},
                })
        elif not uid and depth < 8:
            _walk_inventory(child, out, skip_ids, depth + 1, max_docs)


def _build_text(name: str, desc: str, units: str, attr: str) -> str:
    parts: list[str] = []
    head = name if name != attr else attr.replace("_", " ")
    if desc:
        parts.append(f"{head}. {desc}.")
    else:
        parts.append(f"{head}.")
    if units:
        parts.append(f"Units: {units}.")
    return " ".join(parts)
