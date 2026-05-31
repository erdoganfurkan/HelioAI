"""ChromaDB semantic search over the speasy catalog.

The index is built by `indexer.py` (run: helioai index).
This module provides the read-only search path used at agent query time.
Models load lazily and are cached at module scope.
"""

from __future__ import annotations

import logging
import math
import threading

from helioai.config import settings

log = logging.getLogger(__name__)

_model = None
_reranker = None
_reranker_loaded = False
_collection = None
_lock = threading.Lock()


def _load():
    global _model, _collection
    if _model is not None and _collection is not None:
        return _model, _collection

    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(settings.rag.embed_model)
        if _collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=str(settings.rag.chroma_dir))
            _collection = client.get_collection(name=settings.rag.collection_name)

    return _model, _collection


def _load_reranker():
    global _reranker, _reranker_loaded
    if not settings.rag.rerank_enabled:
        return None
    if _reranker_loaded:
        return _reranker
    with _lock:
        if not _reranker_loaded:
            try:
                from sentence_transformers import CrossEncoder
                _reranker = CrossEncoder(settings.rag.rerank_model)
            except Exception as e:
                log.warning("reranker %s unavailable (%s)", settings.rag.rerank_model, e)
                _reranker = None
            _reranker_loaded = True
    return _reranker


def _truncate(text: str, max_chars: int = 280) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 1].rstrip() + "…"


def search(query: str, top_k: int = 5) -> list[dict]:
    """Semantic search over speasy catalog.

    Returns a list of dicts: {id, name, description, score}
    Raises if the index has not been built yet (run `helioai index`).
    """
    if not query or not query.strip():
        return []

    model, collection = _load()

    query_vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    fetch_k = max(top_k, settings.rag.rerank_fetch_k) if settings.rag.rerank_enabled else top_k

    res = collection.query(
        query_embeddings=query_vec,
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = res.get("ids", [[]])[0]
    documents = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]

    if not ids:
        return []

    candidates: list[dict] = []
    for pid, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
        similarity = 1.0 - float(dist)
        bi_score = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
        candidates.append({
            "id": pid,
            "name": (meta or {}).get("name", "") or pid,
            "description": _truncate(doc_text or ""),
            "_full_text": doc_text or "",
            "score": round(bi_score, 4),
        })

    reranker = _load_reranker()
    if reranker is not None and len(candidates) > 1:
        pairs = [(query, c["_full_text"]) for c in candidates]
        rerank_scores = reranker.predict(pairs)
        for c, s in zip(candidates, rerank_scores):
            c["score"] = round(1.0 / (1.0 + math.exp(-float(s))), 4)
        candidates.sort(key=lambda c: c["score"], reverse=True)

    for c in candidates:
        c.pop("_full_text", None)

    return candidates[:top_k]
