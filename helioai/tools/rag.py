"""ChromaDB semantic search over the speasy catalog.

The index is built by `indexer.py` (run: helioai index).
This module provides the read-only search path used at agent query time.
Models load lazily and are cached at module scope.
"""

from __future__ import annotations

import logging
import math
import re
import threading

from helioai.config import settings

log = logging.getLogger(__name__)

_model = None
_reranker = None
_reranker_loaded = False
_collection = None
_lock = threading.Lock()

_bm25 = None
_bm25_ids: list[str] = []
_bm25_meta: list[dict] = []
_bm25_docs: list[str] = []
_bm25_loaded = False

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase token split on non-alphanumerics.

    `BGSEc` → ['bgsec'], `mms1_dis_numberdensity_fast` → ['mms1','dis','numberdensity','fast'].
    The query is tokenised the same way, so typing an exact id/code matches.
    """
    return _TOKEN_RE.findall(text.lower())


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

            if not settings.rag.chroma_dir.exists():
                raise RuntimeError(
                    f"ChromaDB index not found at {settings.rag.chroma_dir}. "
                    "Run `helioai index` first to build the parameter catalog."
                )
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


def _load_bm25():
    """Build (once, cached) a BM25 index over the documents already in Chroma.

    No indexer change needed: the corpus is pulled from the existing collection.
    Tokenises `id + document` so exact id/code queries match. Returns None if
    rank_bm25 is missing or the collection is empty (search degrades to dense).
    """
    global _bm25, _bm25_ids, _bm25_meta, _bm25_docs, _bm25_loaded
    if _bm25_loaded:
        return _bm25
    with _lock:
        if not _bm25_loaded:
            try:
                from rank_bm25 import BM25Okapi

                _, collection = _load()
                # Paginate: a single get() over the whole catalog blows SQLite's
                # variable limit on large collections (~83k).
                _bm25_ids, _bm25_docs, _bm25_meta = [], [], []
                total = collection.count()
                page = 5000
                for off in range(0, total, page):
                    data = collection.get(
                        include=["documents", "metadatas"], limit=page, offset=off
                    )
                    _bm25_ids.extend(data.get("ids", []) or [])
                    _bm25_docs.extend(data.get("documents", []) or [])
                    _bm25_meta.extend(data.get("metadatas", []) or [])
                corpus = [
                    _tokenize(f"{pid} {doc or ''}") for pid, doc in zip(_bm25_ids, _bm25_docs)
                ]
                _bm25 = BM25Okapi(corpus) if corpus else None
            except Exception as e:
                log.warning("BM25 unavailable (%s) — falling back to dense-only", e)
                _bm25 = None
            _bm25_loaded = True
    return _bm25


def _rrf_fuse(rankings: list[list[str]], k: int) -> dict[str, float]:
    """Reciprocal Rank Fusion: sum of 1/(k + rank) across each ranked id list."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _meta_matches(meta: dict, provider, region, measurement_type) -> bool:
    if provider and (meta or {}).get("provider") != provider:
        return False
    if region and (meta or {}).get("region") != region:
        return False
    if measurement_type and (meta or {}).get("measurement_type") != measurement_type:
        return False
    return True


def _truncate(text: str, max_chars: int = 280) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _build_where(
    provider: str | None, region: str | None, measurement_type: str | None
) -> dict | None:
    """Build a ChromaDB metadata filter. One condition → flat dict, ≥2 → $and."""
    conds: list[dict] = []
    if provider:
        conds.append({"provider": provider})
    if region:
        conds.append({"region": region})
    if measurement_type:
        conds.append({"measurement_type": measurement_type})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def _fuse_query(
    query: str, dense_hit: tuple, top_k: int, *, provider, region, measurement_type, hybrid: bool
) -> list[dict]:
    """Run the hybrid fusion + scoring for ONE query given its dense hit.

    `dense_hit` is (ids, documents, metadatas, distances) for this query.
    Shared by `search` (single) and `search_batch` (multi) so the pipeline is
    defined once.
    """
    ids, documents, metadatas, distances = dense_hit

    # info[id] = {name, full_text, cosine}; dense_ranking preserves order
    info: dict[str, dict] = {}
    dense_ranking: list[str] = []
    for pid, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
        dense_ranking.append(pid)
        similarity = 1.0 - float(dist)
        info[pid] = {
            "name": (meta or {}).get("name", "") or pid,
            "full_text": doc_text or "",
            "cosine": max(0.0, min(1.0, (similarity + 1.0) / 2.0)),
        }

    # Sparse (BM25) channel — exact token / id matching
    sparse_ranking: list[str] = []
    bm25 = _load_bm25() if hybrid else None
    if bm25 is not None:
        scores = bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for idx in order:
            if scores[idx] <= 0:
                break
            meta = _bm25_meta[idx] if idx < len(_bm25_meta) else {}
            if not _meta_matches(meta, provider, region, measurement_type):
                continue
            pid = _bm25_ids[idx]
            sparse_ranking.append(pid)
            if pid not in info:
                info[pid] = {
                    "name": (meta or {}).get("name", "") or pid,
                    "full_text": _bm25_docs[idx] if idx < len(_bm25_docs) else "",
                    "cosine": 0.0,
                }
            if len(sparse_ranking) >= settings.rag.hybrid_fetch_k:
                break

    if sparse_ranking:
        fused = _rrf_fuse([dense_ranking, sparse_ranking], settings.rag.rrf_k)
        ordered_ids = sorted(fused, key=lambda p: fused[p], reverse=True)
        raw = fused
        score_mode = "rrf"
    else:
        ordered_ids = dense_ranking
        raw = {p: info[p]["cosine"] for p in dense_ranking}
        score_mode = "cosine"

    if not ordered_ids:
        return []

    candidates: list[dict] = [
        {
            "id": pid,
            "name": info[pid]["name"],
            "description": _truncate(info[pid]["full_text"]),
            "_full_text": info[pid]["full_text"],
            "_raw": raw.get(pid, 0.0),
        }
        for pid in ordered_ids
    ]

    reranker = _load_reranker()
    if reranker is not None and len(candidates) > 1:
        head = candidates[: max(top_k, settings.rag.rerank_fetch_k)]
        rerank_scores = reranker.predict([(query, c["_full_text"]) for c in head])
        for c, s in zip(head, rerank_scores):
            c["score"] = round(1.0 / (1.0 + math.exp(-float(s))), 4)
        head.sort(key=lambda c: c["score"], reverse=True)
        candidates = head
    elif score_mode == "rrf":
        raws = [c["_raw"] for c in candidates]
        lo, hi = min(raws), max(raws)
        span = (hi - lo) or 1.0
        for c in candidates:
            c["score"] = round((c["_raw"] - lo) / span, 4) if hi > lo else 1.0
    else:
        for c in candidates:
            c["score"] = round(c["_raw"], 4)

    for c in candidates:
        c.pop("_full_text", None)
        c.pop("_raw", None)

    return candidates[:top_k]


def search_batch(
    queries: list[str],
    top_k: int = 5,
    *,
    provider: str | None = None,
    region: str | None = None,
    measurement_type: str | None = None,
) -> list[list[dict]]:
    """Resolve several queries in ONE pass — the 'composed RAG'.

    Encodes all queries in a single embedding pass and issues a single
    multi-vector ChromaDB query (Chroma returns one result set per query
    natively), then fuses each independently. Returns one result list per input
    query, aligned by index (blank queries map to []).
    """
    results: list[list[dict]] = [[] for _ in queries]
    active = [(i, q) for i, q in enumerate(queries) if q and q.strip()]
    if not active:
        return results

    model, collection = _load()
    hybrid = settings.rag.hybrid_enabled
    if hybrid:
        dense_k = settings.rag.hybrid_fetch_k
    elif settings.rag.rerank_enabled:
        dense_k = max(top_k, settings.rag.rerank_fetch_k)
    else:
        dense_k = top_k

    vecs = model.encode(
        [q for _, q in active],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    res = collection.query(
        query_embeddings=vecs,
        n_results=dense_k,
        where=_build_where(provider, region, measurement_type),
        include=["documents", "metadatas", "distances"],
    )
    all_ids = res.get("ids", []) or []
    all_docs = res.get("documents", []) or []
    all_metas = res.get("metadatas", []) or []
    all_dists = res.get("distances", []) or []

    for j, (i, q) in enumerate(active):
        dense_hit = (
            all_ids[j] if j < len(all_ids) else [],
            all_docs[j] if j < len(all_docs) else [],
            all_metas[j] if j < len(all_metas) else [],
            all_dists[j] if j < len(all_dists) else [],
        )
        results[i] = _fuse_query(
            q,
            dense_hit,
            top_k,
            provider=provider,
            region=region,
            measurement_type=measurement_type,
            hybrid=hybrid,
        )
    return results


def search_catalogs(
    query: str,
    top_k: int = 5,
    *,
    product_type: str | None = None,
) -> list[dict]:
    """Semantic search over the AMDA catalog/timetable index.

    `product_type` can be 'catalog', 'timetable', or None (both).
    Returns {id, name, description, score, nb_events, product_type}.
    Requires `helioai index` to have been run at least once.
    Falls back to an empty list if the catalog collection is absent.
    """
    if not query or not query.strip():
        return []
    try:
        import chromadb

        model, _ = _load()  # reuse the already-loaded embedding model
        client = chromadb.PersistentClient(path=str(settings.rag.chroma_dir))
        col = client.get_collection(name=settings.rag.catalogs_collection_name)
    except Exception as e:
        log.debug("catalog collection unavailable (%s)", e)
        return []

    try:
        vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).tolist()
        where: dict | None = {"product_type": product_type} if product_type else None
        res = col.query(
            query_embeddings=vec,
            n_results=min(top_k, col.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        results: list[dict] = []
        for pid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            score = round(max(0.0, min(1.0, (1.0 - float(dist) + 1.0) / 2.0)), 4)
            results.append(
                {
                    "id": pid,
                    "name": (meta or {}).get("name", pid),
                    "description": _truncate(doc or ""),
                    "score": score,
                    "nb_events": (meta or {}).get("nb_events", 0),
                    "product_type": (meta or {}).get("product_type", ""),
                }
            )
        return results
    except Exception as e:
        log.warning("catalog search failed: %s", e)
        return []


def search(
    query: str,
    top_k: int = 5,
    *,
    provider: str | None = None,
    region: str | None = None,
    measurement_type: str | None = None,
) -> list[dict]:
    """Semantic search over speasy catalog (single query).

    Optional metadata filters narrow the search at query time (the metadata is
    already indexed by indexer.py): `provider` (amda/cda/csa/ssc) is the most
    useful — it counters CDA's dominance of the catalog.

    With hybrid search enabled (default), a BM25 lexical channel is fused with
    the dense channel via RRF — this recovers exact id/code matches (e.g.
    `BGSEc`) that dense embeddings miss. When the cross-encoder reranker is off,
    `score` is a RELATIVE confidence in [0,1] (top≈1.0); with the reranker on it
    is the absolute sigmoid score.

    For several queries at once use `search_batch` (one embedding pass + one
    Chroma call). Returns a list of dicts: {id, name, description, score}.
    """
    if not query or not query.strip():
        return []
    return search_batch(
        [query], top_k, provider=provider, region=region, measurement_type=measurement_type
    )[0]
