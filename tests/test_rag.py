"""Tests for helioai.tools.rag.search using in-memory ChromaDB."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import helioai.tools.rag as rag_module
from helioai.tools.rag import search, search_batch


@pytest.fixture(autouse=True)
def isolated_rag(monkeypatch, fake_embed_model, tmp_path):
    """Replace the global model and collection with per-test isolated stubs."""
    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.get_or_create_collection(
        name="test_params",
        metadata={"hnsw:space": "cosine"},
    )

    monkeypatch.setattr(rag_module, "_model", fake_embed_model)
    monkeypatch.setattr(rag_module, "_collection", collection)
    # Reset the BM25 cache so each test rebuilds from its own collection
    monkeypatch.setattr(rag_module, "_bm25", None)
    monkeypatch.setattr(rag_module, "_bm25_loaded", False)
    # Reset search & catalog caches so tests are independent
    monkeypatch.setattr(rag_module, "_search_cache", {})
    monkeypatch.setattr(rag_module, "_catalog_collection", None)

    yield collection


def _seed(collection, n: int = 10) -> list[str]:
    """Insert n synthetic docs and return their ids."""
    rng = np.random.default_rng(0)
    ids = [f"param_{i}" for i in range(n)]
    vecs = rng.random((n, 128)).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-9)
    docs = [f"Parameter {i}: solar wind measurement. Units: nT." for i in range(n)]
    metas = [
        {"name": f"P{i}", "units": "nT", "xmlid": f"param_{i}", "provider": "amda"}
        for i in range(n)
    ]
    collection.add(ids=ids, embeddings=vecs.tolist(), documents=docs, metadatas=metas)
    return ids


def _seed_mixed(collection) -> None:
    """Insert docs split across two providers (amda/<id>, cda/<id>)."""
    rng = np.random.default_rng(1)
    rows = [("amda/a0", "amda"), ("amda/a1", "amda"), ("cda/c0", "cda"), ("cda/c1", "cda")]
    vecs = rng.random((len(rows), 128)).astype("float32")
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    collection.add(
        ids=[r[0] for r in rows],
        embeddings=vecs.tolist(),
        documents=["solar wind magnetic field measurement nT" for _ in rows],
        metadatas=[{"name": r[0], "provider": r[1], "xmlid": r[0]} for r in rows],
    )


def test_search_returns_top_k(isolated_rag) -> None:
    _seed(isolated_rag)
    results = search("solar wind density", top_k=3)
    assert len(results) == 3


def test_search_result_shape(isolated_rag) -> None:
    _seed(isolated_rag)
    results = search("magnetic field ACE", top_k=1)
    assert len(results) == 1
    r = results[0]
    assert "id" in r
    assert "name" in r
    assert "description" in r
    assert "score" in r
    assert 0.0 <= r["score"] <= 1.0


def test_search_empty_query_returns_empty(isolated_rag) -> None:
    _seed(isolated_rag)
    assert search("") == []
    assert search("   ") == []


def test_search_empty_collection_returns_empty(isolated_rag) -> None:
    results = search("solar wind density", top_k=5)
    assert results == []


def test_search_scores_are_sorted_descending(isolated_rag) -> None:
    _seed(isolated_rag, n=10)
    results = search("solar wind measurement", top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_top_k_respected_when_collection_has_more(isolated_rag) -> None:
    _seed(isolated_rag, n=10)
    results = search("plasma", top_k=2)
    assert len(results) <= 2


def test_search_provider_filter(isolated_rag) -> None:
    """Filtered hits come first; out-of-filter hits are appended and flagged.

    The filter used to be purely exclusive. It is now additive, so this asserts
    the stronger property: nothing from another provider can appear *unmarked*.
    """
    _seed_mixed(isolated_rag)
    results = search("solar wind magnetic field", top_k=10, provider="amda")
    assert results
    in_filter = [r for r in results if not r.get("outside_filter")]
    assert in_filter, "the requested provider must still be searched"
    assert all(r["id"].startswith("amda/") for r in in_filter)
    assert all(not r["id"].startswith("amda/") for r in results if r.get("outside_filter")), (
        "a flagged hit should genuinely be from another provider"
    )


def test_search_no_filter_returns_all_providers(isolated_rag) -> None:
    _seed_mixed(isolated_rag)
    results = search("solar wind magnetic field", top_k=10)
    providers = {r["id"].split("/")[0] for r in results}
    assert providers == {"amda", "cda"}


# ──────────────────────────────── hybrid search ─────────────────────────────


def test_rrf_fuse_rewards_agreement() -> None:
    from helioai.tools.rag import _rrf_fuse

    # 'b' is #1 in one list and #1 in the other; 'a' only appears once
    fused = _rrf_fuse([["a", "b"], ["b", "c"]], k=60)
    assert fused["b"] > fused["a"]
    assert fused["b"] > fused["c"]


def _seed_with_code(collection) -> None:
    """One doc carries a rare exact code token; dense (random) embeddings miss it."""
    rng = np.random.default_rng(7)
    ids = [f"cda/PARAM_{i}/generic" for i in range(8)] + ["cda/ACE_H0_MFI/BGSEc"]
    vecs = rng.random((len(ids), 128)).astype("float32")
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    docs = ["generic plasma measurement" for _ in range(8)] + ["ACE magnetic field GSE component"]
    metas = [{"name": pid, "provider": "cda", "xmlid": pid} for pid in ids]
    collection.add(ids=ids, embeddings=vecs.tolist(), documents=docs, metadatas=metas)


def test_hybrid_finds_exact_code(isolated_rag, monkeypatch) -> None:
    from helioai.config import settings

    monkeypatch.setattr(settings.rag, "hybrid_enabled", True)
    _seed_with_code(isolated_rag)
    results = search("BGSEc", top_k=5)
    assert any(r["id"] == "cda/ACE_H0_MFI/BGSEc" for r in results)
    assert results[0]["id"] == "cda/ACE_H0_MFI/BGSEc"


def test_hybrid_disabled_is_dense_only(isolated_rag, monkeypatch) -> None:
    from helioai.config import settings

    monkeypatch.setattr(settings.rag, "hybrid_enabled", False)
    _seed_with_code(isolated_rag)
    # With hybrid off and random dense embeddings, the exact code is not reliably #1
    results = search("BGSEc", top_k=5)
    assert len(results) == 5  # dense still returns top_k


def test_hybrid_respects_provider_filter(isolated_rag, monkeypatch) -> None:
    """The filter still holds — anything outside it is appended and flagged.

    Tightened when the filter became additive: rather than "every result is from
    the provider", the contract is now "every result is from the provider, or is
    explicitly marked as not being". A silent leak would fail this too.
    """
    from helioai.config import settings

    monkeypatch.setattr(settings.rag, "hybrid_enabled", True)
    _seed_mixed(isolated_rag)  # amda/* and cda/*
    results = search("solar wind magnetic field", top_k=10, provider="amda")
    assert results
    for r in results:
        assert r["id"].startswith("amda/") or r.get("outside_filter") is True
    assert results[0]["id"].startswith("amda/")


# ──────────────────────────────── batch search ──────────────────────────────


def test_search_batch_groups(isolated_rag) -> None:
    _seed(isolated_rag, n=10)
    groups = search_batch(["solar wind density", "plasma measurement"], top_k=3)
    assert len(groups) == 2
    assert all(len(g) <= 3 for g in groups)
    assert all("id" in r and "score" in r for g in groups for r in g)


def test_search_batch_single_pass(isolated_rag, monkeypatch) -> None:
    _seed(isolated_rag, n=6)
    calls = {"n": 0}
    real_encode = rag_module._model.encode

    def counting_encode(texts, **kwargs):
        calls["n"] += 1
        return real_encode(texts, **kwargs)

    monkeypatch.setattr(rag_module._model, "encode", counting_encode)
    search_batch(["q1", "q2", "q3"], top_k=2)
    assert calls["n"] == 1  # one embedding pass for the whole batch


def test_search_equals_batch_of_one(isolated_rag) -> None:
    _seed(isolated_rag, n=8)
    single = search("solar wind density", top_k=3)
    batched = search_batch(["solar wind density"], top_k=3)[0]
    assert [r["id"] for r in single] == [r["id"] for r in batched]


def test_search_batch_empty_slot(isolated_rag) -> None:
    _seed(isolated_rag, n=6)
    groups = search_batch(["solar wind density", "   ", "plasma"], top_k=2)
    assert len(groups) == 3
    assert groups[1] == []
    assert groups[0] and groups[2]


# ──────────────────────────── reranker composition ──────────────────────────


# ─────────────────────────── search cache ──────────────────────────────────


def test_search_cache_skips_encode_on_repeat(isolated_rag, monkeypatch) -> None:
    _seed(isolated_rag)
    calls = {"n": 0}
    real_encode = rag_module._model.encode

    def counting_encode(texts, **kwargs):
        calls["n"] += 1
        return real_encode(texts, **kwargs)

    monkeypatch.setattr(rag_module._model, "encode", counting_encode)
    r1 = search("solar wind density", top_k=3)
    r2 = search("solar wind density", top_k=3)

    assert calls["n"] == 1
    assert [r["id"] for r in r1] == [r["id"] for r in r2]


def test_search_batch_partial_cache(isolated_rag, monkeypatch) -> None:
    _seed(isolated_rag)
    encoded_batches: list[list] = []
    real_encode = rag_module._model.encode

    def tracking_encode(texts, **kwargs):
        encoded_batches.append(list(texts))
        return real_encode(texts, **kwargs)

    monkeypatch.setattr(rag_module._model, "encode", tracking_encode)

    search_batch(["q_a", "q_b"], top_k=2)
    assert encoded_batches[-1] == ["q_a", "q_b"]

    search_batch(["q_a", "q_new"], top_k=2)  # q_a is now cached
    assert encoded_batches[-1] == ["q_new"]  # only the uncached query was encoded


def test_search_catalogs_memoizes_collection(isolated_rag, monkeypatch) -> None:
    import chromadb

    from helioai.tools.rag import search_catalogs

    fake_col = MagicMock()
    fake_col.count.return_value = 0
    fake_col.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    fake_client = MagicMock()
    fake_client.get_collection.return_value = fake_col
    client_call_count = [0]

    def fake_persistent_client(*args, **kwargs):
        client_call_count[0] += 1
        return fake_client

    monkeypatch.setattr(chromadb, "PersistentClient", fake_persistent_client)

    search_catalogs("ICME solar wind")
    search_catalogs("bow shock MMS")

    assert client_call_count[0] == 1
    assert fake_client.get_collection.call_count == 1


# ──────────────────────────── reranker composition ──────────────────────────


def test_reranker_composes_with_hybrid_and_batch(isolated_rag, monkeypatch) -> None:
    """Enabling the cross-encoder reranker must re-rank the fused candidates,
    per query, in batch mode — and produce absolute sigmoid scores in [0,1]."""
    from helioai.config import settings

    monkeypatch.setattr(settings.rag, "hybrid_enabled", True)
    monkeypatch.setattr(settings.rag, "rerank_enabled", True)
    _seed(isolated_rag, n=8)  # docs "Parameter i: ...", ids param_i

    class FakeReranker:
        def predict(self, pairs):
            # Promote the doc that mentions 'Parameter 5' to the top
            return [10.0 if "Parameter 5:" in doc else -10.0 for _q, doc in pairs]

    monkeypatch.setattr(rag_module, "_reranker", FakeReranker())
    monkeypatch.setattr(rag_module, "_reranker_loaded", True)

    groups = search_batch(["solar wind", "plasma measurement"], top_k=3)
    assert len(groups) == 2
    for g in groups:
        assert g
        assert g[0]["id"] == "param_5"  # reranker re-ordered the fused set
        assert g[0]["score"] > 0.9  # sigmoid(10) ≈ 1.0 (absolute score)
        assert all(0.0 <= r["score"] <= 1.0 for r in g)


# ── additive provider filter ───────────────────────────────────────────────────


def _seed_two_providers(collection) -> None:
    """Seed the same physical quantity under two providers, with realistic ids."""
    rng = np.random.default_rng(7)
    entries = [
        ("cda/WI_H0_MFI/BGSEa", "cda"),
        ("cda/AC_H2_MFI/BGSEc", "cda"),
        ("cda/AC_H1_MFI/BGSEc", "cda"),
        ("amda/imf_real_gse", "amda"),
        ("amda/dsc_b_gse", "amda"),
    ]
    vecs = rng.random((len(entries), 128)).astype("float32")
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    collection.add(
        ids=[i for i, _ in entries],
        embeddings=vecs.tolist(),
        documents=["interplanetary magnetic field GSE. Units: nT." for _ in entries],
        metadatas=[{"name": i, "units": "nT", "provider": p} for i, p in entries],
    )


def test_provider_of_extracts_the_prefix():
    assert rag_module._provider_of("csa/C3_CP_X/density__C3_CP_X") == "csa"
    assert rag_module._provider_of("amda/imf_real_gse") == "amda"
    assert rag_module._provider_of("no-slash") == ""


def test_append_cross_provider_flags_and_appends_after():
    filtered = [{"id": "cda/a", "score": 0.9}, {"id": "cda/b", "score": 0.8}]
    unfiltered = [{"id": "cda/a", "score": 0.9}, {"id": "amda/x", "score": 0.7}]

    out = rag_module._append_cross_provider(filtered, unfiltered, "cda", 2)

    assert [r["id"] for r in out] == ["cda/a", "cda/b", "amda/x"]
    assert out[-1]["outside_filter"] is True
    assert "outside_filter" not in out[0], "in-filter hits must not be flagged"


def test_append_cross_provider_skips_duplicates_and_same_provider():
    filtered = [{"id": "cda/a"}]
    unfiltered = [{"id": "cda/a"}, {"id": "cda/c"}, {"id": "amda/x"}]

    out = rag_module._append_cross_provider(filtered, unfiltered, "cda", 5)

    assert [r["id"] for r in out] == ["cda/a", "amda/x"]


def test_append_cross_provider_respects_the_limit():
    filtered = [{"id": "cda/a"}]
    unfiltered = [{"id": "amda/x"}, {"id": "csa/y"}, {"id": "ssc/z"}]

    out = rag_module._append_cross_provider(filtered, unfiltered, "cda", 2)

    assert len(out) == 3


def test_provider_filter_no_longer_hides_other_providers(isolated_rag):
    """The point of the change: an over-eager filter must not make a hit unreachable.

    A `provider` filter is exclusive, so asking for `cda` used to put AMDA out of
    reach entirely — "not found" rather than "ranked lower", which is
    indistinguishable from the parameter not existing. The model over-applies the
    filter in practice, so the tool stops making that fatal.
    """
    _seed_two_providers(isolated_rag)

    results = search("interplanetary magnetic field GSE", top_k=3, provider="cda")

    providers = {rag_module._provider_of(r["id"]) for r in results}
    assert "cda" in providers
    assert "amda" in providers, "the excluded provider should still be reachable"


def test_filtered_hits_still_come_first(isolated_rag):
    _seed_two_providers(isolated_rag)

    results = search("interplanetary magnetic field GSE", top_k=3, provider="cda")

    flags = [bool(r.get("outside_filter")) for r in results]
    assert flags == sorted(flags), "out-of-filter hits must be appended, not interleaved"
    assert not flags[0], "the requested provider still ranks first"


def test_unfiltered_search_flags_nothing(isolated_rag):
    """With no filter there is nothing to be outside of."""
    _seed_two_providers(isolated_rag)

    results = search("interplanetary magnetic field GSE", top_k=5)

    assert results
    assert not any("outside_filter" in r for r in results)
