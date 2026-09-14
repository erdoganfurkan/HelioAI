"""Tests for helioai.tools.rag.search using in-memory ChromaDB."""

from __future__ import annotations

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
    monkeypatch.setattr(rag_module, "_dataset_prefixes", None)

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


def _seed_a_log_companion(collection) -> None:
    """Seed a real measurement next to its log-scaled copy.

    `_seed` gives every product the same penalty, which makes `_apply_domain_rerank` a
    no-op and lets any ordering claim pass. A companion product is the cheapest thing
    that actually moves in the fourth reordering step.
    """
    rng = np.random.default_rng(3)
    entries = [
        ("cda/WI_H1_SWE/Proton_Np_nonlin", "Proton number density Np (linear scale)."),
        ("cda/WI_H1_SWE/Proton_Np_nonlin_log", "Proton number density Np (log scale)."),
        ("cda/WI_H1_SWE/Proton_V_nonlin", "Proton bulk speed (linear scale)."),
    ]
    vecs = rng.random((len(entries), 128)).astype("float32")
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    collection.add(
        ids=[i for i, _ in entries],
        embeddings=vecs.tolist(),
        documents=[d for _, d in entries],
        metadatas=[{"name": i, "provider": "cda", "xmlid": i} for i, _ in entries],
    )


def test_search_demotes_the_log_scaled_copy_below_the_real_measurement(isolated_rag) -> None:
    """The domain rerank is the only reordering stage left, and this is what it is for:
    a query for the density must not rank its log-scaled companion above it."""
    _seed_a_log_companion(isolated_rag)

    results = search("Wind proton density", top_k=5)

    ids = [r["id"] for r in results]
    assert "cda/WI_H1_SWE/Proton_Np_nonlin" in ids
    assert ids.index("cda/WI_H1_SWE/Proton_Np_nonlin") < ids.index(
        "cda/WI_H1_SWE/Proton_Np_nonlin_log"
    )


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


def test_coverage_string_from_index_metadata():
    from helioai.tools.rag import _coverage_of

    assert _coverage_of({"start_time": "1997-08-25 17:48:00", "stop_time": "2026-10-05"}) == (
        "1997-08-25 → 2026-10-05"
    )
    assert _coverage_of({}) == ""
    assert _coverage_of(None) == ""


def test_covers_window():
    from helioai.tools.rag import _covers

    wi_or_def = "1994-12-29 → 2001-05-31"
    assert not _covers(wi_or_def, ("2015-03-17", "2015-03-18"))
    assert _covers(wi_or_def, ("1996-01-01", "1996-01-02"))
    assert _covers(wi_or_def, ("2001-05-01", "2002-01-01")), "partial overlap still counts"
    # A stale index must never hide a good product; get_timeseries still guards.
    assert _covers("", ("2015-03-17", "2015-03-18"))
    assert _covers(wi_or_def, None)


def test_window_demotes_but_never_drops():
    from helioai.tools.speasy_tools import _apply_window

    results = [
        {"id": "cda/WI_OR_DEF/GSE_POS", "coverage": "1994-12-29 → 2001-05-31"},
        {"id": "cda/WI_H0_MFI/PGSE", "coverage": "1994-11-01 → 2026-01-01"},
    ]
    out = _apply_window(results, ("2015-03-17", "2015-03-18"))
    assert len(out) == 2, "demoted, not dropped"
    assert out[0]["id"] == "cda/WI_H0_MFI/PGSE"
    assert out[1]["covers_window"] is False


def test_browse_products_are_flagged():
    """Ranking puts three WI_K0_SWE hits first for "science quality" density.

    K0 is the trap the notebook is built around, so the result says so rather than
    relying on the model to notice "[PRELIM]" in the prose.
    """
    from helioai.tools.rag import _quality_of

    assert _quality_of("cda/WI_K0_SWE/Np", "Proton density. Wind SWE Key Parameters") == "browse"
    assert _quality_of("cda/AC_K1_SWE/Tpr", "... Key Parameters [PRELIM] ...") == "browse"
    assert _quality_of("cda/WI_H0_MFI/B3GSM", "... (3 sec). Definitive Data ...") == ""
    assert _quality_of("cda/WI_H1_SWE/Proton_Np_nonlin", "... moments ...") == ""


def test_cadence_is_discoverable_from_the_description():
    """The agent was asked to pick a cadence with no cadence field anywhere.

    speasy's inventory carries none, so the only source is the description text —
    and looping the search does not make one appear.
    """
    from helioai.tools.setup import registry

    schema = next(t for t in registry.list_tool_defs() if t.name == "search_parameters")
    assert "Cadence is stated in `description`" in schema.description


# ── domain reranking ──────────────────────────────────────────────────────────
# Measured on the notebook's own Act I queries against the real index: the correct
# product sat at rank 4, 24, 2, 3 and 7, and rank 1 went to a browse product, another
# mission, or spacecraft housekeeping. After reranking all five are in the top 5.
# These tests pin the logic, not the index, so they stay fast and deterministic.


def test_solar_wind_is_not_the_Wind_spacecraft():
    """Nearly every query here says "solar wind"; matching "wind" naively tags them all."""
    from helioai.tools.rag import _query_mission

    assert _query_mission("ACE proton density in the solar wind") == "ace"
    assert _query_mission("solar wind speed") is None
    assert _query_mission("Wind MFI magnetic field") == "wind"
    assert _query_mission("Wind spacecraft solar wind density") == "wind"
    assert _query_mission("interface region imaging") is None, "'ace' inside a word"


def test_candidate_mission_from_the_dataset_prefix():
    from helioai.tools.rag import _candidate_mission

    assert _candidate_mission("cda/WI_H0_MFI/B3GSM") == "wind"
    assert _candidate_mission("cda/AC_H0_MFI/BGSM") == "ace"
    assert _candidate_mission("cda/MMS1_FGM_SRVY_L2/mms1_fgm_b_gsm") == "mms"
    assert _candidate_mission("amda/imf_gsm") is None, "unknown is not a mismatch"


def test_wrong_mission_and_housekeeping_are_pushed_down():
    from helioai.tools.rag import _apply_domain_rerank

    candidates = [
        {"id": "cda/HK_H0_MAG/B_gsm", "quality": ""},
        {"id": "cda/AC_H0_SWE/Vp", "quality": ""},
        {"id": "cda/WI_K0_SWE/Np", "quality": "browse"},
        {"id": "amda/sw_n", "quality": ""},
        {"id": "cda/WI_H1_SWE/Proton_Np_nonlin", "quality": ""},
    ]
    out = [c["id"] for c in _apply_domain_rerank("Wind proton density solar wind", candidates)]
    assert out[0] == "cda/WI_H1_SWE/Proton_Np_nonlin"
    assert out.index("amda/sw_n") < out.index("cda/AC_H0_SWE/Vp"), "unknown beats wrong mission"
    assert out[-1] == "cda/HK_H0_MAG/B_gsm", "housekeeping is never a science answer"


def test_browse_is_kept_when_it_is_what_was_asked_for():
    from helioai.tools.rag import _rerank_penalty

    k0 = {"id": "cda/WI_K0_SWE/Np", "quality": "browse"}
    assert _rerank_penalty("Wind key parameter density", k0) == 0
    assert _rerank_penalty("Wind proton density", k0) == 1


def test_rerank_never_drops_a_candidate():
    from helioai.tools.rag import _apply_domain_rerank

    candidates = [{"id": f"cda/X{i}_A/B", "quality": ""} for i in range(7)]
    assert len(_apply_domain_rerank("Wind density", candidates)) == 7


def test_derived_companions_are_demoted():
    """`Proton_Np_nonlin_log` is log10(n): 1.24 where the density is 17.4.

    Two Act I runs out of four picked it over the real density, which would have put a
    compression ratio near 1 into an otherwise correct analysis. The archive labels them
    itself — "(linear scale)" against "(log scale)".
    """
    from helioai.tools.rag import _is_auxiliary

    assert _is_auxiliary("cda/WI_H1_SWE/Proton_Np_nonlin_log", "Proton density (log scale).")
    assert _is_auxiliary("cda/WI_H1_SWE/Proton_sigmaNp_nonlin", "1-sigma uncertainty in ...")
    assert _is_auxiliary("cda/WI_H1_SWE/Proton_W_nonlin_errorbars", "... with uncertainties")
    assert _is_auxiliary("cda/WI_K0_SWE/QF_Np", "Quality Flag: proton dens...")
    assert not _is_auxiliary(
        "cda/WI_H1_SWE/Proton_Np_nonlin",
        "Proton number density Np from non-linear analysis (linear scale).",
    ), "'non-linear' and 'linear scale' must not read as a log variant"
    assert not _is_auxiliary("cda/WI_H0_MFI/B3GSM", "Magnetic field vector in GSM (3 sec).")


def test_a_companion_is_kept_when_explicitly_asked_for():
    from helioai.tools.rag import _rerank_penalty

    log_np = {
        "id": "cda/WI_H1_SWE/Proton_Np_nonlin_log",
        "quality": "",
        "description": "Proton number density Np (log scale).",
    }
    assert _rerank_penalty("Wind proton density", log_np) == 2
    assert _rerank_penalty("Wind proton density log scale", log_np) == 0

    sigma = {
        "id": "cda/WI_H1_SWE/Proton_sigmaNp_nonlin",
        "quality": "",
        "description": "1-sigma uncertainty in the proton density.",
    }
    assert _rerank_penalty("Wind density uncertainty", sigma) == 0


# ─────────────── published processing level beats the prose sniff ───────────────


def test_calibrated_product_is_not_demoted_by_the_word_prelim() -> None:
    """AMDA's ace-imf-all reads "Level2/PRELIM Data" and is published Calibrated.

    The sniff read PRELIM and called the definitive ACE IMF vector browse quality,
    which ranked it below amda/imf_real_gse — the NOAA real-time feed, which carries
    no such word. All 99 AMDA products the sniff caught this way are Calibrated.
    """
    text = (
        "b_gse. Magnetic field vector in GSE Cartesian coordinates (16 sec). "
        "Processing: Calibrated. Interplanetary Magnetic Field 16-sec Level2/PRELIM Data."
    )
    assert rag_module._quality_of("amda/imf", text) == ""


def test_key_parameter_product_stays_browse_even_when_calibrated() -> None:
    """The K0 rule is read off the id and is the one this flag was built for."""
    text = "Np. Solar wind density. Processing: Calibrated."
    assert rag_module._quality_of("cda/WI_K0_SWE/Np", text) == "browse"


def test_prelim_without_a_published_level_still_flags() -> None:
    """CDA publishes no processing level here — the sniff stays its own evidence."""
    assert rag_module._quality_of("cda/AC_H0_MFI/BGSEc", "PRELIM data") == "browse"


# ── dataset ids are not fabrications ──────────────────────────────────────────


def _seed_dataset_with_params(collection) -> None:
    """Index parameters under a dataset without indexing the dataset node itself.

    This is the real shape of the CDA and CSA trees: `cda/WI_H2_MFI/BGSM` is a key,
    `cda/WI_H2_MFI` never is. Seeding the dataset node too would make the test pass
    for the wrong reason.
    """
    rng = np.random.default_rng(7)
    ids = ["cda/FAKE_DS/BGSE", "cda/FAKE_DS/BGSM"]
    vecs = rng.random((len(ids), 128)).astype("float32")
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    collection.add(
        ids=ids,
        embeddings=vecs.tolist(),
        documents=["Magnetic field GSE. Units: nT.", "Magnetic field GSM. Units: nT."],
        metadatas=[{"provider": "cda", "xmlid": i.split("/", 1)[1]} for i in ids],
    )


def test_a_real_dataset_id_is_not_reported_as_invented(isolated_rag):
    """A dataset id owns indexed parameters, so it exists — even without its own key.

    Measured against the production index on 2026-09-10:
    `unknown_ids(["cda/MVN_MAG_L2-SUNSTATE-1SEC"])` returned that id, a real MAVEN
    dataset. The detector then told the model "these ids are NOT in the catalogue",
    which is false and costs it a turn, in the CLI and on the web.
    """
    from helioai.tools.rag import unknown_ids

    _seed_dataset_with_params(isolated_rag)

    assert unknown_ids(["cda/FAKE_DS"]) == []


def test_a_fabricated_id_is_still_reported(isolated_rag):
    """The guardrail must not go mute while being taught about datasets."""
    from helioai.tools.rag import unknown_ids

    _seed_dataset_with_params(isolated_rag)

    assert unknown_ids(["cda/NOT_A_REAL_DS/XX"]) == ["cda/NOT_A_REAL_DS/XX"]
    assert unknown_ids(["cda/FAKE_DS/NOPE"]) == ["cda/FAKE_DS/NOPE"], (
        "a bogus parameter under a real dataset is still bogus"
    )


def test_loading_the_encoder_does_not_draw_a_progress_bar(monkeypatch, tmp_path):
    """Live run of 00_quickstart: the first search of a session displayed a
    "Loading weights 0/103" widget in the notebook — transformers 5 wraps its weight
    loading in tqdm, and Jupyter renders that as a progress bar in the middle of the
    agent's transcript. Loading a cached 90 MB model is not something the user needs
    a bar for. The bar must be off by the time the model is constructed.
    """
    import sys
    import types

    import chromadb
    from transformers.utils import logging as hf_logging

    seen: dict[str, bool] = {}

    class _Encoder:
        def __init__(self, name):
            seen["bar_enabled_at_construction"] = hf_logging.is_progress_bar_enabled()

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = _Encoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    chromadb.PersistentClient(path=str(tmp_path / "chroma")).get_or_create_collection("params")
    monkeypatch.setattr(rag_module.settings.rag, "chroma_dir", tmp_path / "chroma")
    monkeypatch.setattr(rag_module.settings.rag, "collection_name", "params")
    monkeypatch.setattr(rag_module, "_model", None)
    monkeypatch.setattr(rag_module, "_collection", None)

    hf_logging.enable_progress_bar()
    try:
        rag_module._load()
    finally:
        hf_logging.enable_progress_bar()

    assert seen["bar_enabled_at_construction"] is False


# ── a missing index names its legacy copy ─────────────────────────────────────


def test_missing_index_message_points_at_a_legacy_copy(monkeypatch, tmp_path):
    """Before 0.3.0 the index ignored HELIOAI_DATA_DIR; after upgrading, an install
    that sets it sees no index where it now looks — while the old one is a `mv` away.
    The message must say so rather than send the user into an hour-long rebuild."""
    import helioai.config as cfg
    from helioai.tools import rag as rag_module

    legacy_root = tmp_path / "legacy"
    (legacy_root / "chroma").mkdir(parents=True)
    monkeypatch.setattr(cfg, "_default_data_dir", lambda: legacy_root)
    monkeypatch.setattr(rag_module.settings.rag, "chroma_dir", tmp_path / "new" / "chroma")

    msg = rag_module._index_missing_message()
    assert str(tmp_path / "new" / "chroma") in msg
    assert str(legacy_root / "chroma") in msg
    assert "helioai migrate-storage" in msg


def test_missing_index_message_is_plain_when_there_is_no_legacy_copy(monkeypatch, tmp_path):
    import helioai.config as cfg
    from helioai.tools import rag as rag_module

    monkeypatch.setattr(cfg, "_default_data_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr(rag_module.settings.rag, "chroma_dir", tmp_path / "new" / "chroma")
    msg = rag_module._index_missing_message()
    assert "helioai index" in msg
    assert "migrate-storage" not in msg
