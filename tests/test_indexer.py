"""Unit tests for helioai.indexer: _get_region, _build_text, _walk."""

from __future__ import annotations

import pytest

from helioai.indexer import _build_text, _extract_dataset_meta, _get_region, _walk

# ─────────────────────────────── _get_region ────────────────────────────────


@pytest.mark.parametrize(
    "uid,expected",
    [
        ("amda/ace_imf_gse", "Heliosphere.NearEarth"),
        ("cda/AC_H0_MFI/BGSEc", "Heliosphere.NearEarth"),  # AC → ACE
        ("amda/cluster_b_gse", "Earth.Magnetosphere"),
        ("cda/C1_CP_FGM_FULL/B_mag", "Earth.Magnetosphere"),
        ("cda/MMS1_FGM_SRVY/B_vec", "Earth.Magnetosphere"),
        ("amda/mms_b_gse", "Earth.Magnetosphere"),
        ("amda/psp_mag_rtn", "Heliosphere.Inner"),
        ("cda/PSP_FLD_L2_MAG/B_RTN", "Heliosphere.Inner"),
        ("amda/cassini_b_mag", "Saturn"),
        ("cda/CAS_MAG_KRTP/B_KRTP", "Saturn"),
        ("amda/maven_sw_density", "Mars"),
        ("amda/unknown_xyz_123", ""),  # no match → empty
    ],
)
def test_get_region(uid: str, expected: str) -> None:
    assert _get_region(uid) == expected


# ─────────────────────────────── _build_text ────────────────────────────────


def test_build_text_minimal() -> None:
    text = _build_text("Bx", "X component of B", "nT", "ace_b_x")
    assert "Bx" in text
    assert "X component of B" in text
    assert "nT" in text


def test_build_text_with_region() -> None:
    text = _build_text("Np", "Proton density", "#/cc", "ace_np", region="Heliosphere.NearEarth")
    assert "Heliosphere.NearEarth" in text


def test_build_text_with_parent_meta() -> None:
    text = _build_text(
        "Np",
        "Proton density",
        "#/cc",
        "ace_np",
        parent_meta={"measurement_type": "ThermalPlasma", "dataset_description": "ACE/SWEPAM 64s"},
    )
    assert "ThermalPlasma" in text
    assert "ACE/SWEPAM 64s" in text


def test_build_text_with_entity_and_prop() -> None:
    text = _build_text(
        "Ne", "Electron density", "#/cc", "mms_ne", entity="Electron", prop="NumberDensity"
    )
    assert "Electron" in text
    assert "NumberDensity" in text


def test_build_text_no_description_uses_xmlid() -> None:
    text = _build_text("ace_b_x", "", "nT", "ace_b_x")
    assert "ace b x" in text or "ace_b_x" in text


# ─────────────────────────────── _walk ──────────────────────────────────────


class _FakeSpeasyIndex:
    """Minimal stand-in for speasy.core.inventory.indexes.SpeasyIndex."""

    pass


def _make_param(
    xmlid: str, description: str = "desc", units: str = "nT", name: str = ""
) -> _FakeSpeasyIndex:
    node = _FakeSpeasyIndex()
    node.xmlid = xmlid
    node.description = description
    node.units = units
    node.name = name or xmlid
    node.__spz_type__ = "ParameterIndex"
    return node


def _make_tree(**children) -> _FakeSpeasyIndex:
    node = _FakeSpeasyIndex()
    for k, v in children.items():
        setattr(node, k, v)
    return node


def test_walk_collects_amda_parameter() -> None:
    param = _make_param("ace_b_gse", "ACE magnetic field GSE", "nT")
    tree = _make_tree(ace_b=param)
    docs: list[dict] = []
    _walk(tree, "amda", docs, set(), _FakeSpeasyIndex)
    assert len(docs) == 1
    assert docs[0]["id"] == "amda/ace_b_gse"
    assert "nT" in docs[0]["text"]


def test_walk_deduplication() -> None:
    param = _make_param("ace_b_gse", "ACE B field", "nT")
    tree = _make_tree(branch_a=_make_tree(p=param), branch_b=_make_tree(p=param))
    docs: list[dict] = []
    skip_ids: set[str] = set()
    _walk(tree, "amda", docs, skip_ids, _FakeSpeasyIndex)
    ids = [d["id"] for d in docs]
    assert ids.count("amda/ace_b_gse") == 1


def test_walk_skips_existing_ids() -> None:
    param = _make_param("ace_b_gse", "ACE B", "nT")
    tree = _make_tree(p=param)
    docs: list[dict] = []
    existing = {"amda/ace_b_gse"}
    _walk(tree, "amda", docs, existing, _FakeSpeasyIndex)
    assert docs == []


def test_walk_multiple_params() -> None:
    p1 = _make_param("ace_np", "ACE proton density", "#/cc")
    p2 = _make_param("ace_vp", "ACE proton speed", "km/s")
    p3 = _make_param("mms_b_gse", "MMS magnetic field", "nT")
    tree = _make_tree(p1=p1, p2=p2, p3=p3)
    docs: list[dict] = []
    _walk(tree, "amda", docs, set(), _FakeSpeasyIndex)
    assert len(docs) == 3
    collected_ids = {d["id"] for d in docs}
    assert collected_ids == {"amda/ace_np", "amda/ace_vp", "amda/mms_b_gse"}


def test_walk_meta_in_output() -> None:
    param = _make_param("ace_np", "Proton density", "#/cc", name="Np")
    tree = _make_tree(p=param)
    docs: list[dict] = []
    _walk(tree, "amda", docs, set(), _FakeSpeasyIndex)
    meta = docs[0]["meta"]
    assert meta["provider"] == "amda"
    assert meta["xmlid"] == "ace_np"


def test_time_axes_are_not_indexed_as_products():
    """cda/AC_OR_SSC/Epoch was indexed as the ACE position and broke the download.

    Its CDF metadata really does say FIELDNAM=XYZ_GSE and CATDESC="ACE X/Y/Z GSE
    coordinates", so only cdf_type distinguishes it from the real product.
    """
    from helioai.indexer import _is_time_axis

    epoch = {
        "FIELDNAM": "XYZ_GSE",
        "CATDESC": "ACE X/Y/Z GSE coordinates (time-series)",
        "cdf_type": "CDF_EPOCH",
        "__spz_name__": "Epoch",
    }
    assert _is_time_axis(epoch)
    assert _is_time_axis({"cdf_type": "CDF_TIME_TT2000"})
    assert not _is_time_axis({"cdf_type": "CDF_FLOAT", "__spz_name__": "GSE_POS"})
    assert not _is_time_axis({})


def test_coverage_is_read_from_the_inventory():
    from helioai.indexer import _coverage

    assert _coverage({"start_date": "1997-08-25 17:48:00", "stop_date": "2026-10-05 23:49:00"}) == (
        "1997-08-25 17:48:00",
        "2026-10-05 23:49:00",
    )
    assert _coverage({}) == ("", "")


def test_the_dates_a_product_covers_are_in_its_text() -> None:
    """Nearly every question names a year and the text never did: "2019" matched nothing and
    diluted the rest. The catalogue index has written `Survey: … to …` since it was built;
    the products now say `Coverage: … to …` too, including the SSC trajectories."""
    from helioai.indexer import _coverage_sentence

    text = _build_text("Bx", "X", "nT", "ace_b_x", coverage=("1997-08-25 17:48:00", "2026-10-05"))
    assert "Coverage: 1997-08-25 to 2026-10-05." in text
    assert "Coverage" not in _build_text("Bx", "X", "nT", "ace_b_x")
    assert _coverage_sentence(("", "2026-10-05")) == "" and _coverage_sentence(None) == ""

    node = _make_param("DS/p", "a field", "nT")
    node.start_date = "2004-11-07 00:00:00"
    node.stop_date = "2026-01-01 00:00:00"
    docs: list[dict] = []
    _walk(_make_tree(ds=_make_tree(p=node)), "cda", docs, set(), _FakeSpeasyIndex)
    assert "Coverage: 2004-11-07 to 2026-01-01." in docs[0]["text"]
    assert docs[0]["meta"]["start_time"] == "2004-11-07 00:00:00"

    tree = _make_tree(Trajectories=_make_tree(mms1=_make_ssc_trajectory("mms1", "MMS 1")))
    ssc: list[dict] = []
    _walk(tree, "ssc", ssc, set(), _FakeSpeasyIndex)
    assert "Coverage: 2015-03-13 to " in ssc[0]["text"]


# ────────────────────── identity carried into the document ──────────────────────


def test_build_text_names_the_mission_that_owns_the_product() -> None:
    """The indexed text must say which spacecraft a product belongs to.

    AMDA parameter ids are opaque — the definitive ACE 16-second IMF vector is
    `amda/imf` — and its description never names the spacecraft either. 780 AMDA
    parameters were unreachable by a query naming their own mission because the
    only field that carried it, the parent dataset, was resolved and then dropped.
    """
    text = _build_text(
        "b_gse",
        "Magnetic field vector in GSE Cartesian coordinates (16 sec)",
        "nT",
        "imf",
        parent_meta={"dataset_id": "ace-imf-all", "mission": "ACE MFI"},
    )
    assert "ace-imf-all" in text
    assert "ACE" in text
    assert "MFI" in text


def test_build_text_names_the_csa_observatory() -> None:
    """Same defect, other provider: _extract_dataset_meta collected these and nothing used them."""
    text = _build_text(
        "3d_ions",
        "HIA 3D ion distribution",
        "s^3 km^-6",
        "3d_ions",
        parent_meta={"observatory": "Cluster-1", "experiments": "CIS-HIA"},
    )
    assert "Cluster-1" in text
    assert "CIS-HIA" in text


def test_build_text_without_identity_is_unchanged() -> None:
    """A product whose dataset carries no identity must not gain empty labels."""
    text = _build_text("Bx", "X component of B", "nT", "ace_b_x", parent_meta={})
    assert "Dataset:" not in text
    assert "Mission:" not in text
    assert "Instrument:" not in text


@pytest.mark.parametrize(
    "spase_id,expected",
    [
        ("spase://CNES/NumericalData/CDPP-AMDA/ACE/MFI/ace-imf-all", "ACE MFI"),
        ("spase://CNES/NumericalData/CDPP-AMDA/CCE/MEPA/cce-mepa-tof", "CCE MEPA"),
        ("spase://CNES/NumericalData/CDPP-AMDA/JUNO/FGM/JNO/juno-fgm-orb", "JUNO FGM JNO"),
        ("spase://CNES/NumericalData/CDPP-AMDA/THEMIS/tha-orb", "THEMIS"),
        ("spase://SOMEONE/ELSE/whatever", ""),
        ("", ""),
    ],
)
def test_amda_mission(spase_id: str, expected: str) -> None:
    from helioai.indexer import _amda_mission

    assert _amda_mission(spase_id) == expected


# ─────────────────── published region beats the guessed table ───────────────────


def test_dataset_meta_carries_the_published_region() -> None:
    """AMDA publishes the SPASE Region per dataset in `target`.

    _get_region guesses one instead, from a 40-entry table matched as a substring,
    and its two-letter keys collide: "ac" (ACE) matches inside "cce_mepa_ion_act",
    so an AMPTE/CCE magnetosheath product was labelled Heliosphere.NearEarth. The
    table disagrees with the source on 1279 products and is silent on 1813 more.
    """
    meta = _extract_dataset_meta({"target": "Earth.Magnetosheath"}, "amda")
    assert meta["region"] == "Earth.Magnetosheath"


def test_dataset_meta_has_no_region_when_the_source_is_silent() -> None:
    """21% of AMDA datasets publish no target — the table stays their only evidence."""
    assert "region" not in _extract_dataset_meta({"desc": "something"}, "amda")


def test_region_for_prefers_the_source_over_the_table() -> None:
    from helioai.indexer import _region_for

    assert _region_for("amda/cce_mepa_ion_act", {"region": "Earth.Magnetosheath"}) == (
        "Earth.Magnetosheath"
    )


def test_region_for_falls_back_to_the_table() -> None:
    from helioai.indexer import _region_for

    assert _region_for("amda/psp_mag_rtn", {}) == "Heliosphere.Inner"


# ── SSC trajectories ───────────────────────────────────────────────────────────


def _make_ssc_trajectory(uid: str, name: str) -> _FakeSpeasyIndex:
    """An SSC node as speasy builds it: no xmlid, no description, no CATDESC."""
    node = _FakeSpeasyIndex()
    node.Id = uid
    node.Resolution = "60"
    node.Geometry = "None"
    node.start_date = "2015-03-13T04:31:30.000Z"
    node.stop_date = "2027-04-12T00:02:30.000Z"
    node.__spz_name__ = name
    node.__spz_uid__ = uid
    node.__spz_type__ = "ParameterIndex"
    return node


def test_walk_indexes_an_ssc_trajectory_under_the_id_speasy_downloads() -> None:
    """The general walk needs `xmlid and description`; an SSC node has neither, so the
    314 trajectories were never indexed and no position query could reach them."""
    tree = _make_tree(Trajectories=_make_tree(mms1=_make_ssc_trajectory("mms1", "MMS 1")))
    docs: list[dict] = []
    _walk(tree, "ssc", docs, set(), _FakeSpeasyIndex)
    assert [d["id"] for d in docs] == ["ssc/mms1"]
    doc = docs[0]
    assert "MMS 1" in doc["text"] and "position" in doc["text"] and "trajectory" in doc["text"]
    assert "GSM" in doc["text"] and "km" in doc["text"]
    assert doc["meta"]["provider"] == "ssc" and doc["meta"]["units"] == "km"
    assert doc["meta"]["start_time"] == "2015-03-13T04:31:30" and doc["meta"]["region"]


def test_an_ssc_trajectory_is_indexed_once_and_a_cda_node_is_untouched_by_the_rule() -> None:
    tree = _make_tree(Trajectories=_make_tree(mms1=_make_ssc_trajectory("mms1", "MMS 1")))
    docs: list[dict] = []
    seen: set[str] = set()
    _walk(tree, "ssc", docs, seen, _FakeSpeasyIndex)
    _walk(tree, "ssc", docs, seen, _FakeSpeasyIndex)
    assert len(docs) == 1
    cda = _make_tree(ds=_make_tree(p=_make_param("DS/p", "a field", "nT")))
    cda_docs: list[dict] = []
    _walk(cda, "cda", cda_docs, set(), _FakeSpeasyIndex)
    assert [d["id"] for d in cda_docs] == ["cda/DS/p"]


# ─────────────────────────────── open_collections ───────────────────────────


def _persisted(chroma_dir) -> tuple[int | None, int]:
    """(seq id the vector segment persisted up to, rows left in the write-ahead log).

    Read from Chroma's own SQLite tables: nothing in its API says whether a write reached
    the on-disk graph or only the log, and that difference is the whole point.
    """
    import sqlite3

    db = sqlite3.connect(f"file:{chroma_dir}/chroma.sqlite3?mode=ro", uri=True)
    vector = [s[0] for s in db.execute("select id, type from segments") if "vector" in s[1]][0]
    row = db.execute("select seq_id from max_seq_id where segment_id = ?", (vector,)).fetchone()
    return (row[0] if row else None), db.execute(
        "select count(*) from embeddings_queue"
    ).fetchone()[0]


def test_open_collections_persists_every_write_and_settles_a_legacy_collection(tmp_path):
    """Chroma persists its HNSW graph every `sync_threshold` writes and replays the rest into
    memory at every start, in a varying order — five processes, five rankings (2026-09-22).
    A collection the indexer opens persists at once; one created before the setting is
    settled on open, so nothing is left to replay."""
    import chromadb
    import numpy as np

    from helioai.indexer import HNSW_EF_SEARCH, HNSW_SYNC_THRESHOLD, open_collections

    chroma_dir = tmp_path / "chroma"
    rng = np.random.default_rng(0)
    legacy = chromadb.PersistentClient(path=str(chroma_dir)).get_or_create_collection(
        "products", metadata={"hnsw:space": "cosine"}
    )
    legacy.upsert(ids=[f"p{i}" for i in range(300)], embeddings=rng.normal(size=(300, 8)).tolist())
    assert _persisted(chroma_dir) == (None, 300), (
        "300 writes below the default threshold: all in the log"
    )

    _, (products, catalogs) = open_collections(chroma_dir, ["products", "catalogs"])

    assert _persisted(chroma_dir)[0] == 300 and _persisted(chroma_dir)[1] <= 1
    assert products.count() == 300
    for c in (products, catalogs):
        assert c.configuration_json["hnsw"]["sync_threshold"] == HNSW_SYNC_THRESHOLD
        assert c.configuration_json["hnsw"]["ef_search"] == HNSW_EF_SEARCH

    products.upsert(ids=["p300"], embeddings=rng.normal(size=(1, 8)).tolist())
    catalogs.upsert(ids=["c0"], embeddings=rng.normal(size=(1, 8)).tolist())
    persisted, pending = _persisted(chroma_dir)
    assert persisted == 301 and pending <= 2, "one write, persisted at once"


def test_open_collections_is_a_plain_open_when_nothing_is_legacy(tmp_path):
    from helioai.indexer import HNSW_SYNC_THRESHOLD, open_collections

    chroma_dir = tmp_path / "chroma"
    _, (first,) = open_collections(chroma_dir, ["products"])
    first.upsert(ids=["a"], embeddings=[[0.0] * 8])
    _, (second,) = open_collections(chroma_dir, ["products"])
    assert second.count() == 1
    assert second.configuration_json["hnsw"]["sync_threshold"] == HNSW_SYNC_THRESHOLD


# ─────────────────────────── classify_measurement_types ────────────────────────


def _answers(label, confidence):
    from helioai.core.judgment import Answers

    return Answers(
        values={"mtype": label if confidence >= 0.9 else None},
        raw={"mtype": {"choice": label, "confidence": confidence, "probabilities": {}}},
        model="jev-test",
        latency_ms=1.0,
    )


def test_classification_fills_the_empty_flags_the_contradicted_and_never_overwrites(
    tmp_path, monkeypatch
):
    """Measured on 200 labelled products (2026-09-22): 89 % agreement at confidence ≥ 0.9,
    and the confident disagreements were the archive's own errors. So: fill where empty,
    flag where the archive disagrees, keep the published label, abstain below the floor."""
    import asyncio

    from helioai.config import settings
    from helioai.core import judgment
    from helioai.indexer import MEASUREMENT_TYPE_FLOOR, classify_measurement_types

    monkeypatch.setattr(settings.judgment, "backend", "jev")
    docs = [
        {"id": "cda/A/x", "text": "MMS1 position GSE. Units: km.", "meta": {"name": "x"}},
        {"id": "cda/B/y", "text": "Some housekeeping. Units: V.", "meta": {"name": "y"}},
        {
            "id": "amda/mms1_dis_ni",
            "text": "density. Measurement: MagneticField. Dataset: mms1-fpi-dismoms.",
            "meta": {"name": "ni", "measurement_type": "MagneticField"},
        },
        {
            "id": "amda/imf",
            "text": "IMF vector. Measurement: MagneticField.",
            "meta": {"name": "imf", "measurement_type": "MagneticField"},
        },
    ]
    seen: dict = {}

    async def fake_batch(site, states, questions, *, concurrency=8, record_to=None):
        seen["site"], seen["states"], seen["record_to"] = site, states, record_to
        assert set(questions) == {"mtype"} and questions["mtype"].floor == MEASUREMENT_TYPE_FLOOR
        return [
            _answers("Ephemeris", 0.97),
            _answers("InstrumentStatus", 0.6),
            _answers("ThermalPlasma", 0.99),
            _answers("Magnetic_Field", 0.95),
        ]

    monkeypatch.setattr(judgment, "batch", fake_batch)
    out = asyncio.run(classify_measurement_types(docs, tmp_path))

    assert (
        seen["site"] == "index_measurement_type"
        and seen["record_to"] == tmp_path / "judgment_index.jsonl"
    )
    assert "Measurement:" not in seen["states"][2]["product"], "the label is stripped before asking"
    filled = out[0]["meta"]
    assert filled["measurement_type"] == "Ephemeris" and filled["measurement_type_source"] == "jev"
    assert filled["measurement_type_confidence"] == 0.97
    assert out[0]["text"].endswith("Measurement: Ephemeris.")
    assert "measurement_type" not in out[1]["meta"], "below the floor: the field stays empty"
    assert out[2]["meta"]["measurement_type"] == "MagneticField", (
        "a published label is never overwritten"
    )
    assert out[2]["meta"]["measurement_type_jev"] == "ThermalPlasma"
    assert out[2]["meta"]["measurement_type_jev_confidence"] == 0.99
    assert "measurement_type_jev" not in out[3]["meta"], "Magnetic_Field agrees with MagneticField"


def test_classification_is_skipped_without_a_judging_backend(tmp_path, monkeypatch, capsys):
    import asyncio

    from helioai.config import settings
    from helioai.indexer import classify_measurement_types

    monkeypatch.setattr(settings.judgment, "backend", "null")
    docs = [{"id": "cda/A/x", "text": "t", "meta": {"name": "x"}}]
    assert asyncio.run(classify_measurement_types(docs, tmp_path, verbose=True)) == docs
    assert "skipping classification" in capsys.readouterr().out
    assert "measurement_type" not in docs[0]["meta"]
