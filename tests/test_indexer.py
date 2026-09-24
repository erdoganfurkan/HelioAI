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
    assert meta["region"] == "Heliosphere.NearEarth" and meta["region_source"] == "table", (
        "no dataset published a target: the region is the table's guess and says so"
    )


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


# ─────────────────────────────── classify_products ───────────────────────────────


def _answers(label, confidence, region=None, region_confidence=0.0):
    from helioai.core.judgment import Answers

    return Answers(
        values={
            "mtype": label if confidence >= 0.9 else None,
            "region": region if region_confidence >= 0.9 else None,
        },
        raw={
            "mtype": {"choice": label, "confidence": confidence, "probabilities": {}},
            "region": {"choice": region, "confidence": region_confidence, "probabilities": {}},
        },
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
    from helioai.indexer import MEASUREMENT_TYPE_FLOOR, REGION_FLOOR, classify_products

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

    async def fake_batch(site, states, questions, *, concurrency=8, record_to=None, keys=None):
        seen["site"], seen["states"], seen["record_to"], seen["keys"] = (
            site,
            states,
            record_to,
            keys,
        )
        assert set(questions) == {"mtype", "region"}
        assert questions["mtype"].floor == MEASUREMENT_TYPE_FLOOR
        assert questions["region"].floor == REGION_FLOOR
        return [
            _answers("Ephemeris", 0.97),
            _answers("InstrumentStatus", 0.6),
            _answers("ThermalPlasma", 0.99),
            _answers("Magnetic_Field", 0.95),
        ]

    monkeypatch.setattr(judgment, "batch", fake_batch)
    out = asyncio.run(classify_products(docs, tmp_path))

    assert (
        seen["site"] == "index_classify" and seen["record_to"] == tmp_path / "judgment_index.jsonl"
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
    assert all("region" not in d["meta"] for d in out), "the judge abstained on every region"


def test_classification_replaces_the_tables_region_and_keeps_the_archives(tmp_path, monkeypatch):
    """Against 8 435 published AMDA targets the substring table agrees on 26.9 %; the judge is
    on the right body 97.1 % of the time at confidence ≥ 0.9 and beats the table 75:1 where
    they differ (2026-09-22). A guess is replaced or a silence filled; a publication stands."""
    import asyncio

    from helioai.config import settings
    from helioai.core import judgment
    from helioai.indexer import classify_products

    monkeypatch.setattr(settings.judgment, "backend", "jev")
    docs = [
        {
            "id": "amda/cce_mepa_ion_act",
            "text": "ion counts. Mission: AMPTE CCE. Region: Heliosphere.NearEarth. Coverage: 1984-08-21 to 1989-01-12.",
            "meta": {"name": "act", "region": "Heliosphere.NearEarth", "region_source": "table"},
        },
        {
            "id": "cda/SILENT/x",
            "text": "Some new mission. Units: nT. Coverage: 2020-01-01 to 2021-01-01.",
            "meta": {"name": "x"},
        },
        {
            "id": "amda/c1_b_gsm",
            "text": "bx. Region: Earth.Magnetosheath. Coverage: 2000-12-02 to 2024-09-08.",
            "meta": {"name": "bx", "region": "Earth.Magnetosheath", "region_source": "archive"},
        },
        {
            "id": "ssc/mms1",
            "text": "MMS1 spacecraft position. Units: km. Region: Earth.Magnetosphere.",
            "meta": {"name": "mms1", "region": "Earth.Magnetosphere", "region_source": "table"},
        },
        {
            "id": "cda/LOW/y",
            "text": "Unsure product. Region: Mars.",
            "meta": {"name": "y", "region": "Mars", "region_source": "table"},
        },
    ]
    seen: dict = {}

    async def fake_batch(site, states, questions, *, concurrency=8, record_to=None, keys=None):
        seen["states"] = states
        return [
            _answers("EnergeticParticles", 0.95, "Earth.Magnetosheath", 0.98),
            _answers("MagneticField", 0.99, "Heliosphere.Inner", 0.93),
            _answers("MagneticField", 0.99, "Earth.Magnetosphere", 0.99),
            _answers("Ephemeris", 0.99, "Earth.Magnetosphere", 0.97),
            _answers("Waves", 0.2, "Venus", 0.55),
        ]

    monkeypatch.setattr(judgment, "batch", fake_batch)
    out = asyncio.run(classify_products(docs, tmp_path, verbose=True))

    assert all("Region:" not in st["product"] for st in seen["states"]), (
        "the table's guess is stripped before asking, like the archive's label"
    )
    replaced = out[0]
    assert replaced["meta"]["region"] == "Earth.Magnetosheath"
    assert (
        replaced["meta"]["region_source"] == "jev" and replaced["meta"]["region_confidence"] == 0.98
    )
    assert replaced["text"] == (
        "ion counts. Mission: AMPTE CCE. Measurement: EnergeticParticles. "
        "Region: Earth.Magnetosheath. Coverage: 1984-08-21 to 1989-01-12."
    ), "one region sentence, before the coverage, where the guess used to be"
    filled = out[1]
    assert (
        filled["meta"]["region"] == "Heliosphere.Inner" and filled["meta"]["region_source"] == "jev"
    )
    assert filled["text"].endswith(
        "Measurement: MagneticField. Region: Heliosphere.Inner. Coverage: 2020-01-01 to 2021-01-01."
    )
    archive = out[2]
    assert (
        archive["meta"]["region"] == "Earth.Magnetosheath"
        and archive["meta"]["region_source"] == "archive"
    )
    assert (
        "region_confidence" not in archive["meta"]
        and "Region: Earth.Magnetosheath." in archive["text"]
    )
    confirmed = out[3]
    assert confirmed["meta"]["region_source"] == "jev" and confirmed["text"].count("Region:") == 1
    low = out[4]
    assert low["meta"] == {"name": "y", "region": "Mars", "region_source": "table"}, (
        "below the floor the guess stays, marked as the guess it is"
    )
    assert low["text"] == "Unsure product. Region: Mars."


def test_classification_is_skipped_without_a_judging_backend(tmp_path, monkeypatch, capsys):
    import asyncio

    from helioai.config import settings
    from helioai.indexer import classify_products

    monkeypatch.setattr(settings.judgment, "backend", "null")
    docs = [{"id": "cda/A/x", "text": "t", "meta": {"name": "x"}}]
    assert asyncio.run(classify_products(docs, tmp_path, verbose=True)) == docs
    assert "skipping classification" in capsys.readouterr().out
    assert "measurement_type" not in docs[0]["meta"]


def test_the_walk_says_where_a_region_came_from():
    from helioai.indexer import _ssc_trajectory_doc

    doc = _ssc_trajectory_doc(
        {"__spz_uid__": "mms1", "__spz_name__": "MMS1", "Resolution": 60}, set()
    )
    assert (
        doc["meta"]["region"] == "Earth.Magnetosphere" and doc["meta"]["region_source"] == "table"
    )


# ───────────── the judge's answers ship with the package and survive rebuilds ─────────────


def test_judged_answers_round_trip_and_later_files_override_earlier(tmp_path):
    from helioai.indexer import load_judged, save_judged

    shipped = tmp_path / "shipped.jsonl.gz"
    local = tmp_path / "local.jsonl.gz"
    save_judged(
        shipped,
        {"date": "2026-09-22", "models": ["jev-1.13.0"]},
        {
            "cda/A/x": {"name": "x", "mtype": {"choice": "Ephemeris", "confidence": 0.97}},
            "cda/B/y": {"name": "y", "mtype": {"choice": "Waves", "confidence": 0.4}},
        },
    )
    save_judged(
        local,
        {"date": "2026-10-01", "models": ["jev-1.13.0", "jev-1.14.0"]},
        {
            "cda/B/y": {
                "name": "y",
                "mtype": {"choice": "Waves", "confidence": 0.4},
                "region": {"choice": "Mars", "confidence": 0.95},
            },
            "cda/NEW/z": {"name": "z", "mtype": {"choice": "MagneticField", "confidence": 0.99}},
        },
    )
    meta, records = load_judged(shipped, local, tmp_path / "absent.jsonl.gz")
    assert meta == {"date": "2026-10-01", "models": ["jev-1.13.0", "jev-1.14.0"], "asked": 2}
    assert set(records) == {"cda/A/x", "cda/B/y", "cda/NEW/z"}
    assert records["cda/B/y"]["region"] == {"choice": "Mars", "confidence": 0.95}, "local wins"
    assert load_judged(tmp_path / "absent.jsonl.gz") == ({}, {})
    (tmp_path / "broken.jsonl.gz").write_bytes(b"not gzip")
    assert load_judged(tmp_path / "broken.jsonl.gz") == ({}, {})


def test_apply_judged_is_the_policy_over_raw_answers():
    """The file keeps what the judge *said* — choice and confidence — not what was decided,
    so the floors and the never-overwrite rule live here and can change without a request.
    An abstention below the floor is on record (not paid for twice) and writes nothing."""
    from helioai.indexer import apply_judged

    judged = {
        "cda/A/x": {"name": "x", "mtype": {"choice": "Ephemeris", "confidence": 0.97}},
        "cda/B/y": {"name": "y", "mtype": {"choice": "InstrumentStatus", "confidence": 0.6}},
        "amda/pub": {"name": "pub", "mtype": {"choice": "ThermalPlasma", "confidence": 0.99}},
        "amda/imf": {"name": "imf", "mtype": {"choice": "Magnetic_Field", "confidence": 0.95}},
        "cda/guess": {
            "name": "g",
            "mtype": {"choice": "MagneticField", "confidence": 0.99},
            "region": {"choice": "Venus", "confidence": 0.95},
        },
        "amda/c1": {
            "name": "bx",
            "mtype": {"choice": "MagneticField", "confidence": 0.99},
            "region": {"choice": "Earth.Magnetosphere", "confidence": 0.99},
        },
        "cda/renamed": {"name": "old name", "mtype": {"choice": "Waves", "confidence": 0.99}},
        "cda/nameless": {"mtype": {"choice": "Radiance", "confidence": 0.99}},
    }
    docs = [
        {
            "id": "cda/A/x",
            "text": "x. Units: km. Coverage: 2019-01-01 to 2020-01-01.",
            "meta": {"name": "x"},
        },
        {"id": "cda/B/y", "text": "y.", "meta": {"name": "y"}},
        {
            "id": "amda/pub",
            "text": "pub. Measurement: MagneticField.",
            "meta": {"name": "pub", "measurement_type": "MagneticField"},
        },
        {
            "id": "amda/imf",
            "text": "imf. Measurement: MagneticField.",
            "meta": {"name": "imf", "measurement_type": "MagneticField"},
        },
        {
            "id": "cda/guess",
            "text": "g. Region: Mars. Coverage: 2001-01-01 to 2002-01-01.",
            "meta": {"name": "g", "region": "Mars", "region_source": "table"},
        },
        {
            "id": "amda/c1",
            "text": "bx. Measurement: MagneticField. Region: Earth.Magnetosheath.",
            "meta": {
                "name": "bx",
                "measurement_type": "MagneticField",
                "region": "Earth.Magnetosheath",
                "region_source": "archive",
            },
        },
        {"id": "cda/renamed", "text": "r.", "meta": {"name": "new name"}},
        {"id": "cda/nameless", "text": "n.", "meta": {"name": "n"}},
        {"id": "cda/unknown", "text": "u.", "meta": {"name": "u"}},
    ]
    assert apply_judged(docs, judged) == 4

    x = docs[0]
    assert (
        x["meta"]["measurement_type"] == "Ephemeris"
        and x["meta"]["measurement_type_source"] == "jev"
    )
    assert x["meta"]["measurement_type_confidence"] == 0.97
    assert x["text"] == "x. Units: km. Measurement: Ephemeris. Coverage: 2019-01-01 to 2020-01-01."
    assert docs[1]["meta"] == {"name": "y"}, "below the floor: on record, nothing written"
    pub = docs[2]["meta"]
    assert (
        pub["measurement_type"] == "MagneticField"
        and pub["measurement_type_jev"] == "ThermalPlasma"
    )
    assert "measurement_type_jev" not in docs[3]["meta"], "Magnetic_Field agrees with MagneticField"
    guess = docs[4]
    assert guess["meta"]["measurement_type"] == "MagneticField"
    assert guess["meta"]["region"] == "Venus" and guess["meta"]["region_source"] == "jev"
    assert guess["text"] == (
        "g. Measurement: MagneticField. Region: Venus. Coverage: 2001-01-01 to 2002-01-01."
    )
    c1 = docs[5]["meta"]
    assert c1["region"] == "Earth.Magnetosheath" and c1["region_source"] == "archive", (
        "a published target stands whatever the judge said"
    )
    assert docs[6]["meta"] == {"name": "new name"}, (
        "the id was reused: the old answer is not evidence"
    )
    assert docs[7]["meta"]["measurement_type"] == "Radiance", "a record without a name is applied"
    assert docs[8]["meta"] == {"name": "u"}


def test_classification_asks_only_the_questions_no_record_answers_and_saves_them(
    tmp_path, monkeypatch
):
    """A second pass over the same catalogue costs nothing; a new provider costs its own
    products; a new question costs one request per product for that question alone."""
    import asyncio

    from helioai.config import settings
    from helioai.core import judgment
    from helioai.indexer import classify_products, load_judged, local_judged_path

    monkeypatch.setattr(settings.judgment, "backend", "jev")
    judged = {
        "cda/old": {"name": "old", "mtype": {"choice": "Ephemeris", "confidence": 0.97}},
        "cda/done": {
            "name": "done",
            "mtype": {"choice": "Waves", "confidence": 0.3},
            "region": {"choice": "Mars", "confidence": 0.2},
        },
    }
    docs = [
        {"id": "cda/old", "text": "old.", "meta": {"name": "old"}},
        {"id": "cda/done", "text": "done.", "meta": {"name": "done"}},
        {
            "id": "cda/new",
            "text": "new. Coverage: 2020-01-01 to 2021-01-01.",
            "meta": {"name": "new"},
        },
    ]
    calls: list[dict] = []

    async def fake_batch(site, states, questions, *, concurrency=8, record_to=None, keys=None):
        calls.append({"states": states, "questions": tuple(questions), "keys": keys})
        if tuple(questions) == ("region",):
            return [_answers("ignored", 0.0, "Heliosphere.Inner", 0.93)]
        return [_answers("MagneticField", 0.95, "Earth.Magnetosphere", 0.55)]

    monkeypatch.setattr(judgment, "batch", fake_batch)
    out = asyncio.run(
        classify_products(
            docs,
            tmp_path,
            judged=judged,
            judged_meta={"date": "2026-09-22", "models": ["jev-1.13.0"]},
            verbose=True,
        )
    )

    by_questions = {c["questions"]: c for c in calls}
    assert set(by_questions) == {("region",), ("mtype", "region")}, "one group per missing set"
    assert by_questions[("region",)]["keys"] == ["cda/old"]
    assert by_questions[("mtype", "region")]["keys"] == ["cda/new"]
    assert by_questions[("mtype", "region")]["states"] == [
        {"product": "new. Coverage: 2020-01-01 to 2021-01-01."}
    ]

    assert out[0]["meta"]["measurement_type"] == "Ephemeris", "from the record, not the request"
    assert (
        out[0]["meta"]["region"] == "Heliosphere.Inner" and out[0]["meta"]["region_source"] == "jev"
    )
    assert out[1]["meta"] == {"name": "done"}, (
        "two abstentions on record: nothing asked, nothing written"
    )
    new = out[2]
    assert new["meta"]["measurement_type"] == "MagneticField"
    assert "region" not in new["meta"], "0.55 is below the region floor"
    assert new["text"] == "new. Measurement: MagneticField. Coverage: 2020-01-01 to 2021-01-01."

    assert judged["cda/new"] == {
        "name": "new",
        "mtype": {"choice": "MagneticField", "confidence": 0.95},
        "region": {"choice": "Earth.Magnetosphere", "confidence": 0.55},
    }, "the abstention is on record too — it will not be paid for again"
    assert judged["cda/old"]["region"] == {"choice": "Heliosphere.Inner", "confidence": 0.93}

    meta, saved = load_judged(local_judged_path())
    assert local_judged_path().is_relative_to(tmp_path.parent.parent) or str(
        local_judged_path()
    ).startswith("/tmp")
    assert saved == judged and meta["asked"] == 3 and meta["models"] == ["jev-1.13.0", "jev-test"]
    assert meta["floors"] == {"mtype": 0.9, "region": 0.9}


def test_the_shipped_answers_load_and_name_their_provenance():
    from helioai.indexer import SHIPPED_JUDGED, load_judged

    meta, records = load_judged(SHIPPED_JUDGED)
    assert meta["asked"] == len(records) >= 82_000, "the first pass: 82 266 products asked"
    assert meta["date"] == "2026-09-22" and meta["models"] == ["jev-1.13.0"]
    assert meta["floors"] == {"mtype": 0.9, "region": 0.9}
    sample = records["cda/RBSP-A_MAGNETOMETER_4SEC-GSM_EMFISIS-L3/Mag"]
    assert sample["name"] == "Mag" and sample["mtype"]["choice"] == "MagneticField"
    assert sample["mtype"]["confidence"] >= 0.9
    assert all(r.get("mtype", {}).get("choice") for r in records.values()), "every record answers"
    assert all("/" in pid for pid in records), "ids are provider/dataset/variable"


# ──────────────── a CDA product's text says whose it is (2026-09-22) ────────────────


def test_cda_spase_words_and_cadence():
    from helioai.indexer import _cda_spase

    assert _cda_spase("spase://NASA/NumericalData/RBSP/A/EMFISIS/MAGNETOMETER/L3/GSM/PT4S") == (
        "RBSP A EMFISIS MAGNETOMETER L3 GSM",
        "4 s",
    )
    assert _cda_spase("spase://ESA/NumericalData/Equator-S/EDI/PP/PT60s") == (
        "Equator-S EDI PP",
        "1 min",
    )
    assert (
        _cda_spase("spase://NASA/NumericalData/THEMIS/Ground/PENGUIn.5/Magnetometer/PT1S")[1]
        == "1 s"
    )
    assert _cda_spase("spase://NASA/NumericalData/BARREL/1K/Magnetometer/L2/PT0.25S")[1] == "250 ms"
    assert _cda_spase("spase://NASA/NumericalData/OMNI/PT1H") == ("OMNI", "1 h")
    assert _cda_spase("spase://NASA/NumericalData/IBEX/H3") == ("IBEX H3", ""), (
        "no duration: no cadence"
    )
    assert _cda_spase(" ") == ("", "") and _cda_spase("spase://NASA/DisplayData/X/PT1S") == ("", "")


def test_cda_components_are_the_vector_labels_only():
    from helioai.indexer import _cda_components

    assert _cda_components({"LABL_PTR_1": ["Bx_GSM  ", "By_GSM  ", "Bz_GSM  "]}) == [
        "Bx_GSM",
        "By_GSM",
        "Bz_GSM",
    ]
    assert _cda_components({"LABL_PTR_1": ["Np"]}) == [], "a scalar's label repeats its name"
    assert _cda_components({"LABL_PTR_1": "Np"}) == [] and _cda_components({}) == []


def test_cda_dataset_meta_carries_the_identity_the_text_lacked():
    """`cda/RBSP-A_MAGNETOMETER_4SEC-GSM_EMFISIS-L3/Mag` read "Mag. Magnetometer vector.
    Fluxgate magnetometer data - Craig Kletzing (University of Iowa)." — no RBSP, no EMFISIS,
    no GSM, no 4 s; the dense channel could not tell it from any other magnetometer, and
    for "Van Allen Probe A EMFISIS 4-second magnetic field GSM" ACE, ISEE and IMP-8
    outranked it. The same defect that once hid 780 AMDA products from their mission name."""
    from helioai.indexer import _build_text, _extract_dataset_meta

    node = {
        "__spz_type__": "DatasetIndex",
        "__spz_uid__": "RBSP-A_MAGNETOMETER_4SEC-GSM_EMFISIS-L3",
        "description": "Fluxgate magnetometer data - Craig Kletzing (University of Iowa)",
        "serviceprovider_ID": "RBSP-A_MAGNETOMETER_4SEC-GSM_EMFISIS-L3",
        "spase_DatasetResourceID": "spase://NASA/NumericalData/RBSP/A/EMFISIS/MAGNETOMETER/L3/GSM/PT4S",
    }
    meta = _extract_dataset_meta(node, "cda")
    assert meta == {
        "dataset_description": "Fluxgate magnetometer data - Craig Kletzing (University of Iowa)",
        "dataset_id": "RBSP-A_MAGNETOMETER_4SEC-GSM_EMFISIS-L3",
        "spase": "RBSP A EMFISIS MAGNETOMETER L3 GSM",
        "cadence": "4 s",
    }
    text = _build_text(
        "Mag",
        "Magnetometer vector",
        "nT",
        "RBSP-A_MAGNETOMETER_4SEC-GSM_EMFISIS-L3/Mag",
        parent_meta=meta,
        region="Earth.Magnetosphere.RadiationBelt",
        coverage=("2012-08-30", "2019-10-14"),
        components=["Bx_GSM", "By_GSM", "Bz_GSM"],
    )
    assert text == (
        "Mag. Magnetometer vector. Dataset: RBSP-A_MAGNETOMETER_4SEC-GSM_EMFISIS-L3. "
        "SPASE: RBSP A EMFISIS MAGNETOMETER L3 GSM. Cadence: 4 s. "
        "Fluxgate magnetometer data - Craig Kletzing (University of Iowa). "
        "Components: Bx_GSM, By_GSM, Bz_GSM. Units: nT. Region: Earth.Magnetosphere.RadiationBelt. "
        "Coverage: 2012-08-30 to 2019-10-14."
    )
    blank = _extract_dataset_meta(
        {"description": "d", "serviceprovider_ID": "SOLO_L2_MAG", "spase_DatasetResourceID": " "},
        "cda",
    )
    assert blank == {"dataset_description": "d", "dataset_id": "SOLO_L2_MAG"}, (
        "a blank SPASE id adds nothing"
    )


def test_walk_writes_cda_components_into_the_text():
    from helioai.indexer import _walk

    param = _make_tree(
        __spz_type__="ParameterIndex",
        __spz_uid__="RBSP-A_MAGNETOMETER_4SEC-GSM_EMFISIS-L3/Mag",
        FIELDNAM="Mag",
        CATDESC="Magnetometer vector",
        UNITS="nT",
        LABL_PTR_1=["Bx_GSM  ", "By_GSM  ", "Bz_GSM  "],
        start_date="2012-08-30 22:54:52",
        stop_date="2019-10-14 12:30:28",
    )
    dataset = _make_tree(
        __spz_type__="DatasetIndex",
        __spz_uid__="RBSP-A_MAGNETOMETER_4SEC-GSM_EMFISIS-L3",
        description="Fluxgate magnetometer data",
        serviceprovider_ID="RBSP-A_MAGNETOMETER_4SEC-GSM_EMFISIS-L3",
        spase_DatasetResourceID="spase://NASA/NumericalData/RBSP/A/EMFISIS/MAGNETOMETER/L3/GSM/PT4S",
        Mag=param,
    )
    docs: list[dict] = []
    _walk(_make_tree(ds=dataset), "cda", docs, set(), _FakeSpeasyIndex)
    assert len(docs) == 1 and docs[0]["id"] == "cda/RBSP-A_MAGNETOMETER_4SEC-GSM_EMFISIS-L3/Mag"
    text = docs[0]["text"]
    assert "Dataset: RBSP-A_MAGNETOMETER_4SEC-GSM_EMFISIS-L3." in text
    assert "SPASE: RBSP A EMFISIS MAGNETOMETER L3 GSM. Cadence: 4 s." in text
    assert "Components: Bx_GSM, By_GSM, Bz_GSM." in text
    assert docs[0]["meta"]["region_source"] == "table"
