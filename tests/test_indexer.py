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
