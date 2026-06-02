"""Unit tests for helioai.indexer: _get_region, _build_text, _walk."""

from __future__ import annotations

import pytest

from helioai.indexer import _build_text, _get_region, _walk


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
