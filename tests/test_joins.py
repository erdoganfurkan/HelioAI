"""The four joins of `core/joins.py`: exact operations on typed fields, no model, and
`None` whenever there is nothing to compare. Each test is one of the failures the join
exists for, built from the cards and claims a turn actually carries."""

from __future__ import annotations

from helioai.core import joins
from helioai.tools import rag


def _card(param_id, **fields):
    return {"kind": "parameter_card", "tool": "run_python", "param_id": param_id, **fields}


def _contract(**fields):
    base = {
        "deliverable": None,
        "quantity": None,
        "frame": None,
        "event_named": None,
        "uncertainty_required": None,
        "method_named": None,
        "two_spacecraft": None,
        "date": None,
        "date_precision": None,
    }
    base.update(fields)
    return base


def test_frame_asked_against_the_frames_the_sandbox_loaded():
    cards = [_card("amda/wnd_b_gse", coord_sys="GSE"), _card("cda/X/Y", coord_sys="gse")]
    assert joins.checks(_contract(frame="GSM"), cards, [])["frame"] == {
        "asked": "GSM",
        "loaded": ["GSE"],
        "match": False,
    }
    assert joins.checks(_contract(frame="GSE"), cards, [])["frame"]["match"] is True
    assert joins.checks(_contract(frame="GSM"), [_card("amda/x")], [])["frame"] is None
    assert joins.checks(_contract(), cards, [])["frame"] is None
    assert joins.checks(_contract(frame="other"), cards, [])["frame"] is None


def test_window_join_reads_obtained_bounds_and_short_series():
    cards = [
        _card(
            "amda/wnd_b_gse",
            start="2004-11-07T00:00:00",
            stop="2004-11-08T00:00:00",
            obtained_start="2004-11-07T00:00:02",
            obtained_stop="2004-11-07T17:59:00",
            coverage_note="obtained 2004-11-07 00:00 → 17:59 (asked 00:00 → 2004-11-08 00:00)",
        ),
        _card("amda/wnd_swe_n", start="2004-11-07T00:00:00", stop="2004-11-08T00:00:00"),
    ]
    day = joins.checks(_contract(date="2004-11-07", date_precision="day"), cards, [])["window"]
    assert day == {"date": "2004-11-07", "covered": True, "short": ["amda/wnd_b_gse"]}

    slipped = joins.checks(_contract(date="2004-11-09", date_precision="day"), cards, [])
    assert slipped["window"]["covered"] is False

    month = joins.checks(_contract(date="2004-11", date_precision="month"), cards, [])
    assert month["window"]["covered"] is True
    year = joins.checks(_contract(date="2003", date_precision="year"), cards, [])
    assert year["window"]["covered"] is False

    no_date = joins.checks(_contract(), cards, [])["window"]
    assert no_date == {"date": None, "covered": None, "short": ["amda/wnd_b_gse"]}
    assert joins.checks(_contract(), [_card("amda/x")], [])["window"] is None
    unbounded = joins.checks(_contract(date="2004-11-07", date_precision="day"), [_card("a/b")], [])
    assert unbounded["window"] is None, "no bound on any card: the join cannot say"


def test_quantity_join_compares_families_from_the_index_by_import(monkeypatch):
    cards = [
        _card("amda/wnd_b_gse"),
        _card("csa/C1_CP_CIS-HIA_ONBOARD_MOMENTS/density"),
        _card("cda/U/T"),
    ]
    types = {
        "amda/wnd_b_gse": "MagneticField",
        "csa/C1_CP_CIS-HIA_ONBOARD_MOMENTS/density": "Ion_Composition, Thermal_Plasma",
        "cda/U/T": None,
    }
    seen = []

    def fake_lookup(ids):
        seen.append(sorted(ids))
        return {i: types.get(i) for i in ids}

    monkeypatch.setattr(rag, "measurement_types_of", fake_lookup)

    thermal = joins.checks(_contract(quantity="ThermalPlasma"), cards, [])["quantity"]
    assert thermal == {
        "asked": "ThermalPlasma",
        "loaded": {"MagneticField": 1, "Ion_Composition, Thermal_Plasma": 1},
        "untyped": 1,
        "match": True,
    }
    assert seen == [sorted(c["param_id"] for c in cards)]

    ephemeris = joins.checks(_contract(quantity="Ephemeris"), cards, [])["quantity"]
    assert ephemeris["match"] is False, "nothing loaded is a position"

    only_untyped = joins.checks(_contract(quantity="MagneticField"), [_card("cda/U/T")], [])
    assert only_untyped["quantity"] == {
        "asked": "MagneticField",
        "loaded": {},
        "untyped": 1,
        "match": None,
    }
    assert joins.checks(_contract(quantity="MagneticField"), [], [])["quantity"] is None
    assert joins.checks(_contract(), cards, [])["quantity"] is None
    assert joins.checks(_contract(quantity="Unknown"), cards, [])["quantity"] is None


def test_the_families_are_the_ones_the_ranking_uses():
    for name, family in rag.TYPE_FAMILIES.items():
        assert rag._types_of(name) <= family or name in ("Spectrum", "Waves"), name
    assert rag._wanted_type("MMS1 FGM magnetic field") == rag.TYPE_FAMILIES["MagneticField"]
    assert rag._wanted_type("proton density") == rag.TYPE_FAMILIES["ThermalPlasma"]


def test_measurement_types_of_maps_unknown_and_unreachable_to_none(monkeypatch):
    class Collection:
        def get(self, ids, include):
            assert include == ["metadatas"]
            return {"ids": ["a/b"], "metadatas": [{"measurement_type": "MagneticField"}]}

    monkeypatch.setattr(rag, "_collection_only", lambda: Collection())
    assert rag.measurement_types_of(["a/b", "c/d"]) == {"a/b": "MagneticField", "c/d": None}
    assert rag.measurement_types_of([]) == {}

    def broken():
        raise RuntimeError("no index")

    monkeypatch.setattr(rag, "_collection_only", broken)
    assert rag.measurement_types_of(["a/b"]) == {"a/b": None}


def test_responsiveness_reads_claims_exports_and_figures():
    artifacts = [
        {"kind": "image", "tool": "run_python", "figure_paths": ["f1.png"]},
        {"kind": "exports", "tool": "run_python", "values": {"theta_bn_deg": 54.3}},
    ]
    claims = [{"name": "theta_bn_deg", "value": 54.3, "units": "deg", "source": "recipe"}]

    asked = _contract(deliverable="value", uncertainty_required=True)
    resp = joins.checks(asked, artifacts, claims)["responsiveness"]
    assert resp["uncertainty"] == {"required": True, "claimed": False}
    assert resp["deliverable"] == {"asked": "value", "claims": 1, "exports": 1, "match": True}

    with_spread = artifacts + [
        {"kind": "exports", "tool": "run_python", "values": {"theta_bn_window_spread_deg": 4.3}}
    ]
    assert joins.checks(asked, with_spread, claims)["responsiveness"]["uncertainty"]["claimed"]

    figure = joins.checks(_contract(deliverable="figure"), [], [])["responsiveness"]
    assert figure["deliverable"] == {"asked": "figure", "figures": 0, "match": False}
    catalogue = joins.checks(_contract(deliverable="catalogue"), [], [])["responsiveness"]
    assert catalogue["deliverable"]["match"] is False
    explanation = joins.checks(_contract(deliverable="explanation"), [], [])["responsiveness"]
    assert explanation["deliverable"] == {"asked": "explanation", "match": None}
    assert joins.checks(_contract(), [], [])["responsiveness"] is None


def test_summary_names_only_the_joins_that_disagreed():
    assert joins.summary(None) == [] and joins.summary({}) == []
    agreed = {
        "frame": {"asked": "GSE", "loaded": ["GSE"], "match": True},
        "window": {"date": "2004-11-07", "covered": True, "short": []},
        "quantity": None,
        "responsiveness": {"uncertainty": {"required": True, "claimed": True}},
    }
    assert joins.summary(agreed) == []
    disagreed = {
        "frame": {"asked": "GSM", "loaded": ["GSE"], "match": False},
        "window": {"date": "2004-11-09", "covered": False, "short": ["amda/wnd_b_gse"]},
        "quantity": {
            "asked": "Ephemeris",
            "loaded": {"MagneticField": 1},
            "untyped": 0,
            "match": False,
        },
        "responsiveness": {
            "uncertainty": {"required": True, "claimed": False},
            "deliverable": {"asked": "figure", "figures": 0, "match": False},
        },
    }
    assert joins.summary(disagreed) == [
        "frame GSM asked, GSE loaded",
        "2004-11-09 not in any loaded series",
        "1 series short of the window asked",
        "Ephemeris asked, loaded MagneticField",
        "uncertainty asked, none claimed",
        "figure asked, none produced",
    ]


def test_the_cli_line_carries_the_mismatches():
    from helioai.interfaces.cli import _intent_line

    data = {
        "deliverable": "value",
        "quantity": "MagneticField",
        "uncertainty_required": True,
        "checks": {"responsiveness": {"uncertainty": {"required": True, "claimed": False}}},
    }
    assert _intent_line(data) == (
        "deliverable: value · quantity: MagneticField · uncertainty required ⚠ uncertainty asked, none claimed"
    )
    assert _intent_line({"deliverable": "figure", "checks": {}}) == "deliverable: figure"
