import json

import pytest

from helioai import provenance
from helioai.core.tool_exec import emit_post_tool_events


def _run_python_result(session_dir, run_idx, exports):
    return json.dumps(
        {
            "stdout": "",
            "figure_paths": [],
            "exports": exports,
            "cards": [],
            "code_path": str(session_dir / f"code_{run_idx}.py"),
            "n_lines": 3,
        }
    )


def _stats(mean, units=""):
    return {"mean": mean, "min": mean - 1, "max": mean + 1, "std": 0.5, "units": units}


def test_two_runs_append_to_one_ledger(tmp_path):
    list(
        emit_post_tool_events(
            "run_python",
            _run_python_result(tmp_path, 0, {"Bd": _stats(14.5, "nT")}),
            tool_result_extra={"turn": 2},
        )
    )
    list(
        emit_post_tool_events(
            "run_python",
            _run_python_result(tmp_path, 3, {"r_B": _stats(2.2)}),
            tool_result_extra={"turn": 4},
            common_extra={"sub_agent_ctx": {"role": "data_analyst", "task_id": "a1b2c3d4"}},
        )
    )

    values = provenance.read_ledger(tmp_path)["values"]
    assert [v["name"] for v in values] == ["Bd", "r_B"]

    bd, rb = values
    assert bd["code_path"] == str(tmp_path / "code_0.py")
    assert bd["run_idx"] == 0
    assert bd["units"] == "nT"
    assert bd["agent"] == "lead"
    assert bd["task_id"] is None
    assert bd["turn"] == 2

    assert rb["run_idx"] == 3
    assert rb["agent"] == "data_analyst"
    assert rb["task_id"] == "a1b2c3d4"
    assert rb["mean"] == pytest.approx(2.2)


def test_missing_or_corrupt_ledger_reads_empty(tmp_path):
    assert provenance.read_ledger(tmp_path) == {"values": []}
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "provenance.json").write_text("{not json", encoding="utf-8")
    assert provenance.read_ledger(tmp_path) == {"values": []}


def test_a_failed_export_is_not_recorded(tmp_path):
    provenance.record(
        {"bad": {"error": "boom", "repr": "<obj>"}, "good": _stats(1.0)},
        code_path=str(tmp_path / "code_0.py"),
    )
    assert [v["name"] for v in provenance.read_ledger(tmp_path)["values"]] == ["good"]


def test_record_never_raises_on_an_unwritable_session(tmp_path):
    provenance.record({"x": _stats(1.0)}, code_path="")
    provenance.record({}, code_path=str(tmp_path / "code_0.py"))
    assert provenance.read_ledger(tmp_path) == {"values": []}


def test_find_value_returns_the_latest_and_match_number_finds_any_statistic(tmp_path):
    provenance.record({"Bd": _stats(14.5, "nT")}, code_path=str(tmp_path / "code_0.py"))
    provenance.record({"Bd": _stats(21.67, "nT")}, code_path=str(tmp_path / "code_1.py"))

    assert provenance.find_value(tmp_path, "Bd")["mean"] == pytest.approx(21.67)
    assert provenance.find_value(tmp_path, "nope") is None

    assert len(provenance.match_number(tmp_path, 14.5)) == 1
    assert len(provenance.match_number(tmp_path, 15.5)) == 1  # the max of the first entry
    # the number actually published in the notebook, computed by nothing
    assert provenance.match_number(tmp_path, 13.02) == []


def test_export_accepts_units_and_defaults_to_empty():
    import numpy as np

    from helioai.tools.sandbox import _SANDBOX_PREAMBLE

    src = _SANDBOX_PREAMBLE[_SANDBOX_PREAMBLE.index("def export(") :]
    ns = {"np": np, "__sandbox_exports": {}}
    exec(src[: src.index("\ndef clean(")], ns)

    ns["export"]("B", [1.0, 2.0, 3.0], units="nT")
    ns["export"]("r", [2.0])
    assert ns["__sandbox_exports"]["B"]["units"] == "nT"
    assert ns["__sandbox_exports"]["B"]["mean"] == pytest.approx(2.0)
    assert ns["__sandbox_exports"]["r"]["units"] == ""


def test_export_flattens_a_summary_dict_instead_of_failing_on_it():
    import numpy as np

    from helioai.tools.sandbox import _SANDBOX_PREAMBLE

    src = _SANDBOX_PREAMBLE[_SANDBOX_PREAMBLE.index("def export(") :]
    ns = {"np": np, "__sandbox_exports": {}}
    exec(src[: src.index("\ndef clean(")], ns)

    ns["export"](
        "shock",
        {"time": "2003-10-29 06:25:40", "B_up": 21.24, "downstream": {"B": 42.16}},
        units="nT",
    )
    exports = ns["__sandbox_exports"]
    assert sorted(exports) == ["shock.B_up", "shock.downstream.B"]
    assert exports["shock.B_up"]["mean"] == pytest.approx(21.24)
    assert exports["shock.downstream.B"]["units"] == "nT"
    assert not any("error" in e for e in exports.values())


def test_findings_are_built_from_the_exports_of_the_run(tmp_path):
    from helioai.core.sub_agents import _findings

    artifacts = [
        {"kind": "code", "code_path": str(tmp_path / "code_0.py")},
        {
            "kind": "exports",
            "code_path": str(tmp_path / "code_0.py"),
            "values": {
                "r_B": {"mean": 2.53, "min": 2.53, "max": 2.53, "units": ""},
                "Bd": {"mean": 14.5, "min": 9.1, "max": 21.7, "units": "nT"},
                "oops": {"error": "boom"},
            },
        },
    ]
    findings = _findings(artifacts)

    assert set(findings) == {"r_B", "Bd"}
    assert findings["r_B"] == {"value": 2.53, "units": "", "code_path": str(tmp_path / "code_0.py")}
    assert "min" not in findings["r_B"]
    assert findings["Bd"]["min"] == 9.1
    assert findings["Bd"]["units"] == "nT"


def test_findings_survive_the_compaction_that_trims_a_stale_result():
    from helioai.core.tool_exec import _summarize_tool_result

    result = json.dumps(
        {
            "findings": {"Bd": {"value": 14.5, "units": "nT", "min": 9.1, "max": 21.7}},
            "summary": "the shock " * 200,
            "n_iterations": 4,
            "error": None,
            "artifacts": [],
        }
    )
    compacted = _summarize_tool_result(result, max_chars=300)
    assert "14.5 nT" in compacted
    assert "21.7" in compacted


def test_a_capped_run_keeps_its_findings_through_compaction():
    from helioai.core.tool_exec import _summarize_tool_result

    result = json.dumps(
        {
            "findings": {"Bd": {"value": 14.5, "units": "nT"}},
            "summary": "",
            "n_iterations": 8,
            "error": "(sub-agent 'data_analyst' reached its 8-turn cap)",
            "artifacts": [],
        }
    )
    compacted = _summarize_tool_result(result, max_chars=300)
    assert "14.5 nT" in compacted
    assert "cap" in compacted


def _reply_e14084cf():
    return """Key upstream/downstream values

- Upstream |B|:
  - 9.17 nT

- Downstream |B|:
  - 13.02 nT

Compression ratios

- Density compression:
  - r_n = n2/n1 = 2.06

Recorded on 2015-03-17T04:30:59 UT, over 2 spacecraft, Act III of 3.
Alfven speed: VA ~ 48.9 km/s, Mach number MA ~ 4.0.
"""


def test_extract_claims_keeps_measurements_and_drops_dates_and_counts():
    from helioai.core.provenance_check import extract_claims

    texts = [c.text for c in extract_claims(_reply_e14084cf())]

    assert "13.02 nT" in texts
    assert "48.9 km/s" in texts
    assert "2.06" in texts
    assert "4.0" in texts  # written with a decimal: a claim, not a count

    assert not any("2015" in t for t in texts)
    assert not any(t.startswith("03") or t.startswith("17") for t in texts)
    assert "2" not in texts  # "2 spacecraft"
    assert "3" not in texts  # "Act III of 3"


def test_verify_contradicts_a_number_the_session_computed_differently():
    from helioai.core.provenance_check import extract_claims, verify

    ledger = {
        "values": [
            {
                "name": "B_downstream",
                "units": "nT",
                "mean": 14.5,
                "min": 12.0,
                "max": 21.7,
                "code_path": "/w/code_2.py",
            },
            {"name": "alfven_speed", "units": "km/s", "mean": 48.9, "code_path": "/w/code_3.py"},
        ]
    }
    report = verify(extract_claims(_reply_e14084cf()), ledger)

    flagged = {d["text"]: d for d in report.details}
    assert flagged["13.02 nT"]["status"] == "contradicted"
    assert flagged["13.02 nT"]["name"] == "B_downstream"
    assert flagged["13.02 nT"]["code_path"] == "/w/code_2.py"
    assert report.matched >= 1  # 48.9 km/s is in the ledger
    assert "48.9 km/s" not in flagged


def test_a_unit_mismatch_is_not_a_contradiction():
    from helioai.core.provenance_check import extract_claims, verify

    ledger = {"values": [{"name": "downstream density", "units": "cm-3", "mean": 34.5}]}
    (claim,) = [c for c in extract_claims("Downstream density peak: 41.0 nT") if c.units == "nT"]
    report = verify([claim], ledger)

    assert report.contradicted == 0
    assert report.unsourced == 1


def test_ratios_and_percentages_are_derived_not_unsourced():
    from helioai.core.provenance_check import extract_claims, verify

    ledger = {"values": [{"name": "n_up", "units": "cm-3", "mean": 16.7}]}
    report = verify(extract_claims("The ratio is 2.06 and that is 29.4 % of the total."), ledger)

    assert report.derived == 2
    assert report.unsourced == 0
    assert report.details == []


def test_check_reply_stays_silent_when_the_session_computed_nothing(tmp_path):
    from helioai.core.provenance_check import check_reply

    assert check_reply("The field reached 13.02 nT.", tmp_path) is None

    provenance.record({"Bd": _stats(14.5, "nT")}, code_path=str(tmp_path / "code_0.py"))
    payload = check_reply("The field reached 13.02 nT.", tmp_path)
    assert payload["unsourced"] + payload["contradicted"] == 1


def test_the_provenance_event_is_emitted_next_to_the_reply(tmp_path, monkeypatch):
    import helioai.workspace as ws
    from helioai.core.agent_loop import _provenance_events

    provenance.record({"Bd": _stats(14.5, "nT")}, code_path=str(tmp_path / "code_0.py"))
    monkeypatch.setattr(ws, "get_session_dir", lambda: tmp_path)

    (event,) = list(_provenance_events("Downstream |B| reached 13.02 nT."))
    assert event["event"] == "provenance"
    assert event["data"]["unsourced"] + event["data"]["contradicted"] == 1

    assert list(_provenance_events("")) == []


def test_a_broken_check_never_costs_the_user_the_reply(monkeypatch):
    import helioai.core.provenance_check as pc
    from helioai.core.agent_loop import _provenance_events

    def boom(*a, **k):
        raise RuntimeError("ledger on fire")

    monkeypatch.setattr(pc, "check_reply", boom)
    assert list(_provenance_events("13.02 nT")) == []


def test_a_generic_name_does_not_accuse_the_words_around_any_number():
    """Regression: fifteen contradictions in one notebook run, all false.

    `min_Bz_nT` splits to min/bz/nt and the length filter keeps only "min", which was
    then matched as a substring — so "Minimum-variance", "30-min medians" and the year
    in "Sonnerup & Cahill 1967" each got reported as contradicting the minimum Bz. And
    `wind_Np_upstream_window_n` found its own word "wind" inside "window". A detector
    that cries wolf teaches the reader to skip the line.
    """
    from helioai.core.provenance_check import _named_entry

    entries = [
        {"name": "min_Bz_nT", "units": ""},
        {"name": "wind_Np_upstream_window_n", "units": ""},
    ]
    for context in (
        "Minimum-variance (Sonnerup & Cahill 1967)",
        "30-min medians from the [-60, -30] min windows",
        "the upstream-side window alone gives n = (+0.04",
        "MVA eigenvalue ratio (intermediate / minimum) is 1.32",
    ):
        assert _named_entry(context, "", entries) is None, context


def test_a_name_written_out_in_prose_is_still_matched():
    """The precision fix must not silence the case the check exists for."""
    from helioai.core.provenance_check import _named_entry

    entries = [{"name": "compression_ratio_density", "units": ""}]
    named = _named_entry("the density compression ratio is 2.46", "", entries)
    assert named is not None and named["name"] == "compression_ratio_density"


def test_a_day_of_the_month_is_not_a_contradicted_measurement():
    """A bare 17 next to `compression_ratio` is the date, not a rival value for the ratio.

    HelioBench failed two runs of `n3_field_compression` on this: the reply opened with
    "the magnetic compression ratio across the 17 March 2015 shock is 2.59", the ledger
    held compression_ratio = 2.59, and the checker read the day of the month inside the
    40-character name window as a number the session computed differently. The answer was
    right and agreed with its own ledger, so the gate failed a correct run.
    """
    from helioai.core.provenance_check import extract_claims, verify

    ledger = {"values": [{"name": "compression_ratio", "units": "", "mean": 2.59}]}
    reply = "The magnetic compression ratio across the 17 March 2015 shock is 2.59."
    report = verify(extract_claims(reply), ledger)

    assert report.contradicted == 0, [d["text"] for d in report.details]
    assert report.matched >= 1


# ── decimal commas ─────────────────────────────────────────────────────────────


def _claims(text):
    from helioai.core.provenance_check import extract_claims

    return {(c.value, c.units) for c in extract_claims(text)}


def test_a_decimal_comma_is_one_number_not_two():
    """A reply in French writes 9,79 nT. The old pattern started a token after the
    comma, so the ledger held 9.79, the reply appeared to claim 79, and every real
    measurement came back `unsourced` — which is how a demo ends up 0 traced / 6
    unsourced while the physics was right."""
    got = _claims("|B| avant le choc ≈ 9,79 nT et après ≈ 24,92 nT")
    assert (9.79, "nT") in got
    assert (24.92, "nT") in got
    assert (79.0, "nT") not in got
    assert (92.0, "nT") not in got


def test_a_thousands_comma_is_still_a_thousands_comma():
    """English prose says "1,800 points"; reading that as 1.8 would be worse."""
    assert (1800.0, "") in _claims("downloaded 1,800 points over the interval")


def test_a_decimal_point_still_works():
    assert (2.59, "") in _claims("compression ratio 2.59 measured")


def test_a_decimal_comma_number_is_treated_as_a_measurement_not_a_count():
    """9,79 is written with a fraction, so the bare-integer floor must not drop it."""
    assert (9.79, "") in _claims("le rapport vaut 9,79 selon le calcul")


# ── vectors ────────────────────────────────────────────────────────────────────

_SHOCK_NORMAL_MC = {
    "units": "",
    "shape": [3],
    "dtype": "float64",
    "min": -0.6609380065362107,
    "max": 0.36192522480537503,
    "mean": -0.31880332840923864,
    "std": 0.4813499466381236,
    "n_finite": 3,
    "n_nan": 0,
    "sample": [-0.6609380065362107, -0.6573972034968802, 0.36192522480537503],
}


def test_a_component_of_an_exported_vector_is_sourced_not_contradicted(tmp_path):
    """Replayed from HelioBench `n3_theta_bn.2`, whose reply and exports are verbatim here.

    The run was right and agreed with its own ledger, and the checker still reported
    `contradicted: -0.65739720 | shock_normal_mc` — the middle component of the very
    vector that entry records. min and max vouched for the two outer components, and a
    three-component export can only ever cover two of its three numbers that way.
    """
    from helioai.core.provenance_check import check_reply

    provenance.record(
        {"shock_normal_mc": _SHOCK_NORMAL_MC, "theta_Bn_deg": _stats(57.49625747764631, "deg")},
        code_path=str(tmp_path / "code_2.py"),
    )
    reply = (
        "the resulting unit shock normal is:\n"
        "- n = [-0.66093801, -0.65739720, 0.36192522]\n"
        "The shock normal angle relative to the upstream magnetic field is:\n"
        "- theta_Bn = 57.49625747764631 deg"
    )

    payload = check_reply(reply, tmp_path)
    assert payload["contradicted"] == 0, payload["details"]
    assert payload["matched"] == 4


def test_a_ledger_written_before_shape_was_recorded_still_accuses(tmp_path):
    """Old sessions keep the old verdict: an unknown shape must not be read as a vector."""
    from helioai.core.provenance_check import check_reply

    provenance.record({"compression_ratio": _stats(2.59)}, code_path=str(tmp_path / "code_0.py"))
    entries = provenance.read_ledger(tmp_path)["values"]
    assert entries[0]["shape"] is None

    payload = check_reply("The density compression ratio is 3.80.", tmp_path)
    assert payload["contradicted"] == 1


def test_a_scalar_export_is_still_contradicted_by_a_different_number(tmp_path):
    """The narrowing must not make the check mute — this is what it exists for."""
    from helioai.core.provenance_check import check_reply

    scalar = dict(_stats(2.59), shape=[], sample=[2.59])
    provenance.record({"compression_ratio": scalar}, code_path=str(tmp_path / "code_0.py"))

    payload = check_reply("The density compression ratio is 3.80.", tmp_path)
    assert payload["contradicted"] == 1
    assert payload["details"][0]["name"] == "compression_ratio"


def test_a_long_series_is_judged_on_its_statistics_not_on_its_first_samples(tmp_path):
    """`sample` is the first eight values of a flattened array. Trusting it on a 1800-point
    series would source any number that happened to sit near its start, so the whole-array
    rule requires the sample to hold every value."""
    from helioai.core.provenance_check import _whole_array, check_reply

    series = {
        "units": "nT",
        "shape": [1800],
        "min": 8.0,
        "max": 25.0,
        "mean": 17.0,
        "std": 4.0,
        "sample": [9.11, 9.14, 9.2, 9.3, 9.25, 9.4, 9.35, 9.5],
    }
    assert _whole_array(series) == []
    provenance.record({"b_series": series}, code_path=str(tmp_path / "code_0.py"))

    payload = check_reply("The field starts at 9.14 nT and peaks at 25.0 nT.", tmp_path)
    assert payload["matched"] == 1
    assert payload["unsourced"] == 1


def test_the_name_nearest_the_number_is_the_one_accused():
    """Two quantities in one window: the sentence is about the nearer one.

    The rule used to be the longest name, which here blames the shock speed for a number
    written about the upstream flow. It also decides the verdict now that only a scalar
    entry can support a contradiction — whichever of a vector and a scalar the window
    picks is the difference between an accusation and none.
    """
    from helioai.core.provenance_check import _named_entry, extract_claims

    entries = [
        {"name": "shock_speed_km_s", "units": "km/s", "mean": 571.07},
        {"name": "upstream_flow", "units": "km/s", "mean": 409.99},
    ]
    reply = "shock speed 571.07 km/s, upstream flow 421.0 km/s"
    (claim,) = [c for c in extract_claims(reply) if c.value == 421.0]

    assert _named_entry(claim.context, claim.units, entries, claim.pos)["name"] == "upstream_flow"
    assert _named_entry(claim.context, claim.units, entries)["name"] == "shock_speed_km_s"


# ── E7: a number is only sourced by the quantity it claims to be ───────────────


def test_a_matching_number_of_another_quantity_does_not_source_a_claim():
    """Audit probe: ledger held B_downstream = 10 nT and density = 25 cm-3; the reply said
    "B downstream = 25 nT" and came back matched=1, contradicted=0. The bare numeric
    coincidence with the density disculpated a wrong field value — the mechanism this
    module exists to expose, defeated by its own lookup order.
    """
    from helioai.core.provenance_check import extract_claims, verify

    ledger = {
        "values": [
            {"name": "B_downstream", "units": "nT", "mean": 10.0, "shape": [], "sample": [10.0]},
            {"name": "density", "units": "cm-3", "mean": 25.0, "shape": [], "sample": [25.0]},
        ]
    }
    report = verify(extract_claims("B downstream = 25 nT"), ledger)

    assert report.matched == 0
    assert report.contradicted == 1
    assert report.details[0]["name"] == "B_downstream"


def test_a_unitless_export_still_sources_a_claim_written_with_its_unit():
    """Most exports carry no unit — the model writes it into the name. Requiring unit
    agreement would have turned every one of those into `unsourced` and made the check
    blind again; only a *conflicting* unit disqualifies.
    """
    from helioai.core.provenance_check import extract_claims, verify

    ledger = {
        "values": [{"name": "B_up_nT", "units": "", "mean": 9.79, "shape": [], "sample": [9.79]}]
    }
    report = verify(extract_claims("upstream field 9.79 nT"), ledger)
    assert report.matched == 1


def test_the_named_scalar_wins_over_a_coincidental_unitless_hit():
    from helioai.core.provenance_check import extract_claims, verify

    ledger = {
        "values": [
            {"name": "B_downstream", "units": "nT", "mean": 10.0, "shape": [], "sample": [10.0]},
            {"name": "ratio", "units": "", "mean": 25.0, "shape": [], "sample": [25.0]},
        ]
    }
    report = verify(extract_claims("B downstream = 25 nT"), ledger)
    assert report.contradicted == 1
    assert report.matched == 0


def test_a_negative_claim_is_not_sourced_by_a_positive_record():
    """Magnitudes are compared on purpose — "a 464.5 s lag" quotes a recorded -464.5 — but
    the other way round is not a magnitude: a reply stating Bz = -10 nT for a recorded
    +10 has the sign wrong, and sign is the physics of Bz.
    """
    from helioai.core.provenance_check import extract_claims, verify

    ledger = {
        "values": [{"name": "Bz_min", "units": "nT", "mean": 10.0, "shape": [], "sample": [10.0]}]
    }
    report = verify(extract_claims("Bz min reached -10 nT"), ledger)
    assert report.matched == 0
    assert report.contradicted == 1

    lag = {
        "values": [{"name": "lag_s", "units": "s", "mean": -464.5, "shape": [], "sample": [-464.5]}]
    }
    assert verify(extract_claims("a lag of 464.5 s"), lag).matched == 1


def test_a_three_decimal_comma_in_a_comma_decimal_reply_is_a_decimal():
    """Audit probe: `Bz = -0,657 nT` was read as -657 nT. Three digits after the comma
    are a thousands separator in "1,800 points" — and a plain decimal in a French reply.
    A leading zero settles it on its own; elsewhere the rest of the reply does.
    """
    assert (-0.657, "nT") in _claims("Bz = -0,657 nT")
    assert (657.0, "nT") not in _claims("Bz = -0,657 nT")
    # the reply already uses a comma as its decimal mark elsewhere → 2,150 is 2.15
    got = _claims("|B| = 9,79 nT et la vitesse 2,150 km/s")
    assert (2.15, "km/s") in got
    assert (2150.0, "km/s") not in got
    # nothing marks the reply as comma-decimal → English thousands, unchanged behaviour
    assert (1800.0, "") in _claims("downloaded 1,800 points over the interval")


def test_scientific_notation_keeps_its_exponent_and_unit():
    assert (1.2e-3, "Hz") in _claims("the spectral peak sits at 1.2e-3 Hz")
    assert (1.2, "") not in _claims("the spectral peak sits at 1.2e-3 Hz")
    assert (3.5e5, "K") in _claims("temperature 3.5E+05 K")


def test_a_degree_sign_in_the_reply_matches_deg_in_the_ledger():
    """Live run, 2026-09-11: the recipe exported theta_Bn_deg = 62.68 with units="deg";
    the reply wrote "62.68 °" and the line under the answer read `unsourced` — for the
    one number the whole demo is about. `°` and `deg` are the same unit.
    """
    from helioai.core.provenance_check import extract_claims, verify

    ledger = {
        "values": [
            {"name": "theta_Bn_deg", "units": "deg", "mean": 62.68, "shape": [], "sample": [62.68]}
        ]
    }
    report = verify(extract_claims("θ_Bn = 62.68° (quasi-perpendicular)"), ledger)
    assert report.matched == 1, report.details
    assert report.unsourced == 0
