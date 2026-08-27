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
