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
