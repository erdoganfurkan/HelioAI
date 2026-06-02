"""Unit tests for agent_loop helpers: _summarize_tool_result, _extract_artifact."""

from __future__ import annotations

import json


from helioai.core.agent_loop import _extract_artifact, _summarize_tool_result


# ──────────────────────────────── _summarize_tool_result ────────────────────


def test_summarize_non_json_truncated() -> None:
    summary = _summarize_tool_result("hello world", max_chars=5)
    assert summary == "hello"


def test_summarize_list_result() -> None:
    summary = _summarize_tool_result(json.dumps([{"id": "a"}, {"id": "b"}]))
    assert "[list, 2 items]" in summary


def test_summarize_error_is_concise() -> None:
    summary = _summarize_tool_result(json.dumps({"error": "boom"}))
    assert summary.startswith("error: boom")


def test_summarize_figure_paths_returns_filename_only() -> None:
    payload = json.dumps(
        {
            "figure_paths": ["/tmp/helioai_abc/fig_0.png", "/tmp/helioai_abc/fig_1.png"],
            "stdout": "done",
        }
    )
    summary = _summarize_tool_result(payload)
    data = json.loads(summary)
    assert data["figure_paths"] == ["fig_0.png", "fig_1.png"]


def test_summarize_empty_figure_paths() -> None:
    payload = json.dumps({"figure_paths": [], "stdout": "no plot"})
    summary = _summarize_tool_result(payload)
    data = json.loads(summary)
    assert data["figure_paths"] == []


def test_summarize_long_string_truncated() -> None:
    payload = json.dumps({"preview": "X" * 200})
    summary = _summarize_tool_result(payload)
    data = json.loads(summary)
    assert data["preview"].endswith("...")
    assert len(data["preview"]) <= 120


def test_summarize_nested_list_compact() -> None:
    payload = json.dumps({"results": [1, 2, 3, 4, 5]})
    summary = _summarize_tool_result(payload)
    data = json.loads(summary)
    assert data["results"] == "[5 items]"


def test_summarize_nested_dict_compact() -> None:
    payload = json.dumps({"exports": {"a": 1, "b": 2, "c": 3}})
    summary = _summarize_tool_result(payload)
    data = json.loads(summary)
    assert data["exports"] == "{3 keys}"


def test_summarize_scalar_fields_kept() -> None:
    payload = json.dumps({"n_points": 1238, "units": "nT", "ok": True})
    summary = _summarize_tool_result(payload)
    data = json.loads(summary)
    assert data["n_points"] == 1238
    assert data["units"] == "nT"
    assert data["ok"] is True


# ──────────────────────────────── _extract_artifact ─────────────────────────


def test_extract_run_python_with_figures() -> None:
    payload = json.dumps(
        {
            "figure_paths": ["/tmp/helioai_abc/fig_0.png"],
            "stdout": "shock detected",
            "exports": {},
        }
    )
    arts = _extract_artifact("run_python", payload)
    assert len(arts) == 1
    assert arts[0]["kind"] == "image"
    assert arts[0]["figure_paths"] == ["/tmp/helioai_abc/fig_0.png"]
    assert arts[0]["stdout"] == "shock detected"


def test_extract_run_python_no_figures_returns_empty() -> None:
    payload = json.dumps({"figure_paths": [], "stdout": "ok", "exports": {}})
    assert _extract_artifact("run_python", payload) == []


def test_extract_run_python_with_param_card() -> None:
    payload = json.dumps(
        {
            "figure_paths": ["/tmp/fig.png"],
            "stdout": "",
            "exports": {},
            "cards": [{"kind": "parameter_card", "param_id": "cda/AC_H0_MFI/BGSEc", "units": "nT"}],
        }
    )
    arts = _extract_artifact("run_python", payload)
    assert len(arts) == 2
    kinds = {a["kind"] for a in arts}
    assert kinds == {"image", "parameter_card"}
    card = next(a for a in arts if a["kind"] == "parameter_card")
    assert card["param_id"] == "cda/AC_H0_MFI/BGSEc"


def test_extract_get_timeseries_preview() -> None:
    payload = json.dumps(
        {
            "param_id": "cda/AC_H0_SWE/Np",
            "name": "Np",
            "units": "#/cc",
            "cadence": "64 s",
            "mission": "cda",
            "instrument": "Solar Wind Electron Proton Alpha Monitor",
            "components": [],
            "n_points": 1238,
            "shape": [1238, 1],
            "start": "2005-01-17T12:00:00",
            "stop": "2005-01-17T14:00:00",
            "preview": "2005-01-17T12:00:22  13.9\n2005-01-17T12:01:26  11.09",
        }
    )
    arts = _extract_artifact("get_timeseries", payload)
    assert len(arts) == 1
    art = arts[0]
    assert art["kind"] == "parameter_card"
    assert art["param_id"] == "cda/AC_H0_SWE/Np"
    assert art["n_points"] == 1238
    assert art["units"] == "#/cc"
    assert art["cadence"] == "64 s"
    assert art["mission"] == "cda"


def test_extract_error_returns_empty() -> None:
    payload = json.dumps({"error": "no data", "param_id": "amda/foo"})
    assert _extract_artifact("get_timeseries", payload) == []
    assert _extract_artifact("run_python", payload) == []


def test_extract_invalid_json_returns_empty() -> None:
    assert _extract_artifact("run_python", "not json at all") == []


def test_extract_non_dict_returns_empty() -> None:
    assert _extract_artifact("search_parameters", json.dumps([{"id": "x"}])) == []


def test_extract_unknown_tool_returns_empty() -> None:
    payload = json.dumps({"result": "ok"})
    assert _extract_artifact("list_missions", payload) == []
