"""Tests for the human-facing event descriptions shared by all three interfaces.

The bar these have to clear: a person reading a live session should be able to say what
just happened without decoding JSON. So the assertions are about what a reader sees —
the parameter id, the cadence, how many results — not about an exact string, which would
freeze the wording.
"""

from __future__ import annotations

import json

from helioai.core.event_display import describe_tool_call, describe_tool_result

# ── tool calls ─────────────────────────────────────────────────────────────────


def test_run_python_reports_size_not_a_truncated_snippet():
    """Showing 40 characters of source cut a comment in half and left a stray paren."""
    code = "\n".join(f"x{i} = {i}" for i in range(108))
    out = describe_tool_call("run_python", {"code": code})
    assert "108" in out
    assert "x0 = 0" not in out


def test_search_parameters_shows_the_query():
    out = describe_tool_call("search_parameters", {"query": "wind B GSM", "provider": "cda"})
    assert "wind B GSM" in out
    assert "cda" in out


def test_search_parameters_batch_reports_the_count():
    out = describe_tool_call("search_parameters", {"queries": ["a", "b", "c"]})
    assert "3" in out


def test_get_timeseries_shows_the_parameter_and_window():
    out = describe_tool_call(
        "get_timeseries",
        {
            "param_id": "cda/WI_H0_MFI/B3GSE",
            "start": "2015-03-17T03:30",
            "stop": "2015-03-17T05:00",
        },
    )
    assert "cda/WI_H0_MFI/B3GSE" in out
    assert "2015-03-17" in out


def test_task_shows_the_role():
    out = describe_tool_call(
        "task", {"agent_role": "data_analyst", "description": "download and plot B"}
    )
    assert "data_analyst" in out


def test_unknown_tool_still_produces_something_readable():
    out = describe_tool_call("some_new_tool", {"alpha": 3, "beta": "x"})
    assert out
    assert "alpha" in out


def test_no_arguments_does_not_crash():
    assert isinstance(describe_tool_call("list_missions", None), str)


# ── tool results ───────────────────────────────────────────────────────────────


def test_get_timeseries_result_reads_like_a_sentence_not_json():
    result = json.dumps(
        {
            "param_id": "cda/WI_H0_MFI/B3GSE",
            "name": "B3GSE",
            "cadence": "3 s",
            "mission": "WI",
            "n_points": 1800,
            "units": "nT (3sec)",
            "dataset_note": "use load_data('b3gse') in run_python — never spz.get_data",
        }
    )
    out = describe_tool_result("get_timeseries", result)
    assert "1800" in out
    assert "3 s" in out
    # The note is written at the model, and is the longest string in the payload.
    assert "load_data" not in out
    assert "{" not in out


def test_search_parameters_result_reports_hits():
    result = json.dumps({"query": "wind B", "results": [{"id": "cda/A/B"}, {"id": "cda/C/D"}]})
    out = describe_tool_result("search_parameters", result)
    assert "2" in out
    assert "cda/A/B" in out


def test_run_python_result_mentions_figures():
    result = json.dumps(
        {"stdout": "ok", "figure_paths": ["/tmp/x/fig_1_0.png"], "n_figures": 1, "n_lines": 108}
    )
    out = describe_tool_result("run_python", result)
    assert "1" in out
    assert "/tmp/x" not in out


def test_an_error_result_leads_with_the_error():
    result = json.dumps({"error": "AttributeError: no attribute 'unit'", "stderr": "Traceback..."})
    out = describe_tool_result("run_python", result)
    assert out.lower().startswith("error") or "AttributeError" in out


def test_a_long_error_is_truncated_but_still_names_the_exception():
    result = json.dumps({"error": "ValueError: " + "x" * 2000})
    out = describe_tool_result("run_python", result)
    assert "ValueError" in out
    assert len(out) < 300


def test_non_json_result_is_passed_through_shortened():
    out = describe_tool_result("whatever", "plain text result")
    assert "plain text" in out


def test_unknown_tool_result_avoids_dumping_raw_json():
    result = json.dumps({"alpha": 1, "beta": "two", "gamma": {"nested": "dict"}})
    out = describe_tool_result("some_new_tool", result)
    assert "alpha" in out
    assert '{"alpha"' not in out


def test_empty_result_does_not_crash():
    assert isinstance(describe_tool_result("x", ""), str)
