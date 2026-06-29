"""Unit tests for agent_loop helpers: _summarize_tool_result, _extract_artifact."""

from __future__ import annotations

import json

import pytest

from helioai.core.agent_loop import _extract_artifact, _history_tool_result, _summarize_tool_result


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
    assert "quality" not in art  # no quality block in payload


def test_extract_get_timeseries_notable_quality_folded_into_card() -> None:
    payload = json.dumps(
        {
            "param_id": "cda/AC_H0_SWE/Np",
            "n_points": 10,
            "start": "2005-01-17T12:00:00",
            "stop": "2005-01-17T14:00:00",
            "preview": "row",
            "quality": {"missing_pct": 34.0, "gaps": [], "outliers_5sigma": 2, "notable": True},
        }
    )
    art = _extract_artifact("get_timeseries", payload)[0]
    assert art["quality"]["missing_pct"] == 34.0


def test_extract_get_timeseries_clean_quality_not_attached() -> None:
    payload = json.dumps(
        {
            "param_id": "cda/AC_H0_SWE/Np",
            "n_points": 10,
            "preview": "row",
            "quality": {"missing_pct": 0.0, "gaps": [], "outliers_5sigma": 0, "notable": False},
        }
    )
    art = _extract_artifact("get_timeseries", payload)[0]
    assert "quality" not in art


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


def test_extract_catalog_preview() -> None:
    payload = json.dumps(
        {
            "_kind": "catalog_preview",
            "catalog_id": "amda/c1",
            "name": "ICME list",
            "type": "catalog",
            "nb_events_total": 341,
            "columns": ["start", "stop", "shock_type"],
            "sample": [{"start": "2005-01-17", "stop": "2005-01-18", "shock_type": "FF"}],
            "survey_start": "1996-01-01",
            "survey_stop": "2022-12-31",
        }
    )
    arts = _extract_artifact("get_catalog", payload)
    assert len(arts) == 1
    art = arts[0]
    assert art["kind"] == "catalog_preview"
    assert art["catalog_id"] == "amda/c1"
    assert art["nb_events_total"] == 341
    assert len(art["sample"]) == 1


def test_extract_catalog_no_marker_returns_empty() -> None:
    payload = json.dumps({"catalog_id": "amda/c1", "nb_events_total": 10})
    assert _extract_artifact("get_catalog", payload) == []


# ──────────────────────── history stores summarized results ──────────────────


def test_summarize_tool_result_max_chars_600() -> None:
    """History uses max_chars=600 — enough to preserve key scalar fields."""
    payload = json.dumps(
        {
            "catalog_id": "amda/c1",
            "param_id": "amda/imf_gsm",
            "n_events_found": 80,
            "n_events_downloaded": 50,
            "n_events_with_data": 48,
            "units": "nT",
            "per_event_stats": [{"event": i, "mean": float(i)} for i in range(50)],
            "dataset": "imf_gsm_events",
            "cap_warning": "Showing first 50/80 events. Pass max_events=80 for full SEA.",
        }
    )
    summary = _summarize_tool_result(payload, max_chars=600)
    data = json.loads(summary)
    assert data["n_events_found"] == 80
    assert data["dataset"] == "imf_gsm_events"
    assert data["per_event_stats"] == "[50 items]"
    assert len(summary) <= 600


def test_summarize_dataset_key_preserved_short_string() -> None:
    """dataset key (short string) is always preserved in history summary."""
    payload = json.dumps(
        {
            "dataset": "ace_bgse",
            "dataset_note": "use load_data('ace_bgse') in run_python — never spz.get_data",
            "n_points": 86400,
            "units": "nT",
        }
    )
    summary = _summarize_tool_result(payload, max_chars=600)
    data = json.loads(summary)
    assert data["dataset"] == "ace_bgse"
    assert "ace_bgse" in data["dataset_note"]


# ──────────────────────── _history_tool_result routing ──────────────────────


def test_history_search_parameters_passed_verbatim() -> None:
    """search_parameters results must reach the LLM verbatim — LLM selects the param_id from this list."""
    payload = json.dumps(
        {
            "query": "solar wind proton density",
            "provider": None,
            "results": [
                {
                    "id": "cda/AC_H0_SWI/Np",
                    "description": "ACE SWICS Proton Density",
                    "score": 0.91,
                },
                {"id": "amda/imf_gsm", "description": "IMF GSM components", "score": 0.42},
            ],
        }
    )
    result = _history_tool_result("search_parameters", payload)
    data = json.loads(result)
    assert isinstance(data["results"], list), "results must be a list, not '[N items]'"
    assert data["results"][0]["id"] == "cda/AC_H0_SWI/Np"


def test_history_list_missions_passed_verbatim() -> None:
    payload = json.dumps(
        {"missions": [{"id": "ace", "name": "ACE"}, {"id": "wind", "name": "Wind"}]}
    )
    result = _history_tool_result("list_missions", payload)
    data = json.loads(result)
    assert isinstance(data["missions"], list)


def test_history_get_events_timeseries_summarized() -> None:
    """get_events_timeseries (SEA) must be summarized — per_event_stats can have 50+ entries."""
    payload = json.dumps(
        {
            "catalog_id": "amda/c1",
            "param_id": "amda/imf_gsm",
            "n_events_found": 80,
            "per_event_stats": [{"event": i, "mean": float(i)} for i in range(50)],
            "dataset": "imf_gsm_events",
        }
    )
    result = _history_tool_result("get_events_timeseries", payload)
    data = json.loads(result)
    assert data["per_event_stats"] == "[50 items]", (
        "per_event_stats must be collapsed to avoid context flood"
    )
    assert data["dataset"] == "imf_gsm_events"


def test_history_get_catalog_summarized() -> None:
    """get_catalog sample rows must be summarized."""
    payload = json.dumps(
        {
            "catalog_id": "amda/c1",
            "nb_events_total": 341,
            "sample": [{"start": "2008-01-01", "stop": "2008-01-02"} for _ in range(20)],
            "_kind": "catalog_preview",
        }
    )
    result = _history_tool_result("get_catalog", payload)
    data = json.loads(result)
    assert data["sample"] == "[20 items]"


# ──────────────────────── plan-of-flight: present_plan ───────────────────────


@pytest.mark.asyncio
async def test_inline_plan_content_emitted_before_tool_call(monkeypatch, tmp_path) -> None:
    """When the provider returns prose alongside tool_calls, it is surfaced as-is (no extra call)."""
    from helioai.core import agent_loop
    from helioai.core.llm.base import Message, ToolCall
    from helioai.core.session import SessionStore

    monkeypatch.setattr(agent_loop, "store", SessionStore(tmp_path / "sessions.db"))

    class _FakeLLM:
        def __init__(self):
            self.calls = 0

        async def chat(self, messages, tools, **k):
            self.calls += 1
            if any(m.role == "tool" for m in messages):
                return Message(role="assistant", content="done")
            return Message(
                role="assistant",
                content="Plan: list the skills, then answer.",
                tool_calls=[ToolCall(id="t1", name="list_skills", arguments={})],
            )

    llm = _FakeLLM()
    events = [ev async for ev in agent_loop.stream_chat(llm, "web", "s1", "hi", restricted=False)]
    kinds = [e["event"] for e in events]
    assert kinds.index("reply") < kinds.index("tool_call")
    assert "Plan:" in events[kinds.index("reply")]["data"]["text"]
    assert llm.calls == 2  # no dedicated planning call when prose is inline


@pytest.mark.asyncio
async def test_present_plan_emits_plan_event_and_continues(monkeypatch, tmp_path) -> None:
    """present_plan emits a structured plan event, then execution continues (non-blocking)."""
    from helioai.core import agent_loop
    from helioai.core.llm.base import Message, ToolCall
    from helioai.core.session import SessionStore

    monkeypatch.setattr(agent_loop, "store", SessionStore(tmp_path / "sessions.db"))

    class _FakeLLM:
        async def chat(self, messages, tools, **k):
            if any(m.role == "tool" for m in messages):
                return Message(role="assistant", content="done")
            return Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="p1",
                        name="present_plan",
                        arguments={
                            "title": "theta_Bn at WIND shock",
                            "steps": [
                                {"description": "resolve B and V", "tool": "search_parameters"},
                                {"description": "download the interval", "tool": "get_timeseries"},
                            ],
                        },
                    )
                ],
            )

    events = [
        ev async for ev in agent_loop.stream_chat(_FakeLLM(), "web", "s1", "hi", restricted=False)
    ]
    plan_evs = [e for e in events if e["event"] == "plan"]
    assert len(plan_evs) == 1
    assert plan_evs[0]["data"]["title"] == "theta_Bn at WIND shock"
    assert len(plan_evs[0]["data"]["steps"]) == 2
    # non-blocking: the loop runs to completion (a normal done, no pause)
    assert any(e["event"] == "done" for e in events)
    assert any(e["event"] == "reply" and e["data"]["text"] == "done" for e in events)
