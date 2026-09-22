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
    payload = json.dumps({"per_provider": {"a": 1, "b": 2, "c": 3}})
    summary = _summarize_tool_result(payload)
    data = json.loads(summary)
    assert data["per_provider"] == "{3 keys}"


def test_summarize_scalar_fields_kept() -> None:
    payload = json.dumps({"n_points": 1238, "units": "nT", "ok": True})
    summary = _summarize_tool_result(payload)
    data = json.loads(summary)
    assert data["n_points"] == 1238
    assert data["units"] == "nT"
    assert data["ok"] is True


# ──────────────────────────────── _extract_artifact ─────────────────────────


def test_extract_run_python_with_figures() -> None:
    payload = {
        "figure_paths": ["/tmp/helioai_abc/fig_0.png"],
        "stdout": "shock detected",
        "exports": {},
    }
    arts = _extract_artifact("run_python", payload)
    assert len(arts) == 1
    assert arts[0]["kind"] == "image"
    assert arts[0]["figure_paths"] == ["/tmp/helioai_abc/fig_0.png"]
    assert arts[0]["stdout"] == "shock detected"


def test_extract_run_python_no_figures_returns_empty() -> None:
    payload = {"figure_paths": [], "stdout": "ok", "exports": {}}
    assert _extract_artifact("run_python", payload) == []


def test_extract_run_python_with_param_card() -> None:
    payload = {
        "figure_paths": ["/tmp/fig.png"],
        "stdout": "",
        "exports": {},
        "cards": [{"kind": "parameter_card", "param_id": "cda/AC_H0_MFI/BGSEc", "units": "nT"}],
    }
    arts = _extract_artifact("run_python", payload)
    assert len(arts) == 2
    kinds = {a["kind"] for a in arts}
    assert kinds == {"image", "parameter_card"}
    card = next(a for a in arts if a["kind"] == "parameter_card")
    assert card["param_id"] == "cda/AC_H0_MFI/BGSEc"


def test_extract_get_timeseries_preview() -> None:
    payload = {
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
    assert "obtained_start" not in art and "coverage_note" not in art


def test_extract_get_timeseries_card_says_what_came_back_when_it_differs() -> None:
    """A series clipped to the archive's coverage or to a gap is the one fact the reader
    most needs on the card; it used to show only the window that was asked for."""
    payload = {
        "param_id": "cda/WI_H0_MFI/B3GSE",
        "n_points": 600,
        "start": "2004-11-07T17:00:00",
        "stop": "2004-11-07T19:00:00",
        "obtained_start": "2004-11-07T17:30:01",
        "obtained_stop": "2004-11-07T17:59:58",
        "coverage_note": "Requested [..] extends past the coverage — the series is clipped.",
        "preview": "row",
    }
    (art,) = _extract_artifact("get_timeseries", payload)
    assert art["obtained_start"] == "2004-11-07T17:30:01"
    assert art["obtained_stop"] == "2004-11-07T17:59:58"
    assert art["coverage_note"].startswith("Requested")


def test_extract_get_timeseries_notable_quality_folded_into_card() -> None:
    payload = {
        "param_id": "cda/AC_H0_SWE/Np",
        "n_points": 10,
        "start": "2005-01-17T12:00:00",
        "stop": "2005-01-17T14:00:00",
        "preview": "row",
        "quality": {"missing_pct": 34.0, "gaps": [], "outliers_5sigma": 2, "notable": True},
    }
    art = _extract_artifact("get_timeseries", payload)[0]
    assert art["quality"]["missing_pct"] == 34.0


def test_extract_get_timeseries_clean_quality_not_attached() -> None:
    payload = {
        "param_id": "cda/AC_H0_SWE/Np",
        "n_points": 10,
        "preview": "row",
        "quality": {"missing_pct": 0.0, "gaps": [], "outliers_5sigma": 0, "notable": False},
    }
    art = _extract_artifact("get_timeseries", payload)[0]
    assert "quality" not in art


def test_extract_error_returns_empty() -> None:
    payload = {"error": "no data", "param_id": "amda/foo"}
    assert _extract_artifact("get_timeseries", payload) == []
    assert _extract_artifact("run_python", payload) == []


def test_extract_invalid_json_returns_empty() -> None:
    assert _extract_artifact("run_python", "not json at all") == []


def test_extract_non_dict_returns_empty() -> None:
    assert _extract_artifact("search_parameters", [{"id": "x"}]) == []


def test_extract_unknown_tool_returns_empty() -> None:
    payload = {"result": "ok"}
    assert _extract_artifact("list_missions", payload) == []


def test_extract_catalog_preview() -> None:
    payload = {
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
    arts = _extract_artifact("get_catalog", payload)
    assert len(arts) == 1
    art = arts[0]
    assert art["kind"] == "catalog_preview"
    assert art["catalog_id"] == "amda/c1"
    assert art["nb_events_total"] == 341
    assert len(art["sample"]) == 1


def test_extract_catalog_no_marker_returns_empty() -> None:
    payload = {"catalog_id": "amda/c1", "nb_events_total": 10}
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


@pytest.mark.asyncio
async def test_crash_mid_loop_still_persists_history(monkeypatch, tmp_path) -> None:
    """A bug/outage anywhere in the loop must not silently drop the user's message.

    Real bug: only `asyncio.CancelledError` was saved on exit — any other exception
    (a flaky LLM call, a bug in a post-tool hook) skipped every store.save(), leaving
    a session with the user prompt and nothing else once persisted.
    """
    from helioai.core import agent_loop
    from helioai.core.llm.base import Message
    from helioai.core.session import SessionStore

    db_path = tmp_path / "sessions.db"
    store = SessionStore(db_path)
    monkeypatch.setattr(agent_loop, "store", store)

    # Simulate an existing session (a completed first turn) — the "new session"
    # early-save at the top of stream_chat only fires once, on turn 1, so it can't
    # mask a missing except-Exception save on a later turn.
    store.save(
        "web",
        "s1",
        [Message(role="user", content="first"), Message(role="assistant", content="ok")],
    )
    store.set_workspace_dir("web", "s1", "some-label")

    class _FakeLLM:
        async def chat(self, messages, tools, **k):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        async for _ in agent_loop.stream_chat(_FakeLLM(), "web", "s1", "second", restricted=False):
            pass

    # fresh store, no in-memory cache — proves the crash path hit the DB, not just RAM
    reloaded = SessionStore(db_path).get_or_create("web", "s1")
    assert [m.role for m in reloaded] == ["user", "assistant", "user"]
    assert reloaded[-1].content == "second"


async def test_empty_llm_turn_is_reported_not_swallowed(monkeypatch, tmp_path):
    """A turn with no text and no tool call is a failure, not an answer.

    Regression: it was yielded as an empty `reply`, so the caller saw the request
    produce nothing at all — silence indistinguishable from success. Act VI of
    examples/02 hit this on Azure, whose reasoning tokens come out of the same
    output allowance.
    """
    from helioai.core.agent_loop import stream_chat
    from helioai.core.llm.base import Message

    class _EmptyClient:
        async def chat(self, messages, tools, system_prompt=""):
            return Message(role="assistant", content="   ", tool_calls=None)

        async def aclose(self):
            return None

    events = [
        e
        async for e in stream_chat(
            _EmptyClient(), "web", "s_empty", "write me a long script", restricted=False
        )
    ]

    kinds = [e["event"] for e in events]
    assert "error" in kinds, f"an empty turn must surface an error, got {kinds}"
    assert "reply" not in kinds, "an empty reply must not be emitted"

    message = next(e["data"]["message"] for e in events if e["event"] == "error")
    assert "HELIOAI_MAX_OUTPUT_TOKENS" in message, "the error must name the knob to raise"
    assert kinds[-1] == "done"


@pytest.mark.asyncio
async def test_two_concurrent_turns_on_one_session_do_not_interleave(monkeypatch, tmp_path):
    """Two turns for the same session must run one after the other.

    `store.get_or_create` hands both of them the same in-memory list; without a lock
    their appends interleave (`user, user, assistant, assistant`) and the whole-list
    `save` persists whichever ordering won — two browser tabs on one session were
    enough to corrupt a transcript.
    """
    import asyncio

    from helioai.core import agent_loop
    from helioai.core.llm.base import Message
    from helioai.core.session import SessionStore

    db_path = tmp_path / "sessions.db"
    monkeypatch.setattr(agent_loop, "store", SessionStore(db_path))

    class _SlowLLM:
        async def chat(self, messages, tools, **k):
            await asyncio.sleep(0.02)
            asked = next(m.content for m in reversed(messages) if m.role == "user")
            return Message(role="assistant", content=f"answer to {asked}")

    async def turn(text: str) -> None:
        async for _ in agent_loop.stream_chat(_SlowLLM(), "web", "s1", text, restricted=False):
            pass

    await asyncio.gather(turn("first"), turn("second"))

    reloaded = SessionStore(db_path).get_or_create("web", "s1")
    assert [m.role for m in reloaded] == ["user", "assistant", "user", "assistant"]
    for question, answer in zip(reloaded[::2], reloaded[1::2], strict=True):
        assert answer.content == f"answer to {question.content}"


@pytest.mark.asyncio
async def test_sub_agent_end_reemitted_by_the_lead_keeps_the_findings(monkeypatch, tmp_path):
    """`findings` is the table of values a sub-agent actually measured — the one part
    of its report with an origin. The lead consumed the sub-agent's `sub_agent_end` and
    re-emitted a stripped copy (summary, n_iterations, error), so no interface ever saw
    the numbers; readers got them only when the model chose to quote them."""
    from helioai.core import agent_loop
    from helioai.core.llm.base import Message, ToolCall
    from helioai.core.session import SessionStore

    monkeypatch.setattr(agent_loop, "store", SessionStore(tmp_path / "sessions.db"))

    findings = {"theta_bn_deg": {"value": 47.3, "units": "deg", "code_path": "code_0.py"}}

    async def fake_subagent(**kwargs):
        yield {
            "event": "sub_agent_end",
            "data": {
                "task_id": kwargs["task_id"],
                "role": kwargs["role"],
                "findings": findings,
                "summary": "theta_Bn by coplanarity",
                "n_iterations": 2,
                "error": None,
                "artifacts": [],
            },
        }

    monkeypatch.setattr(agent_loop, "stream_subagent", fake_subagent)

    responses = [
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="task",
                    arguments={"agent_role": "data_analyst", "description": "compute theta_Bn"},
                )
            ],
        ),
        Message(role="assistant", content="theta_Bn is 47.3 deg."),
    ]

    class _FakeLLM:
        async def chat(self, messages, tools, **k):
            return responses.pop(0)

    events = [
        ev
        async for ev in agent_loop.stream_chat(_FakeLLM(), "web", "s1", "theta?", restricted=False)
    ]
    ends = [e["data"] for e in events if e["event"] == "sub_agent_end"]
    assert len(ends) == 1
    assert ends[0]["findings"] == findings
    assert ends[0]["summary"] == "theta_Bn by coplanarity"


@pytest.mark.asyncio
async def test_lead_and_sub_agent_usage_is_charged_to_the_session(monkeypatch, tmp_path):
    """The providers' token counts rode on Message and were dropped at save. Every lead
    call is now a usage row, and a sub-agent's calls are reported on sub_agent_end and
    charged to the parent session under the role."""
    from helioai.core import agent_loop
    from helioai.core.llm.base import Message, ToolCall
    from helioai.core.session import SessionStore

    store = SessionStore(tmp_path / "sessions.db")
    monkeypatch.setattr(agent_loop, "store", store)

    async def fake_subagent(**kwargs):
        yield {
            "event": "sub_agent_end",
            "data": {
                "task_id": kwargs["task_id"],
                "role": kwargs["role"],
                "findings": {},
                "summary": "done",
                "n_iterations": 2,
                "error": None,
                "artifacts": [],
                "usage": {
                    "prompt_tokens": 3000,
                    "completion_tokens": 400,
                    "cached_tokens": 0,
                    "n_calls": 2,
                },
            },
        }

    monkeypatch.setattr(agent_loop, "stream_subagent", fake_subagent)
    responses = [
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="task",
                    arguments={"agent_role": "data_analyst", "description": "x"},
                )
            ],
            prompt_tokens=1000,
            completion_tokens=50,
        ),
        Message(
            role="assistant",
            content="answer",
            prompt_tokens=1200,
            completion_tokens=80,
            cached_tokens=900,
        ),
    ]

    class _LLM:
        async def chat(self, messages, tools, **k):
            return responses.pop(0)

    async for _ in agent_loop.stream_chat(_LLM(), "web", "s1", "go", restricted=False):
        pass

    totals = store.usage_totals("web", "s1")
    assert totals == {
        "prompt_tokens": 5200,
        "completion_tokens": 530,
        "cached_tokens": 900,
        "n_calls": 3,
    }


@pytest.mark.asyncio
async def test_two_sessions_whose_ids_share_six_characters_get_two_workspaces(
    monkeypatch, tmp_path
) -> None:
    """The bench found it: `bench-wind-…` ids all mapped to one directory, and from the
    second run on the inventory handed the sub-agent the first run's data. The loop now
    passes the user's existing labels to `make_session_label`."""
    from helioai.core import agent_loop
    from helioai.core.llm.base import Message
    from helioai.core.session import SessionStore

    store = SessionStore(tmp_path / "sessions.db")
    monkeypatch.setattr(agent_loop, "store", store)

    class _Say:
        async def chat(self, messages, tools, **k):
            return Message(role="assistant", content="ok")

    for sid in ("session-001", "session-002"):
        async for _ in agent_loop.stream_chat(_Say(), "web", sid, "Find a shock", restricted=False):
            pass
    a = store.get_workspace_dir("web", "session-001")
    b = store.get_workspace_dir("web", "session-002")
    assert a and b and a != b


# ──────────────────────── a capped delegation, as the lead reads it ──────────


def _capped_sub_agent(findings: dict):
    """A `stream_subagent` double whose run ended on its turn cap.

    It yields exactly what the real one yields at a cap (`sub_agents.py`): the cap
    message in `error`, `capped` true, and whatever it measured on the way there.
    """

    async def fake(**kwargs):
        yield {
            "event": "sub_agent_end",
            "data": {
                "task_id": kwargs["task_id"],
                "role": kwargs["role"],
                "findings": findings,
                "summary": "",
                "n_iterations": 12,
                "error": "(sub-agent 'data_analyst' reached its 12-turn cap)",
                "artifacts": [],
                "capped": True,
                "usage": {},
            },
        }

    return fake


async def _task_result_the_lead_reads(monkeypatch, tmp_path, findings: dict) -> dict:
    """Run one delegating turn and return the `task` payload the model was handed.

    The assertion target is the tool message in the history of the lead's *second*
    call — what the model actually reads — not the `sub_agent_end` event, which is
    what the three renderers read and which `4c0c150` already fixed.
    """
    from helioai.core import agent_loop
    from helioai.core.llm.base import Message, ToolCall
    from helioai.core.session import SessionStore

    monkeypatch.setattr(agent_loop, "store", SessionStore(tmp_path / "sessions.db"))
    monkeypatch.setattr(agent_loop, "stream_subagent", _capped_sub_agent(findings))

    responses = [
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="task",
                    arguments={"agent_role": "data_analyst", "description": "compute theta_Bn"},
                )
            ],
        ),
        Message(role="assistant", content="done."),
    ]
    seen: list[list] = []

    class _FakeLLM:
        async def chat(self, messages, tools, **k):
            seen.append(list(messages))
            return responses.pop(0)

    async for _ in agent_loop.stream_chat(_FakeLLM(), "web", "s1", "theta?", restricted=False):
        pass

    tool_msgs = [
        m
        for m in seen[-1]
        if getattr(m, "role", None) == "tool" and getattr(m, "name", None) == "task"
    ]
    assert tool_msgs, "the lead's last call must carry the task result"
    return json.loads(tool_msgs[0].content)


@pytest.mark.asyncio
async def test_a_capped_run_with_findings_reaches_the_lead_as_capped_not_failed(
    monkeypatch, tmp_path
) -> None:
    """`4c0c150` taught the three renderers that a role which ran out of turns with
    values on the table finished at its cap; it did not teach the lead. The `task`
    result the model reads carries no `capped` key at all, so eleven measured values
    arrive next to `error: "(sub-agent … reached its 12-turn cap)"` and the only
    reading available to the model is that the delegation failed."""
    findings = {"theta_bn_deg": {"value": 47.3, "units": "deg", "code_path": "code_0.py"}}
    payload = await _task_result_the_lead_reads(monkeypatch, tmp_path, findings)

    assert payload["capped"] is True, "the model must be able to tell a cap from a crash"
    assert payload["error"] is None, "a run that measured something did not fail"
    assert payload["findings"] == findings


@pytest.mark.asyncio
async def test_a_capped_run_that_measured_nothing_still_reaches_the_lead_as_an_error(
    monkeypatch, tmp_path
) -> None:
    """The other half, and the reason this is not a blanket `error = None`: a role that
    spent its whole budget and exported nothing has nothing for the lead to report, and
    `describe_sub_agent_end` gives that case the red cross it deserves."""
    payload = await _task_result_the_lead_reads(monkeypatch, tmp_path, {})

    assert payload["capped"] is True
    assert payload["error"], "a cap with nothing measured is a failure"
