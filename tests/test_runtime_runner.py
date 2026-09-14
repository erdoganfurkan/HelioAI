"""The one loop: what `Runner` does with a policy, independent of either wrapper."""

from __future__ import annotations

import pytest
from support.scripted import ScriptedLLM, ScriptedRegistry, assistant_calls, assistant_text

from helioai.core.llm.base import Message, ToolDef
from helioai.runtime.policies import Policy
from helioai.runtime.runner import RunEnd, Runner
from helioai.tools.results import ToolResult

_TOOLS = (ToolDef(name="list_recipes", description="", parameters={"type": "object"}),)


def _policy(**overrides) -> Policy:
    base = dict(name="t", system_prompt="be brief", tools=_TOOLS, max_turns=3)
    return Policy(**{**base, **overrides})


async def _drain(runner: Runner, history: list[Message]) -> tuple[list[dict], RunEnd]:
    events: list[dict] = []
    end = None
    async for item in runner.run(history):
        if isinstance(item, RunEnd):
            end = item
        else:
            events.append(item)
    assert end is not None
    return events, end


async def test_a_plain_answer_ends_the_run_with_its_text_and_usage():
    llm = ScriptedLLM([assistant_text("done")])
    history = [Message(role="user", content="q")]
    events, end = await _drain(Runner(_policy(), llm), history)
    assert events == []
    assert end.final_text == "done" and end.turns == 1 and not end.capped and not end.empty
    assert end.usage["n_calls"] == 1
    assert [m.role for m in history] == ["user", "assistant"]


async def test_a_tool_turn_emits_call_then_result_and_appends_the_tool_message():
    llm = ScriptedLLM([assistant_calls("list_recipes"), assistant_text("done")])
    reg = ScriptedRegistry({"list_recipes": {"recipes": ["theta_bn"]}})
    events, end = await _drain(
        Runner(_policy(), llm, registry=reg), [Message(role="user", content="q")]
    )
    assert [e["event"] for e in events] == ["tool_call", "tool_result"]
    assert events[0]["data"]["turn"] == 1 and events[1]["data"]["name"] == "list_recipes"
    assert reg.invoked == ["list_recipes"]
    assert end.turns == 2


async def test_the_intercept_forwards_its_events_then_the_result_then_the_trailing_ones():
    """The protocol the lead's `task` and `present_plan` rely on: what a handler yields
    before its result is shown before the tool's own events, what it yields after comes
    after them — the `sub_agent_end` a lead re-emits lands after `tool_result`."""

    async def handler(tc, turn):
        yield {"event": "skill_loaded", "data": {"name": "before"}}
        yield ToolResult.from_raw(tc.name, {"ok": True})
        yield {"event": "skill_loaded", "data": {"name": "after"}}

    llm = ScriptedLLM([assistant_calls("list_recipes"), assistant_text("done")])
    reg = ScriptedRegistry()
    runner = Runner(_policy(), llm, registry=reg, intercept=lambda tc, turn: handler(tc, turn))
    events, _ = await _drain(runner, [Message(role="user", content="q")])
    assert [(e["event"], e["data"].get("name")) for e in events] == [
        ("tool_call", "list_recipes"),
        ("skill_loaded", "before"),
        ("tool_result", "list_recipes"),
        ("skill_loaded", "after"),
    ]
    assert reg.invoked == [], "an intercepted call never reaches the registry"


async def test_an_intercept_that_declines_lets_the_registry_answer():
    llm = ScriptedLLM([assistant_calls("list_recipes"), assistant_text("done")])
    reg = ScriptedRegistry({"list_recipes": {"recipes": []}})
    runner = Runner(_policy(), llm, registry=reg, intercept=lambda tc, turn: None)
    await _drain(runner, [Message(role="user", content="q")])
    assert reg.invoked == ["list_recipes"]


async def test_an_intercept_that_raises_becomes_a_failed_result_not_a_crash():
    async def boom(tc, turn):
        raise RuntimeError("handler exploded")
        yield  # noqa: RET503 — makes this an async generator

    llm = ScriptedLLM([assistant_calls("list_recipes"), assistant_text("done")])
    runner = Runner(_policy(), llm, registry=ScriptedRegistry(), intercept=boom)
    history = [Message(role="user", content="q")]
    events, _ = await _drain(runner, history)
    tool_msg = next(m for m in history if m.role == "tool")
    assert "handler exploded" in tool_msg.content
    assert events[1]["data"]["summary"].startswith("error")


async def test_the_cap_ends_the_run_without_an_answer():
    llm = ScriptedLLM([assistant_calls("list_recipes")] * 3)
    reg = ScriptedRegistry()
    _, end = await _drain(
        Runner(_policy(max_turns=3), llm, registry=reg), [Message(role="user", content="q")]
    )
    assert end.capped and end.final_text is None and end.turns == 3
    assert reg.invoked == ["list_recipes"] * 3


async def test_an_empty_reply_stops_the_lead_and_ends_a_role_with_an_empty_summary():
    lead = Runner(_policy(stop_on_empty_reply=True), ScriptedLLM([assistant_text("   ")]))
    _, end = await _drain(lead, [Message(role="user", content="q")])
    assert end.empty and end.final_text == ""

    role = Runner(_policy(stop_on_empty_reply=False), ScriptedLLM([assistant_text("")]))
    _, end = await _drain(role, [Message(role="user", content="q")])
    assert not end.empty and end.final_text == ""


async def test_a_role_asks_for_a_tool_on_its_first_turn_and_the_lead_leaves_the_default():
    llm = ScriptedLLM([assistant_calls("list_recipes"), assistant_text("done")])
    await _drain(
        Runner(_policy(tool_choice_first="required"), llm, registry=ScriptedRegistry()),
        [Message(role="user", content="q")],
    )
    assert [c["tool_choice"] for c in llm.calls] == ["required", "auto"]

    llm = ScriptedLLM([assistant_text("done")])
    await _drain(Runner(_policy(), llm), [Message(role="user", content="q")])
    assert llm.calls[0]["tool_choice"] == "auto", (
        "the lead never sends tool_choice; the client default is auto"
    )


async def test_commentary_alongside_tool_calls_is_a_reply_only_when_the_policy_says_so():
    script = [assistant_calls("list_recipes", content="Let me look."), assistant_text("done")]
    events, _ = await _drain(
        Runner(_policy(comment_replies=True), ScriptedLLM(script), registry=ScriptedRegistry()),
        [Message(role="user", content="q")],
    )
    assert events[0] == {"event": "reply", "data": {"text": "Let me look."}}

    events, _ = await _drain(
        Runner(
            _policy(comment_replies=False), ScriptedLLM(list(script)), registry=ScriptedRegistry()
        ),
        [Message(role="user", content="q")],
    )
    assert [e["event"] for e in events] == ["tool_call", "tool_result"]


async def test_a_roles_context_rides_on_every_event_and_is_stripped_from_the_artifacts(tmp_path):
    fig = tmp_path / "fig.png"
    fig.write_bytes(b"\x89PNG")
    ctx = {"role": "data_analyst", "task_id": "t1"}
    llm = ScriptedLLM([assistant_calls("run_python"), assistant_text("done")])
    reg = ScriptedRegistry(
        {
            "run_python": {
                "stdout": "",
                "figure_paths": [str(fig)],
                "exports": {},
                "cards": [],
                "code_path": str(tmp_path / "code_0.py"),
                "n_lines": 1,
            }
        }
    )
    tools = _TOOLS + (ToolDef(name="run_python", description="", parameters={"type": "object"}),)
    runner = Runner(_policy(tools=tools, sub_agent_ctx=ctx), llm, registry=reg)
    events, end = await _drain(runner, [Message(role="user", content="q")])
    assert all(e["data"]["sub_agent_ctx"] == ctx for e in events)
    assert end.artifacts and all("sub_agent_ctx" not in a for a in end.artifacts)


async def test_progress_is_readable_after_the_model_raised():
    """A role that blows up on its second call still reports the turns it made and what
    they cost — the wrapper reads them off the runner when there is no `RunEnd`."""

    class Boom(ScriptedLLM):
        async def chat(self, *a, **kw):
            if len(self.calls) == 1:
                raise RuntimeError("provider down")
            return await super().chat(*a, **kw)

    llm = Boom([assistant_calls("list_recipes")])
    runner = Runner(_policy(), llm, registry=ScriptedRegistry())
    with pytest.raises(RuntimeError):
        await _drain(runner, [Message(role="user", content="q")])
    assert runner.turns == 2 and runner.usage["n_calls"] == 1


async def test_an_answer_quoting_an_unknown_id_buys_one_corrected_turn(monkeypatch):
    from helioai.runtime import runner as runtime_runner

    monkeypatch.setattr(
        "helioai.tools.rag.unknown_ids", lambda ids: [i for i in ids if "BOGUS" in i]
    )
    monkeypatch.setattr(
        "helioai.tools.rag.extract_ids", lambda text: ["cda/BOGUS/x"] if "BOGUS" in text else []
    )
    llm = ScriptedLLM([assistant_text("use cda/BOGUS/x"), assistant_text("use cda/REAL/x")])
    history = [Message(role="user", content="q")]
    _, end = await _drain(runtime_runner.Runner(_policy(), llm), history)
    assert end.final_text == "use cda/REAL/x" and end.turns == 2
    correction = history[2]
    assert correction.role == "user" and correction.origin == "correction"
