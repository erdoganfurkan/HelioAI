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


# ── progressive disclosure ───────────────────────────────────────────────────────


def _tiered_policy(**overrides) -> Policy:
    tools = tuple(
        ToolDef(name=n, description=d, parameters={"type": "object"})
        for n, d in (
            ("search_parameters", "find speasy parameter ids"),
            ("run_python", "run code"),
            ("plasma_beta", "ratio of thermal to magnetic pressure"),
            ("gyrofrequency", "particle gyrofrequency in a field"),
            ("get_catalog", "inspect an event catalog"),
        )
    )
    return _policy(
        tools=tools,
        deferred=frozenset({"plasma_beta", "gyrofrequency", "get_catalog"}),
        **overrides,
    )


async def test_deferred_tools_are_withheld_until_search_tools_reveals_the_matching_ones():
    """Twenty-one definitions rode on every call of a lead turn; the ten formulary and
    catalogue ones are used in a minority of sessions. Withheld, they cost nothing until
    the model asks — and then only the ones that match what it asked for appear."""
    llm = ScriptedLLM(
        [
            assistant_calls("search_tools", arguments={"query": "plasma beta"}),
            assistant_calls("plasma_beta", arguments={"B_nT": 5}),
            assistant_text("done"),
        ]
    )
    reg = ScriptedRegistry({"plasma_beta": {"beta": 0.4}})
    history = [Message(role="user", content="q")]
    events, _ = await _drain(Runner(_tiered_policy(), llm, registry=reg), history)

    first, second, third = (c["tools"] for c in llm.calls)
    assert "plasma_beta" not in first and "get_catalog" not in first
    assert "search_tools" in first and "search_parameters" in first
    assert "plasma_beta" in second and "gyrofrequency" not in second and "get_catalog" not in second
    assert "search_tools" in second, "other tools are still withheld, so the finder stays"
    assert third == second
    finder_reply = next(m for m in history if m.role == "tool" and m.name == "search_tools")
    assert "plasma_beta" in finder_reply.content and "gyrofrequency" not in finder_reply.content
    assert [e["data"]["name"] for e in events if e["event"] == "tool_result"] == [
        "search_tools",
        "plasma_beta",
    ]
    assert reg.invoked == ["plasma_beta"]


async def test_a_query_matching_nothing_reveals_every_deferred_tool():
    llm = ScriptedLLM(
        [assistant_calls("search_tools", arguments={"query": "xyzzy"}), assistant_text("done")]
    )
    await _drain(
        Runner(_tiered_policy(), llm, registry=ScriptedRegistry()),
        [Message(role="user", content="q")],
    )
    assert {"plasma_beta", "gyrofrequency", "get_catalog"} <= set(llm.calls[1]["tools"])
    assert "search_tools" not in llm.calls[1]["tools"], "nothing is withheld any more"


async def test_calling_a_deferred_tool_by_name_reveals_it_and_runs_it():
    """The name was right, so the model already knows the tool; refusing would only cost
    a turn (the lead's prompt still names these tools)."""
    llm = ScriptedLLM(
        [assistant_calls("gyrofrequency", arguments={"B_nT": 10}), assistant_text("done")]
    )
    reg = ScriptedRegistry({"gyrofrequency": {"frequency_Hz": 0.15}})
    events, _ = await _drain(
        Runner(_tiered_policy(), llm, registry=reg), [Message(role="user", content="q")]
    )
    assert reg.invoked == ["gyrofrequency"]
    assert [e["event"] for e in events] == ["tool_call", "tool_result"]
    assert "gyrofrequency" in llm.calls[1]["tools"] and "plasma_beta" not in llm.calls[1]["tools"]


async def test_a_policy_with_nothing_deferred_shows_every_tool_and_no_finder():
    llm = ScriptedLLM([assistant_text("done")])
    await _drain(Runner(_policy(), llm), [Message(role="user", content="q")])
    assert llm.calls[0]["tools"] == ["list_recipes"]


def test_the_lead_defers_ten_tools_and_the_gain_is_what_was_measured():
    """Pins the measurement the deferral rests on: 21 definitions per call before,
    11 plus the finder after. A tool added to the registry shows up here on purpose."""
    import json

    import helioai.tools.setup  # noqa: F401
    from helioai.core.agent_loop import _DEFERRED_TOOLS, _INTERNAL_TOOLS
    from helioai.core.sub_agents import task_tool_def
    from helioai.tools.registry import registry

    defs = tuple(registry.list_tool_defs() + _INTERNAL_TOOLS + [task_tool_def()])
    assert _DEFERRED_TOOLS <= {t.name for t in defs}
    runner = Runner(_policy(tools=defs, deferred=_DEFERRED_TOOLS), ScriptedLLM([]))
    shown = runner.visible_tools()

    def size(ts):
        return sum(
            len(json.dumps({"n": t.name, "d": t.description, "p": t.parameters})) for t in ts
        )

    assert len(defs) == 21 and len(shown) == 12
    assert size(shown) < 0.62 * size(defs), "the finder must not eat the saving"
