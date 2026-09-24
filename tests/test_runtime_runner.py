"""The one loop: what `Runner` does with a policy, independent of either wrapper."""

from __future__ import annotations

import pytest
from support.scripted import ScriptedLLM, ScriptedRegistry, assistant_calls, assistant_text

from helioai.core.llm.base import Message, ToolCall, ToolDef
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
    events, end = await _drain(runtime_runner.Runner(_policy(), llm), history)
    assert end.final_text == "use cda/REAL/x" and end.turns == 2
    correction = history[2]
    assert correction.role == "user" and correction.origin == "correction"
    # The note is also an event, so a replay from the journal shows what the model was
    # told — the persisted message alone was the one thing the journal lost.
    noted = [e for e in events if e["event"] == "correction"]
    assert len(noted) == 1
    assert noted[0]["data"] == {"ids": ["cda/BOGUS/x"], "text": correction.content}


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
    11 plus the finder after — 22 and 12 since `run_recipe`, which is shown, not deferred,
    and so costs the saving a point. A tool added to the registry shows up here on
    purpose."""
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

    assert len(defs) == 22 and len(shown) == 13
    assert size(shown) < 0.63 * size(defs), "the finder must not eat the saving"


# ── streaming ─────────────────────────────────────────────────────────────────────


class _StreamingLLM(ScriptedLLM):
    """A scripted model whose text replies arrive in three slices."""

    async def stream_chat(self, messages, tools, system_prompt=None, tool_choice="auto"):
        reply = await self.chat(
            messages, tools, system_prompt=system_prompt, tool_choice=tool_choice
        )
        if reply.content and not reply.tool_calls:
            third = max(1, len(reply.content) // 3)
            for i in range(0, len(reply.content), third):
                yield reply.content[i : i + third]
        yield reply


async def test_a_streaming_policy_yields_deltas_that_add_up_to_the_reply():
    llm = _StreamingLLM([assistant_text("θ_Bn = 57.5 deg, quasi-perpendicular.")])
    events, end = await _drain(
        Runner(_policy(stream_replies=True), llm), [Message(role="user", content="q")]
    )
    deltas = [e for e in events if e["event"] == "reply_delta"]
    assert len(deltas) >= 3
    assert "".join(e["data"]["text"] for e in deltas) == end.final_text
    assert [e["event"] for e in events] == ["reply_delta"] * len(deltas)


async def test_a_policy_that_does_not_stream_asks_for_the_reply_whole():
    llm = _StreamingLLM([assistant_text("whole")])
    events, end = await _drain(
        Runner(_policy(stream_replies=False), llm), [Message(role="user", content="q")]
    )
    assert events == [] and end.final_text == "whole"


async def test_the_lead_streams_its_answer_and_the_journal_keeps_only_the_reply(
    monkeypatch, tmp_path
):
    """Deltas are transient: the journal keeps the `reply` that follows and nothing else,
    so a replay shows the answer once."""
    from helioai.core import agent_loop
    from helioai.core.session import SessionStore

    test_store = SessionStore(tmp_path / "sessions.db")
    monkeypatch.setattr(agent_loop, "store", test_store)
    llm = _StreamingLLM([assistant_text("streamed answer")])

    live = [ev async for ev in agent_loop.stream_chat(llm, "web", "s-stream", "q")]
    kinds = [e["event"] for e in live]
    assert kinds.count("reply_delta") >= 3 and kinds.count("reply") == 1
    journaled = [e["event"] for e in test_store.events("web", "s-stream")]
    assert "reply_delta" not in journaled and journaled.count("reply") == 1


# ── final_answer ──────────────────────────────────────────────────────────────────


def _final_answer(answer: str, claims: list | None = None) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                id="call_fa", name="final_answer", arguments={"answer": answer, "claims": claims}
            )
        ],
    )


async def test_final_answer_closes_the_run_with_its_text_and_claims_and_a_plain_history():
    """The answer arrives as a tool call; the run ends with it, the claims travel on the
    RunEnd, and the history keeps a plain assistant message — the export, the replay and
    the next turn read assistant text, and the wire needs no reply to a vanished call."""
    claims = [
        {"name": "theta_bn", "value": 57.5, "units": "deg", "source": "theta_bn"},
        {"name": "r_B", "value": 2.5, "source": "compression_ratio"},
        {"name": "Dst", "value": -223, "units": "nT", "source": "literature"},
    ]
    llm = ScriptedLLM([_final_answer("θ_Bn = 57.5°, r_B = 2.5, Dst −223 nT.", claims)])
    history = [Message(role="user", content="q")]
    events, end = await _drain(Runner(_policy(final_answer=True), llm), history)

    assert events == []
    assert end.final_text == "θ_Bn = 57.5°, r_B = 2.5, Dst −223 nT."
    assert end.claims == [
        {"name": "theta_bn", "value": 57.5, "units": "deg", "source": "theta_bn"},
        {"name": "r_B", "value": 2.5, "units": "", "source": "compression_ratio"},
        {"name": "Dst", "value": -223, "units": "nT", "source": "literature"},
    ]
    assert history[-1].role == "assistant" and history[-1].tool_calls is None
    assert history[-1].content == end.final_text
    assert "final_answer" in llm.calls[0]["tools"]


async def test_malformed_claims_are_dropped_without_losing_the_answer():
    llm = ScriptedLLM(
        [_final_answer("ok", [{"value": 1}, "junk", {"name": "n", "value": 2, "units": None}])]
    )
    _, end = await _drain(
        Runner(_policy(final_answer=True), llm), [Message(role="user", content="q")]
    )
    assert end.final_text == "ok"
    assert end.claims == [{"name": "n", "value": 2, "units": "", "source": "asserted"}]


async def test_final_answer_bundled_with_other_calls_is_refused_and_the_run_goes_on():
    bundled = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(id="c0", name="list_recipes", arguments={}),
            ToolCall(id="c1", name="final_answer", arguments={"answer": "too early"}),
        ],
    )
    llm = ScriptedLLM([bundled, _final_answer("now")])
    reg = ScriptedRegistry({"list_recipes": {"recipes": []}})
    history = [Message(role="user", content="q")]
    events, end = await _drain(Runner(_policy(final_answer=True), llm, registry=reg), history)
    refusal = next(m for m in history if m.role == "tool" and m.name == "final_answer")
    assert "alone" in refusal.content
    assert end.final_text == "now" and end.turns == 2
    assert [e["data"]["name"] for e in events if e["event"] == "tool_result"] == [
        "list_recipes",
        "final_answer",
    ]


async def test_a_policy_without_final_answer_neither_offers_nor_honours_it():
    llm = ScriptedLLM([assistant_text("plain prose")])
    _, end = await _drain(Runner(_policy(), llm), [Message(role="user", content="q")])
    assert "final_answer" not in llm.calls[0]["tools"]
    assert end.claims == [] and end.final_text == "plain prose"


async def test_the_lead_puts_the_claims_on_its_reply_and_journals_them(monkeypatch, tmp_path):
    from helioai.core import agent_loop
    from helioai.core.session import SessionStore

    test_store = SessionStore(tmp_path / "sessions.db")
    monkeypatch.setattr(agent_loop, "store", test_store)
    claims = [{"name": "theta_bn", "value": 57.5, "units": "deg", "source": "theta_bn"}]
    llm = ScriptedLLM([_final_answer("θ_Bn = 57.5°.", claims)])

    live = [ev async for ev in agent_loop.stream_chat(llm, "web", "s-claims", "q")]
    reply = next(e for e in live if e["event"] == "reply")
    assert reply["data"]["text"] == "θ_Bn = 57.5°." and reply["data"]["claims"] == [
        {"name": "theta_bn", "value": 57.5, "units": "deg", "source": "theta_bn"}
    ]
    journaled = next(e for e in test_store.events("web", "s-claims") if e["event"] == "reply")
    assert journaled["data"]["claims"] == reply["data"]["claims"]
    saved = test_store.get_or_create("web", "s-claims")
    assert saved[-1].role == "assistant" and saved[-1].content == "θ_Bn = 57.5°."


# ── search budget ─────────────────────────────────────────────────────────────────


def _hits(*ids: str) -> dict:
    return {"results": [{"id": i, "name": i.rsplit("/", 1)[-1]} for i in ids]}


async def test_a_role_that_keeps_searching_is_told_to_use_the_ids_it_has():
    """The live data_analyst made twelve search_parameters calls and no download,
    re-asking for `Proton_Temp` while `Proton_W_nonlin` sat in its first results. Past
    the budget, one correction lists the ids its searches returned and tells it to
    download; the next turn sees it, and the run goes on."""
    reg = ScriptedRegistry(
        {
            "search_parameters": _hits(
                "cda/WI_H1_SWE/Proton_Np_nonlin", "cda/WI_H1_SWE/Proton_W_nonlin"
            ),
            "get_timeseries": {"ok": True},
        }
    )
    llm = ScriptedLLM(
        [
            assistant_calls("search_parameters", arguments={"query": "a"}),
            assistant_calls("search_parameters", arguments={"query": "b"}),
            assistant_calls("search_parameters", arguments={"query": "c"}),
            assistant_calls("search_parameters", arguments={"query": "d"}),
            assistant_calls(
                "get_timeseries", arguments={"param_id": "cda/WI_H1_SWE/Proton_Np_nonlin"}
            ),
            assistant_text("downloaded"),
        ]
    )
    policy = _policy(max_turns=8, search_budget=3)
    history = [Message(role="user", content="q")]
    events, end = await _drain(Runner(policy, llm, registry=reg), history)

    corrections = [e for e in events if e["event"] == "correction"]
    assert len(corrections) == 1, "once per run"
    note = corrections[0]["data"]["text"]
    assert "4 searches" in note and "cda/WI_H1_SWE/Proton_W_nonlin" in note
    assert corrections[0]["data"]["ids"][:2] == [
        "cda/WI_H1_SWE/Proton_Np_nonlin",
        "cda/WI_H1_SWE/Proton_W_nonlin",
    ]
    persisted = [m for m in history if m.origin == "correction"]
    assert len(persisted) == 1 and persisted[0].role == "user"
    fifth_call = llm.calls[4]["messages"]
    assert fifth_call[-1].content == note, "the model reads the correction before its next turn"
    assert end.final_text == "downloaded" and not end.capped


async def test_a_role_that_downloaded_early_is_never_corrected_however_much_it_searches():
    reg = ScriptedRegistry({"search_parameters": _hits("x/y/z"), "get_timeseries": {"ok": True}})
    llm = ScriptedLLM(
        [
            assistant_calls("get_timeseries", arguments={"param_id": "x/y/z"}),
            *[assistant_calls("search_parameters", arguments={"query": str(i)}) for i in range(5)],
            assistant_text("done"),
        ]
    )
    events, _ = await _drain(
        Runner(_policy(max_turns=8, search_budget=3), llm, registry=reg),
        [Message(role="user", content="q")],
    )
    assert not [e for e in events if e["event"] == "correction"]


async def test_a_zero_budget_disables_the_guard():
    reg = ScriptedRegistry({"search_parameters": _hits("x/y/z")})
    llm = ScriptedLLM(
        [
            *[assistant_calls("search_parameters", arguments={"query": str(i)}) for i in range(5)],
            assistant_text("done"),
        ]
    )
    events, _ = await _drain(
        Runner(_policy(max_turns=8), llm, registry=reg), [Message(role="user", content="q")]
    )
    assert not [e for e in events if e["event"] == "correction"]


def test_the_roles_have_the_budgets_their_jobs_call_for():
    from helioai.core.sub_agents import AGENT_ROLES

    assert AGENT_ROLES["data_analyst"].search_budget == 3
    assert AGENT_ROLES["plasma_physicist"].search_budget == 2
    assert AGENT_ROLES["parameter_hunter"].search_budget == 0, "searching is its whole job"
    assert AGENT_ROLES["librarian"].search_budget == 0
