"""Tests for stream_subagent — the delegation engine.

This function had no test, and that gap has already cost a production bug: when
`_extract_artifact` changed signature, agent_loop was updated and sub_agents was
not, and CI stayed green because nothing exercised this path (lessons.md,
2026-06-02).

The tests drive it with a scripted LLM and a stubbed registry, so no network and
no real tool ever runs. They assert on the emitted event stream, which is the
sub-agent's entire contract with the agent loop and the UI.
"""

from __future__ import annotations

import json

import pytest

import helioai.tools.setup  # noqa: F401  — populates the registry; without it it is empty
from helioai.core import sub_agents
from helioai.core.llm.base import LLMClient, Message, ToolCall


class ScriptedLLM(LLMClient):
    """Replays scripted responses and records how it was called."""

    def __init__(self, responses: list[Message]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def chat(self, messages, tools, system_prompt=None, tool_choice="auto"):
        self.calls.append(
            {
                "messages": list(messages),
                "tools": [t.name for t in tools],
                "system_prompt": system_prompt,
                "tool_choice": tool_choice,
            }
        )
        return self._responses.pop(0)


def text(content: str) -> Message:
    return Message(role="assistant", content=content)


def calls(*names: str) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id=f"call_{i}", name=n, arguments={}) for i, n in enumerate(names)],
    )


@pytest.fixture
def stub_registry(monkeypatch):
    """Replace tool dispatch with a recorder; no real tool is ever invoked."""
    invoked: list[str] = []
    results: dict[str, str] = {}

    async def fake_call_tool(name, arguments, *, trusted=None):
        invoked.append(name)
        return results.get(name, json.dumps({"ok": True}))

    monkeypatch.setattr(sub_agents.registry, "call_tool", fake_call_tool)
    return invoked, results


async def drain(**kwargs) -> list[dict]:
    return [ev async for ev in sub_agents.stream_subagent(**kwargs)]


def base(llm, role="parameter_hunter", description="find Bz") -> dict:
    return {
        "role": role,
        "description": description,
        "parent_session_id": "sess-1",
        "user_id": "cli",
        "llm_client": llm,
    }


def final(events: list[dict]) -> dict:
    return events[-1]["data"]


# ── role validation ────────────────────────────────────────────────────────────


async def test_unknown_role_ends_with_an_error_instead_of_raising():
    events = await drain(**base(ScriptedLLM([]), role="astrologer"))
    assert len(events) == 1
    data = final(events)
    assert "unknown agent_role" in data["error"]
    assert "parameter_hunter" in data["error"], "the error should list the valid roles"
    assert data["n_iterations"] == 0


# ── happy path ─────────────────────────────────────────────────────────────────


async def test_text_only_response_ends_the_sub_agent(stub_registry, monkeypatch):
    # No id in the answer, so id verification never touches the index.
    events = await drain(**base(ScriptedLLM([text("resolved the parameter")])))
    data = final(events)
    assert events[-1]["event"] == "sub_agent_end"
    assert data["summary"] == "resolved the parameter"
    assert data["error"] is None
    assert data["n_iterations"] == 1


async def test_summary_is_truncated_to_200_chars(stub_registry):
    events = await drain(**base(ScriptedLLM([text("x" * 500)])))
    assert len(final(events)["summary"]) == 200


async def test_tool_call_emits_call_then_result(stub_registry):
    llm = ScriptedLLM([calls("search_parameters"), text("done")])
    events = await drain(**base(llm))
    names = [e["event"] for e in events]
    assert "tool_call" in names
    assert names.index("tool_call") < names.index("tool_result")


async def test_tool_is_actually_dispatched(stub_registry):
    invoked, _ = stub_registry
    llm = ScriptedLLM([calls("search_parameters"), text("done")])
    await drain(**base(llm))
    assert invoked == ["search_parameters"]


async def test_every_event_carries_the_sub_agent_context(stub_registry):
    """The UI uses sub_agent_ctx to indent and attribute nested activity."""
    llm = ScriptedLLM([calls("search_parameters"), text("done")])
    events = await drain(**base(llm))
    for ev in events:
        if ev["event"] == "sub_agent_end":
            continue
        assert ev["data"]["sub_agent_ctx"]["role"] == "parameter_hunter"
        assert ev["data"]["sub_agent_ctx"]["task_id"]


async def test_task_id_is_honoured_when_supplied(stub_registry):
    events = await drain(**base(ScriptedLLM([text("ok")])), task_id="fixed123")
    assert final(events)["task_id"] == "fixed123"


# ── tool_choice sequencing ─────────────────────────────────────────────────────


async def test_first_turn_forces_a_tool_call_then_relaxes(stub_registry):
    """A sub-agent that answers from memory on turn one has not done its job."""
    llm = ScriptedLLM([calls("search_parameters"), text("done")])
    await drain(**base(llm))
    assert llm.calls[0]["tool_choice"] == "required"
    assert llm.calls[1]["tool_choice"] == "auto"


# ── tool whitelisting (security boundary) ──────────────────────────────────────


async def test_tool_outside_the_role_whitelist_is_refused(stub_registry):
    invoked, _ = stub_registry
    llm = ScriptedLLM([calls("run_python"), text("done")])
    await drain(**base(llm, role="parameter_hunter"))
    assert invoked == [], "a denied tool must never reach the registry"


async def test_refusal_is_reported_back_to_the_model(stub_registry):
    llm = ScriptedLLM([calls("run_python"), text("done")])
    await drain(**base(llm, role="parameter_hunter"))
    tool_reply = llm.calls[1]["messages"][-1]
    assert tool_reply.role == "tool"
    payload = json.loads(tool_reply.content)
    assert "not available" in payload["error"]
    assert "search_parameters" in payload["error"], "the model should be told what it may use"


async def test_only_whitelisted_tools_are_offered(stub_registry):
    """The model is never shown a tool its role may not call."""
    llm = ScriptedLLM([text("ok")])
    await drain(**base(llm, role="parameter_hunter"))
    allowed = set(sub_agents.AGENT_ROLES["parameter_hunter"].allowed_tools)
    assert llm.calls[0]["tools"], "the registry must be populated"
    assert set(llm.calls[0]["tools"]) == allowed
    assert "run_python" not in llm.calls[0]["tools"]


# ── turn cap ───────────────────────────────────────────────────────────────────


async def test_reaching_the_turn_cap_ends_cleanly(stub_registry):
    role = sub_agents.AGENT_ROLES["parameter_hunter"]
    llm = ScriptedLLM([calls("search_parameters") for _ in range(role.max_turns)])

    events = await drain(**base(llm))
    data = final(events)
    assert data["n_iterations"] == role.max_turns
    assert "cap" in data["summary"]
    assert data["error"] is None, "hitting the cap is not an error"


# ── artifacts ──────────────────────────────────────────────────────────────────


async def test_artifacts_are_collected_without_the_sub_agent_context(stub_registry, monkeypatch):
    """Regression guard for the bug in lessons.md.

    Artifacts are re-accumulated with sub_agent_ctx stripped so the persisted
    shape matches what the lead agent produces.
    """
    monkeypatch.setattr(
        sub_agents,
        "emit_post_tool_events",
        lambda name, result, tool_result_extra=None, common_extra=None: [
            {
                "event": "artifact",
                "data": {"kind": "image", "figure_paths": ["/tmp/a.png"], **(common_extra or {})},
            }
        ],
    )
    llm = ScriptedLLM([calls("search_parameters"), text("done")])
    events = await drain(**base(llm))

    streamed = next(e for e in events if e["event"] == "artifact")
    assert "sub_agent_ctx" in streamed["data"], "the live event keeps the context for the UI"

    collected = final(events)["artifacts"]
    assert len(collected) == 1
    assert "sub_agent_ctx" not in collected[0], "the persisted artifact must not carry the context"
    assert collected[0]["figure_paths"] == ["/tmp/a.png"]


# ── failure handling ───────────────────────────────────────────────────────────


async def test_an_exception_becomes_an_error_event_not_a_crash(stub_registry, monkeypatch):
    """A sub-agent blowing up must not take the whole conversation with it."""

    class Boom(LLMClient):
        async def chat(self, *a, **kw):
            raise RuntimeError("provider exploded")

    events = await drain(**base(Boom()))
    data = final(events)
    assert events[-1]["event"] == "sub_agent_end"
    assert "provider exploded" in data["error"]
    assert data["artifacts"] == []


async def test_workspace_session_is_reset_even_on_failure(stub_registry):
    import helioai.workspace as ws

    class Boom(LLMClient):
        async def chat(self, *a, **kw):
            raise RuntimeError("nope")

    before = ws.get_session_id() if hasattr(ws, "get_session_id") else None
    await drain(**base(Boom()))
    after = ws.get_session_id() if hasattr(ws, "get_session_id") else None
    assert before == after, "the session contextvar must be restored"


# ── invented parameter ids ─────────────────────────────────────────────────────


def test_extract_ids_finds_every_provider_prefix():
    from helioai.tools.rag import extract_ids

    text = (
        "Use amda/imf_real_gse and cda/AC_H0_SWE/Np, plus "
        "csa/C3_CP_CIS-HIA_ONBOARD_MOMENTS/density__C3_CP_CIS-HIA_ONBOARD_MOMENTS."
    )
    assert extract_ids(text) == [
        "amda/imf_real_gse",
        "cda/AC_H0_SWE/Np",
        "csa/C3_CP_CIS-HIA_ONBOARD_MOMENTS/density__C3_CP_CIS-HIA_ONBOARD_MOMENTS",
    ]


def test_extract_ids_ignores_prose_without_ids():
    from helioai.tools.rag import extract_ids

    assert extract_ids("I could not find a suitable parameter for that.") == []


@pytest.fixture
def fake_index(monkeypatch):
    """Pretend the catalogue holds exactly one id."""
    real = "csa/C3_CP_CIS-HIA_ONBOARD_MOMENTS/density__C3_CP_CIS-HIA_ONBOARD_MOMENTS"

    class _Collection:
        def get(self, ids):
            return {"ids": [i for i in ids if i == real]}

    from helioai.tools import rag

    monkeypatch.setattr(rag, "_collection_only", lambda: _Collection())
    return real


async def test_invented_id_is_contradicted_in_the_summary(stub_registry, fake_index):
    """The real regression: a spliced id that exists in neither source dataset."""
    bogus = "csa/C3_PP_CIS/C3_CP_CIS-HIA_ONBOARD_MOMENTS/N_p__C3_CP_CIS-HIA_ONBOARD_MOMENTS"
    llm = ScriptedLLM([text(f"Use {bogus} for the ion density.")])

    events = await drain(**base(llm))

    flagged = [e for e in events if e["event"] == "invalid_ids"]
    assert flagged, "an id absent from the catalogue must raise an invalid_ids event"
    assert flagged[0]["data"]["ids"] == [bogus]
    assert flagged[0]["data"]["sub_agent_ctx"]["role"] == "parameter_hunter"


async def test_real_id_passes_silently(stub_registry, fake_index):
    llm = ScriptedLLM([text(f"Use {fake_index} for the ion density.")])
    events = await drain(**base(llm))
    assert not [e for e in events if e["event"] == "invalid_ids"]


def test_correction_names_the_bad_id_and_keeps_the_original_text(fake_index):
    """The wrong id is contradicted, not silently rewritten.

    The lead agent must be able to see that its sub-agent was unreliable, and
    guessing a replacement would repeat the original mistake.
    """
    from helioai.core.sub_agents import _flag_unknown_ids

    bogus = "csa/C3_PP_CIS/bogus__X"
    out, flagged = _flag_unknown_ids(f"Recommended: {bogus}")

    assert flagged == [bogus]
    assert "Recommended:" in out, "the original answer must remain visible"
    assert bogus in out
    assert "NOT in the catalogue" in out
    assert "search_parameters" in out, "the correction should say how to recover"


def test_verification_outage_raises_no_false_alarm(monkeypatch):
    """If the index cannot be read, stay silent rather than cry wolf."""
    from helioai.tools import rag

    def boom():
        raise RuntimeError("index missing")

    monkeypatch.setattr(rag, "_collection_only", boom)
    assert rag.unknown_ids(["cda/whatever/X"]) == []
