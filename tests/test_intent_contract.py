"""The intent contract, in observation: read once from the question, emitted after the
answer, written into no message. Every guarantee is a test here: the question reaches
every sub-agent through the context; the date is assembled by the code; a turn with the
judge off emits nothing and looks exactly like today's; a turn with the judge on emits one
`intent` event and its history is byte-identical to the turn with the judge off.
"""

from __future__ import annotations

import asyncio

import pytest

from helioai.config import settings
from helioai.core import judgment
from helioai.core.judgment import INTENT_QUESTIONS, Answers, contract_fields
from helioai.core.llm.base import Message
from helioai.runtime.context import RunContext


def test_the_question_rides_on_the_context_down_to_every_sub_agent(tmp_path):
    ctx = RunContext(
        user_id="u", session_id="s", session_dir=tmp_path, query="θ_Bn of the 2015-03-17 shock"
    )
    child = ctx.child(agent="data_analyst", task_id="t1")
    assert child.query == "θ_Bn of the 2015-03-17 shock" and child.agent == "data_analyst"
    assert RunContext(user_id="u", session_id="s", session_dir=tmp_path).query is None


def _answers(**values):
    full = {name: None for name in INTENT_QUESTIONS}
    full.update(values)
    return Answers(values=full, raw={}, model="jev-test", latency_ms=12.3)


def test_the_code_assembles_the_date_and_none_means_not_decided():
    """The judge names components over closed sets; ordering dates is arithmetic."""
    full = contract_fields(
        _answers(
            deliverable="value",
            quantity="MagneticField",
            frame="GSM",
            year="2015",
            month="03",
            day="17",
            event_named=True,
            uncertainty_required=False,
        )
    )
    assert full["date"] == "2015-03-17" and full["date_precision"] == "day"
    assert full["quantity"] == "MagneticField" and full["frame"] == "GSM"
    assert full["event_named"] is True and full["uncertainty_required"] is False
    assert full["method_named"] is None, "abstained is None, not False"
    assert "year" not in full and full["model"] == "jev-test" and full["latency_ms"] == 12.3

    month_only = contract_fields(_answers(year="2004", month="11", day=judgment.NONE))
    assert month_only["date"] == "2004-11" and month_only["date_precision"] == "month"
    year_only = contract_fields(_answers(year="2004"))
    assert year_only["date"] == "2004" and year_only["date_precision"] == "year"
    nothing = contract_fields(_answers(quantity=judgment.NONE))
    assert nothing["date"] is None and nothing["quantity"] is None


@pytest.mark.asyncio
async def test_intent_contract_asks_the_question_alone_and_abstains_on_blank(monkeypatch):
    seen = {}

    async def fake_ask(site, state, questions, *, decided=None):
        seen.update(site=site, state=state, questions=questions)
        return _answers(deliverable="figure")

    monkeypatch.setattr(judgment, "ask", fake_ask)
    assert (await judgment.intent_contract("   ")) is None and not seen
    answers = await judgment.intent_contract("plot Bz")
    assert seen["site"] == "intent" and seen["state"] == {"question": "plot Bz"}
    assert seen["questions"] is INTENT_QUESTIONS and answers["deliverable"] == "figure"


async def _collect(gen):
    return [ev async for ev in gen]


@pytest.mark.asyncio
async def test_a_turn_with_the_judge_on_emits_one_intent_event_and_writes_no_message(
    monkeypatch, tmp_path, fake_llm_factory
):
    from helioai.core import agent_loop
    from helioai.core.session import SessionStore

    monkeypatch.setattr(agent_loop, "store", SessionStore(tmp_path / "sessions.db"))
    monkeypatch.setattr(settings.judgment, "backend", "jev")
    monkeypatch.setattr(settings.judgment, "timeout_s", 1.0)
    monkeypatch.setattr(settings.agent, "experiments", frozenset({"judgment_intent"}))

    async def fake_ask(site, state, questions, *, decided=None):
        await asyncio.sleep(0.01)
        return _answers(
            deliverable="value", quantity="MagneticField", year="2015", month="03", day="17"
        )

    monkeypatch.setattr(judgment, "ask", fake_ask)
    llm = fake_llm_factory([Message(role="assistant", content="θ_Bn is 60.9°.")])
    events = await _collect(agent_loop.stream_chat(llm, "u", "s-on", "θ_Bn on 2015-03-17"))

    kinds = [e["event"] for e in events]
    assert kinds.count("intent") == 1 and kinds.index("intent") < kinds.index("done")
    intent = next(e for e in events if e["event"] == "intent")["data"]
    assert intent["deliverable"] == "value" and intent["date"] == "2015-03-17"
    assert intent["quantity"] == "MagneticField"
    history_on = agent_loop.store.get_or_create("u", "s-on")
    assert [m.role for m in history_on] == ["user", "assistant"], "the contract enters no message"

    monkeypatch.setattr(settings.judgment, "backend", "null")
    llm = fake_llm_factory([Message(role="assistant", content="θ_Bn is 60.9°.")])
    events_off = await _collect(agent_loop.stream_chat(llm, "u", "s-off", "θ_Bn on 2015-03-17"))
    assert "intent" not in [e["event"] for e in events_off]
    history_off = agent_loop.store.get_or_create("u", "s-off")
    assert [(m.role, m.content) for m in history_off] == [(m.role, m.content) for m in history_on]


@pytest.mark.asyncio
async def test_a_slow_or_failing_judge_costs_the_turn_nothing(
    monkeypatch, tmp_path, fake_llm_factory
):
    from helioai.core import agent_loop
    from helioai.core.session import SessionStore

    monkeypatch.setattr(agent_loop, "store", SessionStore(tmp_path / "sessions.db"))
    monkeypatch.setattr(settings.judgment, "backend", "jev")
    monkeypatch.setattr(settings.judgment, "timeout_s", 0.05)
    monkeypatch.setattr(settings.agent, "experiments", frozenset({"judgment_intent"}))
    monkeypatch.setattr(judgment, "_warned", set())

    async def hangs(site, state, questions, *, decided=None):
        await asyncio.sleep(5)

    monkeypatch.setattr(judgment, "ask", hangs)
    llm = fake_llm_factory([Message(role="assistant", content="fine.")])
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    events = await _collect(agent_loop.stream_chat(llm, "u", "s", "anything"))
    assert loop.time() - t0 < 4.0, "never the 5 s the hanging judge would take"
    assert "intent" not in [e["event"] for e in events]
    assert events[-1]["event"] == "done"
    assert next(e for e in events if e["event"] == "reply")["data"]["text"] == "fine."


def test_every_intent_question_is_closed_or_yes_no():
    """The judge never writes free text into the system."""
    from helioai.core.judgment import Choice, Noul

    for name, q in INTENT_QUESTIONS.items():
        assert isinstance(q, (Choice, Noul)), name
        if isinstance(q, Choice):
            assert len(q.options) == len(set(q.options)), name
            assert judgment.NONE in q.options or "other" in q.options, f"{name} needs an out"
