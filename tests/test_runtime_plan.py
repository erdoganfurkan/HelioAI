"""The plan as data, and the report of how the turn followed it — never a block."""

from __future__ import annotations

import pytest

from helioai.core import events
from helioai.core.llm.base import Message, ToolCall
from helioai.runtime.plan import Plan, Step, adherence, executed_tools
from tests.support.scripted import ScriptedLLM, ScriptedRegistry, assistant_calls, assistant_text

PLAN = {
    "title": "theta_Bn at the WIND shock",
    "steps": [
        {"description": "resolve B and V", "tool": "search_parameters"},
        {"description": "download the interval", "tool": "get_timeseries"},
        {"description": "coplanarity, recipe theta_bn", "tool": "run_python"},
        {"description": "write it up"},
    ],
}


def _call(name: str, **data) -> dict:
    return events.make("tool_call", turn=1, name=name, arguments={}, display=name, **data)


# ── the plan as data ───────────────────────────────────────────────────────────────


def test_a_plan_is_read_from_the_payload_with_its_looseness_removed():
    plan = Plan.from_payload(
        {
            "title": "t",
            "steps": [
                {"description": "a", "tool": "search_parameters"},
                {"description": "b", "tool": "  "},
                "a bare string step",
                42,
                {"tool": "run_python"},
            ],
        }
    )
    assert plan.steps == (
        Step("a", "search_parameters"),
        Step("b", None),
        Step("a bare string step", None),
        Step("", "run_python"),
    )
    assert plan.tools == ["search_parameters", "run_python"]


def test_a_payload_with_nothing_in_it_is_an_empty_plan():
    assert Plan.from_payload({}) == Plan("", ())
    assert Plan.from_payload({"title": None, "steps": None}).tools == []


def test_a_tool_named_by_two_steps_is_planned_once():
    plan = Plan.from_payload(
        {"title": "t", "steps": [{"description": "a", "tool": "run_python"}] * 3}
    )
    assert plan.tools == ["run_python"]


# ── what counts as executed ────────────────────────────────────────────────────────


def test_only_the_leads_own_analysis_calls_count():
    turn = [
        _call("present_plan"),
        _call("load_skill"),
        _call("search_tools"),
        _call("search_parameters"),
        _call("task"),
        _call("get_timeseries", sub_agent_ctx={"role": "data_analyst", "task_id": "t1"}),
        _call("run_python", sub_agent_ctx={"role": "data_analyst", "task_id": "t1"}),
        _call("search_parameters"),
        _call("final_answer"),
        events.make("reply", text="…"),
    ]
    assert executed_tools(turn) == ["search_parameters", "task"]


# ── the report, healthy case first ─────────────────────────────────────────────────


def test_a_plan_followed_to_the_letter_reports_a_full_ratio_and_no_differences():
    plan = Plan.from_payload(PLAN)
    turn = [_call("search_parameters"), _call("get_timeseries"), _call("run_python")]
    report = adherence(plan, turn)
    assert report == {
        "title": "theta_Bn at the WIND shock",
        "planned": ["search_parameters", "get_timeseries", "run_python"],
        "executed": ["search_parameters", "get_timeseries", "run_python"],
        "unplanned_tools": [],
        "missed_tools": [],
        "ratio": 1.0,
    }


def test_a_step_skipped_and_a_tool_improvised_are_both_named():
    plan = Plan.from_payload(PLAN)
    turn = [_call("search_parameters"), _call("find_papers"), _call("run_python")]
    report = adherence(plan, turn)
    assert report["missed_tools"] == ["get_timeseries"]
    assert report["unplanned_tools"] == ["find_papers"]
    assert report["ratio"] == pytest.approx(0.67)


def test_a_plan_that_names_no_tool_has_no_ratio_but_still_says_what_ran():
    plan = Plan.from_payload({"title": "vague", "steps": ["look", "think", "answer"]})
    report = adherence(plan, [_call("search_parameters")])
    assert report["ratio"] is None
    assert report["planned"] == [] and report["executed"] == ["search_parameters"]
    assert report["unplanned_tools"] == ["search_parameters"]


def test_the_report_is_a_contract_event():
    ev = events.make("plan_report", **adherence(Plan.from_payload(PLAN), []))
    assert (
        ev["data"]["ratio"] == 0.0 and ev["data"]["missed_tools"] == Plan.from_payload(PLAN).tools
    )


# ── through the lead ───────────────────────────────────────────────────────────────


def _present_plan() -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id="p1", name="present_plan", arguments=PLAN)],
    )


async def _lead(monkeypatch, tmp_path, llm, question="hi"):
    from helioai.core import agent_loop
    from helioai.core.session import SessionStore

    store = SessionStore(tmp_path / "sessions.db")
    monkeypatch.setattr(agent_loop, "store", store)
    monkeypatch.setattr(agent_loop, "registry", ScriptedRegistry())
    live = [ev async for ev in agent_loop.stream_chat(llm, "web", "s-plan", question)]
    return live, store


async def test_a_turn_that_presented_a_plan_ends_with_one_report_journaled_before_done(
    monkeypatch, tmp_path
):
    llm = ScriptedLLM(
        [
            _present_plan(),
            assistant_calls("search_parameters", "get_timeseries"),
            assistant_text("θ_Bn is 57°."),
        ]
    )
    live, store = await _lead(monkeypatch, tmp_path, llm)
    kinds = [e["event"] for e in live]
    assert kinds.count("plan_report") == 1
    assert kinds.index("reply") < kinds.index("plan_report") == len(kinds) - 2
    assert kinds[-1] == "done"
    report = live[kinds.index("plan_report")]["data"]
    assert report["missed_tools"] == ["run_python"] and report["unplanned_tools"] == []
    assert report["ratio"] == pytest.approx(0.67)
    journaled = [e for e in store.events("web", "s-plan") if e["event"] == "plan_report"]
    assert journaled == [live[kinds.index("plan_report")]]


async def test_a_turn_without_a_plan_has_no_report(monkeypatch, tmp_path):
    llm = ScriptedLLM([assistant_calls("search_parameters"), assistant_text("done")])
    live, _ = await _lead(monkeypatch, tmp_path, llm)
    assert "plan_report" not in [e["event"] for e in live]


async def test_a_capped_turn_still_reports_how_far_the_plan_got(monkeypatch, tmp_path):
    """The report is most useful exactly when the run did not finish: it says which
    planned steps happened before the cap. It precedes the error, and never replaces it."""
    from helioai.config import settings

    monkeypatch.setattr(settings.agent, "max_iterations", 2)
    llm = ScriptedLLM([_present_plan(), assistant_calls("search_parameters")])
    live, _ = await _lead(monkeypatch, tmp_path, llm)
    kinds = [e["event"] for e in live]
    assert kinds[-2:] == ["plan_report", "error"]
    report = live[-2]["data"]
    assert report["executed"] == ["search_parameters"]
    assert report["missed_tools"] == ["get_timeseries", "run_python"]
