"""The plan as data, and the report of how the turn followed it — never a block."""

from __future__ import annotations

import pytest

from helioai.core import events
from helioai.core.llm.base import Message, ToolCall
from helioai.runtime.plan import Plan, Step, adherence, calls_made
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
    assert plan.tools() == ["search_parameters", "run_python"]


def test_a_payload_with_nothing_in_it_is_an_empty_plan():
    assert Plan.from_payload({}) == Plan("", ())
    assert Plan.from_payload({"title": None, "steps": None}).tools() == []


def test_a_tool_named_by_two_steps_is_planned_once():
    plan = Plan.from_payload(
        {"title": "t", "steps": [{"description": "a", "tool": "run_python"}] * 3}
    )
    assert plan.tools() == ["run_python"]


# ── what counts as executed ────────────────────────────────────────────────────────


def test_the_leads_own_calls_and_its_sub_agents_are_told_apart_and_scaffolding_dropped():
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
    assert calls_made(turn) == (["search_parameters", "task"], ["get_timeseries", "run_python"])


# ── the report, healthy case first ─────────────────────────────────────────────────


def test_a_plan_followed_to_the_letter_reports_a_full_ratio_and_no_differences():
    plan = Plan.from_payload(PLAN)
    turn = [_call("search_parameters"), _call("get_timeseries"), _call("run_python")]
    report = adherence(plan, turn)
    assert report == {
        "title": "theta_Bn at the WIND shock",
        "planned": ["search_parameters", "get_timeseries", "run_python"],
        "executed": ["search_parameters", "get_timeseries", "run_python"],
        "delegated": [],
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
    assert ev["data"]["ratio"] == 0.0
    assert ev["data"]["missed_tools"] == Plan.from_payload(PLAN).tools()


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
    monkeypatch.setattr(
        agent_loop,
        "registry",
        ScriptedRegistry({"search_parameters": {}, "get_timeseries": {}, "run_python": {}}),
    )
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


# ── what the first live run taught (2026-09-14, a876b27) ──────────────────────────

LIVE_PLAN = {
    "title": "Wind shock 2015-03-17 — |B|/Bz plot, theta_Bn & literature",
    "steps": [
        {"description": "Resolve and download", "tool": "search_parameters + get_timeseries"},
        {"description": "Plot |B| and Bz", "tool": "run_python"},
        {"description": "theta_Bn via the recipe", "tool": "run_recipe"},
        {"description": "Two papers", "tool": "find_papers"},
    ],
}
KNOWN = {"search_parameters", "get_timeseries", "run_python", "run_recipe", "find_papers", "task"}


def _sub(name: str, role: str) -> dict:
    return _call(name, sub_agent_ctx={"role": role, "task_id": f"t-{role}"})


def test_a_plan_carried_out_through_delegation_was_followed():
    """The lead planned the analysis in its sub-agents' tools, then delegated every step:
    the report said `0/4 planned tools used … unplanned: task`. A step done by the
    sub-agent the lead spawned for it was done; `task` is how, not a deviation."""
    turn = [
        _call("task"),
        _sub("search_parameters", "data_analyst"),
        _sub("get_timeseries", "data_analyst"),
        _sub("run_python", "data_analyst"),
        _sub("run_recipe", "data_analyst"),
        _call("task"),
        _sub("find_papers", "librarian"),
    ]
    report = adherence(Plan.from_payload(LIVE_PLAN), turn, known=KNOWN)
    assert report["planned"] == [
        "search_parameters",
        "get_timeseries",
        "run_python",
        "run_recipe",
        "find_papers",
    ]
    assert report["delegated"] == [
        "search_parameters",
        "get_timeseries",
        "run_python",
        "run_recipe",
        "find_papers",
    ]
    assert report["executed"] == ["task"]
    assert report["missed_tools"] == [] and report["unplanned_tools"] == []
    assert report["ratio"] == 1.0


def test_a_step_naming_two_tools_plans_both_and_an_unknown_word_plans_nothing():
    plan = Plan.from_payload(
        {
            "title": "t",
            "steps": [
                {"description": "d", "tool": "task (data_analyst)"},
                {"description": "e", "tool": "search_parameters + get_timeseries"},
            ],
        }
    )
    report = adherence(
        plan, [_call("task"), _sub("search_parameters", "data_analyst")], known=KNOWN
    )
    assert report["planned"] == ["task", "search_parameters", "get_timeseries"]
    assert report["missed_tools"] == ["get_timeseries"] and report["unplanned_tools"] == []


def test_a_sub_agents_tools_are_never_unplanned_but_a_planned_tool_nobody_ran_is_missed():
    """`task` never counts as unplanned — it is a means — and neither does what the role
    did with it: its whitelist governs that, not the lead's plan. The signal is the step
    nobody ran."""
    plan = Plan.from_payload({"title": "t", "steps": [{"description": "d", "tool": "run_python"}]})
    report = adherence(plan, [_call("task"), _sub("find_papers", "librarian")], known=KNOWN)
    assert report["unplanned_tools"] == [] and report["missed_tools"] == ["run_python"]
    assert report["delegated"] == ["find_papers"] and report["ratio"] == 0.0


def test_a_plan_written_in_delegations_alone_is_followed_by_delegating():
    """Second live run: every step said `task (data_analyst)` or `task (librarian)`; the
    analyst then called six tools, all reported as unplanned. They are the role's
    business."""
    plan = Plan.from_payload(
        {
            "title": "t",
            "steps": [
                {"description": "resolve, download, plot, θ_Bn", "tool": "task (data_analyst)"},
                {"description": "papers", "tool": "task (librarian)"},
            ],
        }
    )
    turn = [
        _call("task"),
        _sub("search_parameters", "data_analyst"),
        _sub("get_timeseries", "data_analyst"),
        _sub("load_recipe", "data_analyst"),
        _sub("run_python", "data_analyst"),
        _sub("run_recipe", "data_analyst"),
        _call("task"),
        _sub("find_papers", "librarian"),
    ]
    report = adherence(plan, turn, known=KNOWN)
    assert report["planned"] == ["task"] and report["ratio"] == 1.0
    assert report["unplanned_tools"] == [] and report["missed_tools"] == []
    assert len(report["delegated"]) == 6


def test_the_leads_own_improvisation_is_still_unplanned():
    plan = Plan.from_payload({"title": "t", "steps": [{"description": "d", "tool": "task"}]})
    report = adherence(plan, [_call("task"), _call("find_papers")], known=KNOWN)
    assert report["unplanned_tools"] == ["find_papers"]


def test_without_a_known_set_every_word_of_a_tool_field_is_taken_as_a_tool():
    plan = Plan.from_payload({"title": "t", "steps": [{"description": "d", "tool": "a + b"}]})
    assert adherence(plan, [])["planned"] == ["a", "b"]
