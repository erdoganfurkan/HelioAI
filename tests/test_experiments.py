"""The behaviours under evaluation are off unless named, and each switch moves only its own piece.

Four changes to what the model sees or is told — hidden tools, `final_answer`, the search
budget, the variable list on a search hit — each shipped on the strength of a single live
run, and none was measured against the loop it replaced. These tests pin the contract that
makes them measurable: with `HELIOAI_EXPERIMENTS` unset the lead and the roles behave as
the reference loop did, and naming one experiment changes exactly its prompt text and its
`Policy` field, never another's. `final_answer` graduated on the third bench (34 clean
runs, four configurations: no cost on any question, the claim verdict back with zero
contradictions) and is now part of the default; the tests here pin that it is always
offered and always described.
"""

from __future__ import annotations

import pytest
from support.scripted import ScriptedLLM

import helioai.tools.setup  # noqa: F401  — populates the registry; without it it is empty
from helioai import config
from helioai.config import settings
from helioai.core import agent_loop, sub_agents
from helioai.core.llm.base import Message
from helioai.runtime.runner import Runner

# ── parsing ────────────────────────────────────────────────────────────────────


def test_no_experiments_by_default(monkeypatch):
    monkeypatch.delenv("HELIOAI_EXPERIMENTS", raising=False)
    assert config._load().agent.experiments == frozenset()


def test_experiments_parse_as_a_normalised_set(monkeypatch):
    monkeypatch.setenv("HELIOAI_EXPERIMENTS", " Search_Budget, deferred_tools ,, ")
    assert config._load().agent.experiments == {"search_budget", "deferred_tools"}


def test_an_unknown_experiment_is_refused_where_the_model_is_built(monkeypatch):
    """`import helioai.config` must not fail on a typo — the MCP server and the web app
    start from it before any model is built (PR #11 made the import key-free for that).
    The refusal lives where the API key is already checked, and in `doctor`."""
    from helioai.core.llm.factory import build_llm_client
    from helioai.doctor import check_experiments

    monkeypatch.setenv("HELIOAI_EXPERIMENTS", "final_anwser")
    loaded = config._load()
    assert loaded.agent.experiments == {"final_anwser"}
    with pytest.raises(RuntimeError, match="final_anwser"):
        config.validate_experiments(loaded.agent.experiments)

    monkeypatch.setattr(settings.agent, "experiments", frozenset({"final_anwser"}))
    with pytest.raises(RuntimeError, match="final_anwser"):
        build_llm_client("groq")
    assert check_experiments().status == "fail"
    monkeypatch.setattr(settings.agent, "experiments", frozenset({"search_budget"}))
    assert check_experiments().status == "ok"


def test_every_experiment_the_code_consults_is_a_known_name():
    assert config.EXPERIMENTS == {"deferred_tools", "search_budget", "search_variables"}
    assert "final_answer" not in config.EXPERIMENTS, "graduated on bench 3; it is the default"


# ── the lead's prompt ──────────────────────────────────────────────────────────


def test_default_lead_prompt_describes_final_answer_but_no_hidden_tools():
    prompt = agent_loop.build_lead_system_prompt(False, frozenset())
    assert prompt == agent_loop.SYSTEM_PROMPT
    assert "search_tools" not in prompt
    assert "## Closing an analysis" in prompt and "`final_answer(answer, claims)`" in prompt
    assert prompt.index("## Closing an analysis") < prompt.index("## Workflow rules")
    assert "whole UT day" in prompt, "the interval rule the third bench asked for"


def test_deferred_tools_adds_its_sentence_under_the_tools_heading_only():
    prompt = agent_loop.build_lead_system_prompt(False, frozenset({"deferred_tools"}))
    heading = "## Tools (arguments in each schema)\n"
    assert prompt.count(agent_loop.DEFERRED_TOOLS_NOTE) == 1
    assert prompt.index(heading) + len(heading) == prompt.index(agent_loop.DEFERRED_TOOLS_NOTE)
    assert prompt.replace(agent_loop.DEFERRED_TOOLS_NOTE, "", 1) == agent_loop.SYSTEM_PROMPT


def test_the_experiment_composes_with_the_guardrail():
    prompt = agent_loop.build_lead_system_prompt(True, frozenset({"deferred_tools"}))
    assert agent_loop.DEFERRED_TOOLS_NOTE in prompt
    assert "## Closing an analysis" in prompt
    assert prompt.endswith(agent_loop.SCOPE_GUARDRAIL)


def test_the_prompt_reads_the_settings_when_no_set_is_given(monkeypatch):
    monkeypatch.setattr(settings.agent, "experiments", frozenset({"deferred_tools"}))
    assert agent_loop.DEFERRED_TOOLS_NOTE in agent_loop.build_lead_system_prompt(False)
    monkeypatch.setattr(settings.agent, "experiments", frozenset())
    assert agent_loop.build_lead_system_prompt(False) == agent_loop.SYSTEM_PROMPT


# ── what the lead's model is actually shown ────────────────────────────────────


async def _lead_first_call(monkeypatch, tmp_path, experiments: frozenset[str]) -> dict:
    """Run one text-only lead turn and return what the model received on its call."""
    from helioai.core.session import SessionStore

    monkeypatch.setattr(agent_loop, "store", SessionStore(tmp_path / "sessions.db"))
    monkeypatch.setattr(settings.agent, "experiments", experiments)
    llm = ScriptedLLM([Message(role="assistant", content="hello")])
    events = [ev async for ev in agent_loop.stream_chat(llm, "web", "s1", "hi", restricted=False)]
    assert events[-1]["event"] == "done"
    return llm.calls[0]


@pytest.mark.asyncio
async def test_by_default_the_lead_sees_every_tool_and_final_answer_but_no_search_tools(
    monkeypatch, tmp_path
):
    call = await _lead_first_call(monkeypatch, tmp_path, frozenset())
    shown = set(call["tools"])
    assert {
        "plasma_beta",
        "list_catalogs",
        "get_events_timeseries",
        "task",
        "final_answer",
    } <= shown
    assert "search_tools" not in shown
    assert call["system_prompt"] == agent_loop.SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_deferred_tools_hides_the_formulary_and_offers_search_tools(monkeypatch, tmp_path):
    call = await _lead_first_call(monkeypatch, tmp_path, frozenset({"deferred_tools"}))
    shown = set(call["tools"])
    assert "search_tools" in shown
    assert not agent_loop._DEFERRED_TOOLS & shown
    assert "final_answer" in shown
    assert agent_loop.DEFERRED_TOOLS_NOTE in call["system_prompt"]


# ── the roles' search budget ───────────────────────────────────────────────────


@pytest.fixture
def role_policies(monkeypatch):
    """Record the `Policy` each delegated role is run under."""
    seen: list = []

    class RecordingRunner(Runner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen.append(self.policy)

    monkeypatch.setattr(sub_agents, "Runner", RecordingRunner)
    return seen


async def _run_role(role: str) -> None:
    llm = ScriptedLLM([Message(role="assistant", content="nothing to report")])
    async for _ in sub_agents.stream_subagent(
        role=role,
        description="anything",
        parent_session_id="sess-1",
        user_id="cli",
        llm_client=llm,
    ):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["data_analyst", "plasma_physicist"])
async def test_roles_have_no_search_budget_by_default(monkeypatch, role_policies, role):
    monkeypatch.setattr(settings.agent, "experiments", frozenset())
    await _run_role(role)
    assert role_policies[-1].search_budget == 0


@pytest.mark.asyncio
async def test_the_search_budget_experiment_applies_each_roles_allowance(
    monkeypatch, role_policies
):
    monkeypatch.setattr(settings.agent, "experiments", frozenset({"search_budget"}))
    await _run_role("data_analyst")
    await _run_role("plasma_physicist")
    await _run_role("parameter_hunter")
    assert [p.search_budget for p in role_policies] == [3, 2, 0]


# ── the prompts the roles carry ────────────────────────────────────────────────


@pytest.mark.parametrize("role", ["data_analyst", "plasma_physicist"])
def test_role_prompts_are_the_pre_experiment_text_and_run_recipe_stays_reachable(role):
    """The addons are the reference loop's; `run_recipe` is in both whitelists. The
    `data_analyst` skill carries the two paragraphs the third bench asked for — shock
    candidates screened against the plasma, and the window spread as the ± to report —
    and nothing else that names `run_recipe`."""
    from helioai.core.skills_loader import list_skills, load_skill

    addon = sub_agents.AGENT_ROLES[role].system_addon
    assert "load_recipe" in addon
    assert "run_recipe" not in addon
    assert "run_recipe" in sub_agents.AGENT_ROLES[role].allowed_tools
    meta = next(m for m in list_skills() if m.name == role)
    assert "run_recipe" in meta.allowed_tools
    body = load_skill(role)
    assert "load_recipe" in body
    if role == "data_analyst":
        assert "theta_bn_window_spread_deg" in body
        assert body.count("run_recipe(") == 1, "only the shock-candidate screen names the tool"
    else:
        assert "run_recipe" not in body
