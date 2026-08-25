"""Tests for sub-agent role definitions and tool filtering."""

from __future__ import annotations

import pytest

from helioai.core.sub_agents import AGENT_ROLES, TASK_TOOL_NAME, task_tool_def

# ──────────────────────────────── role definitions ──────────────────────────


def test_four_roles_defined() -> None:
    assert set(AGENT_ROLES.keys()) == {
        "parameter_hunter",
        "data_analyst",
        "plasma_physicist",
        "librarian",
    }


def test_parameter_hunter_tools() -> None:
    role = AGENT_ROLES["parameter_hunter"]
    assert "search_parameters" in role.allowed_tools
    assert "list_missions" in role.allowed_tools
    assert "get_timeseries" not in role.allowed_tools
    assert "run_python" not in role.allowed_tools


def test_data_analyst_tools() -> None:
    role = AGENT_ROLES["data_analyst"]
    assert "run_python" in role.allowed_tools
    assert "search_parameters" in role.allowed_tools
    assert "get_timeseries" in role.allowed_tools


def test_librarian_tools() -> None:
    role = AGENT_ROLES["librarian"]
    assert role.allowed_tools == ("find_papers",)
    assert role.max_turns <= 4
    assert "librarian" in role.auto_load_skills


def test_plasma_physicist_tools() -> None:
    role = AGENT_ROLES["plasma_physicist"]
    assert "run_python" in role.allowed_tools
    assert "get_timeseries" not in role.allowed_tools
    assert "search_parameters" in role.allowed_tools
    assert role.max_turns == 4


def test_plasma_physicist_can_reach_the_recipes() -> None:
    """Its skill points at rankine_hugoniot, theta_bn and walen_test.

    The whitelist is enforced, so without these the skill would advertise tools the
    role cannot call — and the role reinvented the jump conditions each time instead
    of using the recipe that ships with its scientific reference.
    """
    role = AGENT_ROLES["plasma_physicist"]
    assert "load_recipe" in role.allowed_tools
    assert "list_recipes" in role.allowed_tools


def test_parameter_hunter_has_lower_max_turns_than_data_analyst() -> None:
    assert AGENT_ROLES["parameter_hunter"].max_turns < AGENT_ROLES["data_analyst"].max_turns


# ──────────────────────────────── task_tool_def ──────────────────────────────


def test_task_tool_def_name() -> None:
    tdef = task_tool_def()
    assert tdef.name == TASK_TOOL_NAME


def test_task_tool_def_has_required_params() -> None:
    tdef = task_tool_def()
    required = tdef.parameters.get("required", [])
    assert "description" in required
    assert "agent_role" in required


def test_task_tool_def_enum_matches_roles() -> None:
    tdef = task_tool_def()
    enum_vals = set(tdef.parameters["properties"]["agent_role"]["enum"])
    assert enum_vals == set(AGENT_ROLES.keys())


def test_task_tool_def_description_mentions_all_roles() -> None:
    tdef = task_tool_def()
    for role_name in AGENT_ROLES:
        assert role_name in tdef.description


# ──────────────────────────────── role isolation ─────────────────────────────


@pytest.mark.parametrize(
    "role_name,forbidden",
    [
        ("parameter_hunter", ["get_timeseries", "run_python"]),
        ("data_analyst", ["list_missions"]),
        ("plasma_physicist", ["get_timeseries", "list_missions", "get_events_timeseries"]),
        ("librarian", ["run_python", "get_timeseries", "search_parameters"]),
    ],
)
def test_role_cannot_call_forbidden_tools(role_name: str, forbidden: list[str]) -> None:
    allowed = set(AGENT_ROLES[role_name].allowed_tools)
    for tool in forbidden:
        assert tool not in allowed, f"{role_name} should NOT have {tool}"


def test_task_tooldef_states_no_routing_rule() -> None:
    """Routing lives in the lead prompt alone.

    Stating it in both places let the two drift apart: the ToolDef called
    `parameter_hunter` "required" for unknown ids while the lead prompt said never to run
    it first. The model resolved that conflict differently run to run — three delegations,
    three, then zero on the same notebook.
    """
    from helioai.core.agent_loop import SYSTEM_PROMPT
    from helioai.core.sub_agents import task_tool_def

    desc = task_tool_def().description
    assert "Required when" not in desc
    assert "parameter_hunter" in SYSTEM_PROMPT
    # The role names still have to reach the model, just as an inventory, not as routing.
    for role in AGENT_ROLES:
        assert role in desc


def test_lead_prompt_gives_a_countable_delegation_test() -> None:
    """A fuzzy "skip delegation for something simple" left the choice to sampling."""
    from helioai.core.agent_loop import SYSTEM_PROMPT

    assert "count the stages" in SYSTEM_PROMPT.lower()
    assert "Skip delegation entirely" not in SYSTEM_PROMPT
