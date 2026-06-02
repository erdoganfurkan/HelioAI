"""Tests for sub-agent role definitions and tool filtering."""

from __future__ import annotations

import pytest

from helioai.core.sub_agents import AGENT_ROLES, TASK_TOOL_NAME, task_tool_def


# ──────────────────────────────── role definitions ──────────────────────────


def test_three_roles_defined() -> None:
    assert set(AGENT_ROLES.keys()) == {
        "parameter_hunter",
        "data_analyst",
        "plasma_physicist",
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
    assert "get_timeseries" not in role.allowed_tools


def test_plasma_physicist_tools() -> None:
    role = AGENT_ROLES["plasma_physicist"]
    assert "run_python" in role.allowed_tools
    assert "get_timeseries" not in role.allowed_tools


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
        ("data_analyst", ["get_timeseries", "list_missions"]),
        ("plasma_physicist", ["get_timeseries"]),
    ],
)
def test_role_cannot_call_forbidden_tools(role_name: str, forbidden: list[str]) -> None:
    allowed = set(AGENT_ROLES[role_name].allowed_tools)
    for tool in forbidden:
        assert tool not in allowed, f"{role_name} should NOT have {tool}"
