"""Tests for helioai.tools.setup — the declarative tool registration surface.

setup.py is what the agent actually sees: 17 `registry.register(...)` calls whose
JSON Schemas are the only contract between the model and the Python functions.
A schema that drifts from its implementation is invisible to every other test —
the model simply starts sending arguments the function does not accept — so this
module pins the wiring itself.
"""

from __future__ import annotations

import inspect

import pytest

import helioai.tools.setup  # noqa: F401  — importing performs the registration
from helioai.tools.registry import registry

EXPECTED_TOOLS = {
    "search_parameters",
    "list_missions",
    "get_timeseries",
    "plasma_beta",
    "gyrofrequency",
    "debye_length",
    "alfven_speed",
    "inertial_length",
    "power_spectrum",
    "run_python",
    "list_catalogs",
    "get_catalog",
    "get_events_timeseries",
    "save_catalog",
    "list_recipes",
    "load_recipe",
    "find_papers",
}


def _tools() -> dict:
    return {t.name: t for t in registry._tools.values()}


def test_expected_tools_are_registered():
    assert set(_tools()) == EXPECTED_TOOLS


@pytest.mark.parametrize("name", sorted(EXPECTED_TOOLS))
def test_tool_has_a_usable_description(name):
    """The description is the model's only clue about when to call a tool."""
    tool = _tools()[name]
    assert tool.description.strip(), f"{name} has no description"
    assert len(tool.description) > 20, f"{name} description is too terse to be useful"


@pytest.mark.parametrize("name", sorted(EXPECTED_TOOLS))
def test_tool_schema_is_a_well_formed_json_schema_object(name):
    params = _tools()[name].parameters
    assert params.get("type") == "object", f"{name} schema is not an object"
    assert isinstance(params.get("properties", {}), dict)
    for required in params.get("required", []):
        assert required in params["properties"], (
            f"{name} marks {required!r} required but never declares it"
        )


@pytest.mark.parametrize("name", sorted(EXPECTED_TOOLS))
def test_every_declared_property_exists_on_the_function(name):
    """The real failure mode: a schema advertising a parameter the function lacks.

    The model reads the schema, sends the argument, and the call dies with an
    unexpected-keyword TypeError at runtime.
    """
    tool = _tools()[name]
    sig = inspect.signature(tool.func)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        pytest.skip(f"{name} accepts **kwargs")
    accepted = set(sig.parameters)
    declared = set(tool.parameters.get("properties", {}))
    unknown = declared - accepted
    assert not unknown, f"{name} schema declares {unknown} which {tool.func.__name__} cannot accept"


@pytest.mark.parametrize("name", sorted(EXPECTED_TOOLS))
def test_required_parameters_are_accepted_by_the_function(name):
    tool = _tools()[name]
    sig = inspect.signature(tool.func)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        pytest.skip(f"{name} accepts **kwargs")
    for required in tool.parameters.get("required", []):
        assert required in sig.parameters, (
            f"{name} requires {required!r} but {tool.func.__name__} has no such parameter"
        )


@pytest.mark.parametrize("name", sorted(EXPECTED_TOOLS))
def test_every_tool_is_async(name):
    """registry.call_tool always awaits; a sync tool fails silently at dispatch."""
    assert inspect.iscoroutinefunction(_tools()[name].func), f"{name} is not async"


def test_tool_defs_are_exposed_to_the_llm():
    defs = {d.name for d in registry.list_tool_defs()}
    assert defs == EXPECTED_TOOLS


def test_tool_defs_can_be_filtered_for_sub_agents():
    defs = registry.list_tool_defs(only={"search_parameters"})
    assert [d.name for d in defs] == ["search_parameters"]


def test_task_is_agent_side_and_not_a_registry_tool():
    """`task` is deliberately not registered.

    Delegation is intercepted by the agent loop, which spawns a sub-agent rather
    than dispatching through the registry. It is offered to the model as a
    synthetic ToolDef, so it must never appear in the registry — if it did,
    registry.call_tool would try to await a function that does not exist.
    """
    from helioai.core.sub_agents import TASK_TOOL_NAME

    assert TASK_TOOL_NAME == "task"
    assert TASK_TOOL_NAME not in _tools()
