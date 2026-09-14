"""Tests for helioai.tools.setup — the declarative tool registration surface.

setup.py is what the agent actually sees: 17 `registry.register(...)` calls whose
JSON Schemas are the only contract between the model and the Python functions.
A schema that drifts from its implementation is invisible to every other test —
the model simply starts sending arguments the function does not accept — so this
module pins the wiring itself.
"""

from __future__ import annotations

import inspect
import json

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
def test_schema_defaults_match_the_signature_defaults(name):
    """A `default` in the schema is a promise about what happens when the model omits
    the argument — and what happens is the *signature's* default. `run_python.timeout`
    advertised 30 s for months after the signature moved to 60 s, so the model was
    reasoning about a budget the sandbox did not have.
    """
    tool = _tools()[name]
    sig = inspect.signature(tool.func)
    for prop, schema in tool.parameters.get("properties", {}).items():
        if "default" not in schema or prop not in sig.parameters:
            continue
        assert schema["default"] == sig.parameters[prop].default, (
            f"{name}.{prop}: schema default {schema['default']!r} but the function "
            f"defaults to {sig.parameters[prop].default!r}"
        )


@pytest.mark.parametrize("name", sorted(EXPECTED_TOOLS))
def test_required_and_optional_agree_with_the_signature(name):
    """A parameter without a default must be `required`, or the model may omit it and
    the call dies with a TypeError; one with a default must not be, or the model is
    forced to invent a value it did not need."""
    tool = _tools()[name]
    sig = inspect.signature(tool.func)
    required = set(tool.parameters.get("required", []))
    for prop in tool.parameters.get("properties", {}):
        param = sig.parameters.get(prop)
        if param is None:
            continue
        has_default = param.default is not inspect.Parameter.empty
        assert (prop in required) == (not has_default), (
            f"{name}.{prop}: required={prop in required} but signature default="
            f"{'<none>' if not has_default else param.default!r}"
        )


@pytest.mark.parametrize("name", sorted(EXPECTED_TOOLS))
def test_every_public_parameter_is_declared_in_the_schema(name):
    """The reverse of the property check: a public parameter the schema does not
    mention is one the model can never use. Underscore-prefixed parameters are the
    framework's trusted channel and stay hidden on purpose."""
    tool = _tools()[name]
    sig = inspect.signature(tool.func)
    declared = set(tool.parameters.get("properties", {}))
    public = {
        p.name
        for p in sig.parameters.values()
        if not p.name.startswith("_")
        and p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }
    assert public <= declared, (
        f"{name}: {public - declared} exist on the function, not in the schema"
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


async def test_an_exception_without_a_message_is_still_reported_as_an_error():
    """Audit probe: `raise TimeoutError()` became `{"error": ""}`, and the human-facing
    description read that falsy string as success — "ok" for a tool that never ran.
    """
    from helioai.core.event_display import describe_tool_result
    from helioai.tools.registry import ToolRegistry

    reg = ToolRegistry()

    @reg.register(name="boom", description="raises silently", parameters={"type": "object"})
    async def boom() -> dict:
        raise TimeoutError()

    result = await reg.call_tool("boom", {})
    assert json.loads(result)["error"] == "TimeoutError"
    assert describe_tool_result("boom", result).lower().startswith("error")
