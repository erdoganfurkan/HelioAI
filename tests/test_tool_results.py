"""`ToolResult.for_llm()` is, byte for byte, what `registry.call_tool` used to return.

The fixtures under `tests/fixtures/tool_results/` are real results captured with the
string-returning registry on 2026-09-14 — sandbox runs (one with a figure and exports,
one failing), the recipe tools, a formulary call and its invalid-input error, a live
`search_parameters` against the 83k-product index, `list_missions`, `list_catalogs`, a
speasy download, the registry's own refusals, and the `task`, `present_plan` and
`list_skills` strings the lead recorded in a live session. Home paths are anonymised.
If `for_llm()` ever produced a different string for any of them, the model would be
reading something it never read before, and this file is where that would show.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helioai.tools.results import ToolResult

FIXTURES = sorted((Path(__file__).parent / "fixtures" / "tool_results").glob("*.json"))

# Not registry tools: the lead builds these as JSON *strings* (`json.dumps` with its
# ASCII-escaping default) and the model reads the string. They exercise the string
# path below, not the dict serialisation.
_STRING_RESULTS = {"task", "present_plan", "list_skills", "load_skill"}


def _load(path: Path) -> tuple[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["tool"], data["raw"]


def _legacy_is_error(raw: str) -> bool:
    try:
        payload = json.loads(raw)
    except ValueError:
        return False
    return isinstance(payload, dict) and bool(payload.get("error"))


def test_the_fixtures_cover_the_shapes_that_matter():
    names = {p.stem for p in FIXTURES}
    assert {
        "run_python_ok",
        "run_python_error",
        "search_parameters",
        "get_timeseries",
        "load_recipe",
        "task_data_analyst",
        "unknown_tool",
    } <= names


@pytest.mark.parametrize(
    "path",
    [
        p
        for p in FIXTURES
        if json.loads(p.read_text(encoding="utf-8"))["tool"] not in _STRING_RESULTS
    ],
    ids=lambda p: p.stem,
)
def test_a_dict_result_serialises_to_the_captured_text(path: Path):
    tool, raw = _load(path)
    assert tool not in _STRING_RESULTS
    payload = json.loads(raw)
    assert isinstance(payload, dict), "every shipped tool returns a dict"

    result = ToolResult.from_raw(tool, payload)

    assert result.payload is payload
    assert result.for_llm() == raw
    assert result.ok == (not _legacy_is_error(raw))


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_a_string_result_is_handed_back_untouched_and_still_readable(path: Path):
    """The internal tools and the `task` dispatch return JSON *strings*; the model must
    see exactly that string, and the readers must still get the object."""
    tool, raw = _load(path)

    result = ToolResult.from_raw(tool, raw)

    assert result.for_llm() == raw
    assert result.payload == json.loads(raw)


def test_non_ascii_survives_the_way_it_always_did():
    payload = {"stdout": "θ_Bn = 57.5°", "exports": {"angle_deg": {"mean": 57.5}}}
    assert ToolResult.from_raw("run_python", payload).for_llm() == json.dumps(
        payload, ensure_ascii=False, default=str
    )


def test_unknown_types_are_stringified_like_the_registry_did():
    from datetime import UTC, datetime

    when = datetime(2015, 3, 17, 4, 0, tzinfo=UTC)
    assert ToolResult.from_raw("t", {"when": when}).for_llm() == json.dumps(
        {"when": str(when)}, ensure_ascii=False
    )


def test_a_plain_text_result_is_kept_as_text():
    result = ToolResult.from_raw("remote_mcp_tool", "42 rows returned\n[non-text content omitted]")
    assert result.payload == "42 rows returned\n[non-text content omitted]"
    assert result.for_llm() == "42 rows returned\n[non-text content omitted]"
    assert result.ok


def test_failure_is_the_registry_error_line():
    result = ToolResult.failure("no_such_tool", "unknown tool 'no_such_tool'")
    assert result.for_llm() == json.dumps({"error": "unknown tool 'no_such_tool'"})
    assert not result.ok
    assert result.error == "unknown tool 'no_such_tool'"


def test_an_empty_error_message_still_reads_as_a_failure():
    """`str(TimeoutError())` is "" — a tool that never ran was once displayed as ok."""
    result = ToolResult.failure("get_timeseries", str(TimeoutError()))
    assert not result.ok
    assert result.error == "error"


def test_an_amended_payload_is_what_the_model_reads():
    """The figure review appends its verdict to a run_python payload; the text the model
    gets must carry it, so the original string cannot be reused."""
    original = ToolResult.from_raw(
        "run_python", json.dumps({"stdout": "ok", "figure_paths": ["a.png"]})
    )
    amended = original.with_payload({**original.payload, "figure_review": "axes labelled"})
    assert json.loads(amended.for_llm())["figure_review"] == "axes labelled"
    assert original.for_llm() == json.dumps({"stdout": "ok", "figure_paths": ["a.png"]})


def test_error_is_none_for_a_result_that_carries_error_none():
    """A `task` result carries `"error": null` on success; that is not a failure."""
    result = ToolResult.from_raw("task", {"findings": {}, "summary": "done", "error": None})
    assert result.ok
    assert result.error is None
