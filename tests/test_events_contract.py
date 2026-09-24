"""The event contract holds: what the loops emit is what `core/events.py` lists, and
what the interfaces render.

Two layers. The static one greps the source for the kinds the emitters name in
`make(...)` / `artifact(...)` and the renderers test for, and catches the drift this
exists for — a kind added to a loop and to two of the three interfaces, or documented
and never emitted — in milliseconds, on every run. The dynamic one is `make` itself:
every emission checks its required keys as it is produced, so the loop tests that
exercise a branch also prove its payload; the tests at the bottom pin that behaviour.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helioai.core import events

ROOT = Path(__file__).resolve().parents[1] / "helioai"

EMITTERS = [
    ROOT / "core" / "agent_loop.py",
    ROOT / "core" / "sub_agents.py",
    ROOT / "core" / "tool_exec.py",
    ROOT / "runtime" / "runner.py",
]
CLI = ROOT / "interfaces" / "cli.py"
JUPYTER = ROOT / "interfaces" / "jupyter_magic.py"
WEB_JS = ROOT / "interfaces" / "web" / "static" / "app.js"


def _literals(paths, pattern) -> set[str]:
    found: set[str] = set()
    for p in paths:
        found.update(re.findall(pattern, p.read_text(encoding="utf-8")))
    return found


def _emitted_kinds() -> set[str]:
    return _literals(EMITTERS, r'\bmake\(\s*"([a-z_]+)"')


def _emitted_artifact_kinds() -> set[str]:
    return _literals([ROOT / "core" / "tool_exec.py"], r'\bartifact\(\s*"([a-z_]+)"')


def test_every_emitted_kind_is_in_the_contract_and_vice_versa():
    assert _emitted_kinds() == set(events.KINDS)


def test_every_emitted_artifact_kind_is_in_the_contract_and_vice_versa():
    assert _emitted_artifact_kinds() == set(events.ARTIFACT_KINDS)


def test_no_emitter_builds_an_event_dict_by_hand():
    """A literal `{"event": ...}` would bypass the key check `make` performs."""
    assert _literals(EMITTERS, r'"event":\s*"([a-z_]+)"') == set()
    assert _literals([ROOT / "core" / "tool_exec.py"], r'"kind":\s*"([a-z_]+)"') == set()


def test_make_checks_the_kind_and_its_keys_and_keeps_the_shape():
    assert events.make("reply", text="hi") == {"event": "reply", "data": {"text": "hi"}}
    with_extra = events.make("invalid_ids", ids=["x"], sub_agent_ctx={"role": "r"})
    assert with_extra["data"]["sub_agent_ctx"] == {"role": "r"}
    with pytest.raises(ValueError, match="tool_result event is missing"):
        events.make("tool_result", turn=1, name="t")
    with pytest.raises(KeyError, match="not an event kind"):
        events.make("reply_chunk", text="hi")


def test_artifact_checks_the_kind_and_its_keys():
    art = events.artifact("image", tool="run_python", figure_paths=["f.png"])
    assert art == {"kind": "image", "tool": "run_python", "figure_paths": ["f.png"]}
    assert events.make("artifact", **art)["data"]["kind"] == "image"
    with pytest.raises(ValueError, match="code artifact is missing"):
        events.artifact("code", tool="run_python", code_path="c.py")
    with pytest.raises(KeyError, match="not an artifact kind"):
        events.artifact("table", tool="run_python")


@pytest.mark.parametrize(
    ("interface", "path", "pattern"),
    [
        ("cli", CLI, r'name == "([a-z_]+)"'),
        ("jupyter", JUPYTER, r'name == "([a-z_]+)"'),
        ("web", WEB_JS, r"event === '([a-z_]+)'"),
    ],
)
def test_every_interface_renders_every_kind(interface, path, pattern):
    """A kind an interface does not mention is a kind it silently drops."""
    rendered = _literals([path], pattern)
    missing = set(events.KINDS) - rendered
    assert not missing, f"{interface} renders nothing for {sorted(missing)}"


@pytest.mark.parametrize(
    ("interface", "path", "pattern"),
    [
        ("jupyter", JUPYTER, r'kind == "([a-z_]+)"'),
        ("web", WEB_JS, r"kind === '([a-z_]+)'"),
    ],
)
def test_interfaces_render_every_artifact_kind_they_are_meant_to(interface, path, pattern):
    rendered = _literals([path], pattern)
    expected = set(events.ARTIFACT_KINDS) - events.RENDER_OPTIONAL
    if interface == "jupyter":
        # Known gap, tracked in the 0.3.x plan (Q7): the notebook shows cards and
        # figures but neither the script nor the recipe chip yet.
        expected -= {"code", "recipe_used"}
    missing = expected - rendered
    assert not missing, f"{interface} renders nothing for artifact kinds {sorted(missing)}"


def test_cli_renders_only_figures_for_now():
    """The CLI's artifact gap is real and tracked (0.3.x plan, Q7); this pins its size so
    it cannot grow unnoticed and so closing it is visible in the diff."""
    rendered = _literals([CLI], r'kind == "([a-z_]+)"')
    assert rendered == {"image"}


def test_the_sub_agent_end_the_lead_reemits_carries_every_contract_key():
    """The re-emitted event used to drop `findings`; the contract now names it."""
    src = (ROOT / "core" / "agent_loop.py").read_text(encoding="utf-8")
    start = src.index("sub_end_event = {")
    depth, end = 0, src.index("{", start)
    for pos in range(end, len(src)):
        depth += {"{": 1, "}": -1}.get(src[pos], 0)
        if depth == 0:
            end = pos
            break
    block = src[start:end]
    for key in events.KINDS["sub_agent_end"]:
        assert f'"{key}"' in block, f"the lead's sub_agent_end lacks {key!r}"


def test_module_docstrings_defer_to_the_contract():
    """Three docstrings listed the kinds and disagreed; they must point here instead of
    keeping their own copy."""
    for path in (ROOT / "core" / "agent_loop.py", ROOT / "core" / "sub_agents.py"):
        assert "core/events.py" in path.read_text(encoding="utf-8")[:4000], path.name
