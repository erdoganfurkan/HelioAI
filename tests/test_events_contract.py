"""The event contract holds: what the loops emit is what `core/events.py` lists, and
what the interfaces render.

Static, by design. Running both loops through every branch that emits every kind would
need a scripted scenario per kind; grepping the source for the literals the emitters
and renderers use catches the drift this exists for — a kind added to a loop and to two
of the three interfaces, or documented and never emitted — in milliseconds, on every
run.
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
    return _literals(EMITTERS, r'"event":\s*"([a-z_]+)"')


def _emitted_artifact_kinds() -> set[str]:
    # tool_exec builds artifacts; `data.get("kind")` reads are not emissions.
    return _literals([ROOT / "core" / "tool_exec.py"], r'"kind":\s*"([a-z_]+)"')


def test_every_emitted_kind_is_in_the_contract_and_vice_versa():
    assert _emitted_kinds() == set(events.KINDS)


def test_every_emitted_artifact_kind_is_in_the_contract_and_vice_versa():
    assert _emitted_artifact_kinds() == set(events.ARTIFACT_KINDS)


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
