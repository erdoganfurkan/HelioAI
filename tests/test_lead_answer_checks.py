"""The lead loop's own answers get the checks that only sub-agents used to get.

Both `_flag_unknown_ids` and `_flag_recipe_bypass` were written inside the sub-agent
loop and stayed there. On the showcase notebook run of 2026-08-12 the lead did Acts III
and IV itself — Rankine-Hugoniot and two-spacecraft timing, both with a calibrated
recipe on the shelf — and `load_recipe` was called zero times in the whole session
without a word from either detector. They were not silent because the run was clean.
Nothing called them.
"""

from __future__ import annotations

import json

import pytest

from helioai.core.llm.base import Message, ToolCall


def _exports_result(session_dir, exports):
    return json.dumps(
        {
            "stdout": "",
            "figure_paths": [],
            "exports": exports,
            "cards": [],
            "code_path": str(session_dir / "code_0.py"),
            "n_lines": 12,
        }
    )


async def _collect(gen):
    return [ev async for ev in gen]


@pytest.mark.asyncio
async def test_lead_flags_a_recipe_it_never_loaded(monkeypatch, tmp_path, fake_llm_factory):
    from helioai.config import settings
    from helioai.core import agent_loop
    from helioai.core.session import SessionStore

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(agent_loop, "store", SessionStore(tmp_path / "sessions.db"))

    exports = {
        "compression_ratio_density": {"mean": 2.46, "min": 2.46, "max": 2.46, "units": ""},
        "alfven_mach_number": {"mean": 3.05, "min": 3.05, "max": 3.05, "units": ""},
    }

    async def fake_call_tool(name, arguments, trusted=None):
        return _exports_result(tmp_path, exports)

    monkeypatch.setattr(agent_loop.registry, "call_tool", fake_call_tool)

    llm = fake_llm_factory(
        [
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="c1", name="run_python", arguments={"code": "..."})],
            ),
            Message(role="assistant", content="Compression ratio 2.46, M_A 3.05."),
        ]
    )

    events = await _collect(agent_loop.stream_chat(llm, "u", "s", "do the shock physics"))

    flagged = [e for e in events if e["event"] == "recipe_bypassed"]
    assert flagged, "the lead computed a Rankine-Hugoniot without ever loading the recipe"
    assert flagged[0]["data"]["recipes"] == [{"recipe": "rankine_hugoniot", "reason": "not_loaded"}]

    reply = next(e for e in events if e["event"] == "reply")
    assert "RECIPE CHECK" in reply["data"]["text"]


@pytest.mark.asyncio
async def test_lead_says_nothing_when_it_exported_nothing(monkeypatch, tmp_path, fake_llm_factory):
    """A search or a catalogue listing must not be accused of skipping a recipe."""
    from helioai.config import settings
    from helioai.core import agent_loop
    from helioai.core.session import SessionStore

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(agent_loop, "store", SessionStore(tmp_path / "sessions.db"))

    llm = fake_llm_factory([Message(role="assistant", content="Wind flies at L1.")])
    events = await _collect(agent_loop.stream_chat(llm, "u", "s", "where is Wind"))

    assert not [e for e in events if e["event"] == "recipe_bypassed"]
    assert next(e for e in events if e["event"] == "reply")["data"]["text"] == "Wind flies at L1."
