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


@pytest.mark.asyncio
async def test_lead_retries_once_on_an_invented_id(monkeypatch, tmp_path, fake_llm_factory):
    """The fabrication must cost a turn, not earn a footnote.

    HelioBench `n1_mms_fgm` failed 3/3 byte-identically: the lead spliced a speasy
    *path* into an id that exists nowhere, the detector fired, and the loop ended at
    `n_iterations: 2` — shipping the invented id with the correction stapled to it.
    """
    from helioai.config import settings
    from helioai.core import agent_loop
    from helioai.core.session import SessionStore
    from helioai.tools import rag

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(agent_loop, "store", SessionStore(tmp_path / "sessions.db"))

    real = "cda/MMS1_FGM_SRVY_L2/mms1_fgm_b_gsm_srvy_l2"
    bogus = "cda/MMS/MMS1/FGM/SRVY/mms1_fgm_srvy_l2/mms1_fgm_b_gsm_srvy_l2_clean"

    class _Collection:
        def get(self, ids):
            return {"ids": [i for i in ids if i == real]}

    monkeypatch.setattr(rag, "_collection_only", lambda: _Collection())

    llm = fake_llm_factory(
        [
            Message(role="assistant", content=f"Use {bogus}."),
            Message(role="assistant", content=f"Use {real}."),
        ]
    )

    events = await _collect(agent_loop.stream_chat(llm, "u", "s", "MMS1 FGM survey B in GSM"))

    assert len(llm.calls) == 2, "the correction must be spent on another turn"
    correction = llm.calls[1]["messages"][-1]
    assert correction.role == "user" and bogus in correction.content

    assert not [e for e in events if e["event"] == "invalid_ids"]
    reply = [e for e in events if e["event"] == "reply"][-1]["data"]["text"]
    assert real in reply and "AUTOMATED CORRECTION" not in reply


@pytest.mark.asyncio
async def test_lead_gives_up_after_one_retry(monkeypatch, tmp_path, fake_llm_factory):
    """One retry, not a loop — a model that repeats itself still gets contradicted."""
    from helioai.config import settings
    from helioai.core import agent_loop
    from helioai.core.session import SessionStore
    from helioai.tools import rag

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(agent_loop, "store", SessionStore(tmp_path / "sessions.db"))

    bogus = "cda/MMS1_FGM_SRVY_L2/mms1_fgm_b_gsm_srvy_l2_clean"
    monkeypatch.setattr(
        rag, "_collection_only", lambda: type("C", (), {"get": lambda s, ids: {"ids": []}})()
    )

    llm = fake_llm_factory([Message(role="assistant", content=f"Use {bogus}.")] * 2)

    events = await _collect(agent_loop.stream_chat(llm, "u", "s", "MMS1 FGM survey B in GSM"))

    assert len(llm.calls) == 2, "exactly one retry, then the answer ships annotated"
    flagged = [e for e in events if e["event"] == "invalid_ids"]
    assert flagged and flagged[0]["data"]["ids"] == [bogus]
    assert (
        "AUTOMATED CORRECTION" in [e for e in events if e["event"] == "reply"][-1]["data"]["text"]
    )
