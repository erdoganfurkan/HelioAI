"""Tests for the shared tool-execution helpers (helioai.core.tool_exec)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import helioai.core.tool_exec as te
from helioai.core.llm.base import Message
from helioai.core.tool_exec import (
    _extract_artifact,
    _summarize_tool_result,
    compact_history,
    emit_post_tool_events,
    inject_run_python_args,
)
from helioai.tools.results import ToolResult

# ──────────────────────────────── inject_run_python_args ────────────────────


def test_inject_only_for_run_python(monkeypatch) -> None:
    import helioai.workspace as ws

    # str(Path(...)) is separator-dependent, so the expected value has to be built the
    # same way — spelling it "/tmp/ws" failed on Windows against "\tmp\ws".
    ws_dir = Path("/tmp/ws")
    monkeypatch.setattr(ws, "get_session_dir", lambda: ws_dir)
    monkeypatch.setattr(ws, "get_next_run_idx", lambda d: 3)

    # Returns only the trusted args, passed to call_tool(..., trusted=...).
    out = inject_run_python_args("run_python")
    assert out == {"_plot_dir": str(ws_dir), "_run_idx": 3}


def test_inject_noop_for_other_tools() -> None:
    assert inject_run_python_args("get_timeseries") == {}


# ──────────────────────────────── emit_post_tool_events ─────────────────────


def test_emit_tool_result_carries_extra() -> None:
    result = json.dumps({"results": [1, 2, 3]})
    events = list(
        emit_post_tool_events(
            "search_parameters",
            ToolResult.from_raw("search_parameters", result),
            tool_result_extra={"turn": 2},
        )
    )
    assert events[0]["event"] == "tool_result"
    assert events[0]["data"]["turn"] == 2
    assert events[0]["data"]["name"] == "search_parameters"


def test_emit_run_python_image_artifact() -> None:
    result = json.dumps(
        {
            "stdout": "ok",
            "figure_paths": ["/tmp/ws/fig_0_0.png"],
            "exports": {},
        }
    )
    events = list(
        emit_post_tool_events(
            "run_python", ToolResult.from_raw("run_python", result), tool_result_extra={"turn": 1}
        )
    )
    artifacts = [e for e in events if e["event"] == "artifact"]
    assert len(artifacts) == 1
    assert artifacts[0]["data"]["kind"] == "image"
    assert artifacts[0]["data"]["figure_paths"] == ["/tmp/ws/fig_0_0.png"]


def test_emit_run_python_exports_artifact() -> None:
    # The sandbox can compute everything and print nothing; without this artifact the
    # values never leave run_python's own result and the lead has no numbers to quote.
    exports = {"compression_ratio_density": {"shape": [], "mean": 2.37, "sample": [2.37]}}
    result = json.dumps({"stdout": "", "figure_paths": [], "exports": exports})
    events = list(
        emit_post_tool_events(
            "run_python", ToolResult.from_raw("run_python", result), tool_result_extra={"turn": 1}
        )
    )
    artifacts = [e["data"] for e in events if e["event"] == "artifact"]
    assert [a["kind"] for a in artifacts] == ["exports"]
    assert artifacts[0]["values"] == exports


def test_emit_common_extra_on_artifact() -> None:
    result = json.dumps({"figure_paths": ["/tmp/ws/fig_0_0.png"], "stdout": ""})
    ctx = {"role": "data_analyst", "task_id": "abc123"}
    events = list(
        emit_post_tool_events(
            "run_python",
            ToolResult.from_raw("run_python", result),
            tool_result_extra={"turn": 1, "sub_agent_ctx": ctx},
            common_extra={"sub_agent_ctx": ctx},
        )
    )
    artifact = next(e for e in events if e["event"] == "artifact")
    assert artifact["data"]["sub_agent_ctx"] == ctx


def test_emit_skill_loaded_for_load_skill() -> None:
    result = json.dumps({"name": "plotting", "body": "# procedure"})
    events = list(
        emit_post_tool_events(
            "load_skill", ToolResult.from_raw("load_skill", result), tool_result_extra={"turn": 1}
        )
    )
    skill_events = [e for e in events if e["event"] == "skill_loaded"]
    assert len(skill_events) == 1
    assert skill_events[0]["data"]["name"] == "plotting"


def test_emit_recipe_used_artifact() -> None:
    result = json.dumps(
        {
            "name": "theta_bn",
            "code": "def theta_bn(...): ...",
            "metadata": {"reference": "Schwartz 1998", "description": "Shock normal angle."},
        }
    )
    events = list(
        emit_post_tool_events(
            "load_recipe", ToolResult.from_raw("load_recipe", result), tool_result_extra={"turn": 1}
        )
    )
    artifacts = [e for e in events if e["event"] == "artifact"]
    assert len(artifacts) == 1
    assert artifacts[0]["data"]["kind"] == "recipe_used"
    assert artifacts[0]["data"]["name"] == "theta_bn"
    assert artifacts[0]["data"]["reference"] == "Schwartz 1998"


def test_emit_method_used_card_becomes_recipe_artifact() -> None:
    result = json.dumps(
        {
            "stdout": "",
            "figure_paths": [],
            "cards": [
                {
                    "kind": "method_used",
                    "name": "MVAB",
                    "reference": "Sonnerup & Scheible 1998",
                    "method": "minimum variance analysis",
                }
            ],
        }
    )
    events = list(
        emit_post_tool_events(
            "run_python", ToolResult.from_raw("run_python", result), tool_result_extra={"turn": 1}
        )
    )
    recipes = [
        e for e in events if e["event"] == "artifact" and e["data"].get("kind") == "recipe_used"
    ]
    assert len(recipes) == 1
    assert recipes[0]["data"]["name"] == "MVAB"
    assert recipes[0]["data"]["reference"] == "Sonnerup & Scheible 1998"


def test_emit_no_skill_loaded_on_error() -> None:
    result = json.dumps({"error": "no such skill"})
    events = list(
        emit_post_tool_events(
            "load_skill", ToolResult.from_raw("load_skill", result), tool_result_extra={"turn": 1}
        )
    )
    assert not [e for e in events if e["event"] == "skill_loaded"]


def test_emit_event_order() -> None:
    result = json.dumps({"name": "plotting", "body": "x", "figure_paths": ["/tmp/f.png"]})
    events = [
        e["event"]
        for e in emit_post_tool_events(
            "load_skill", ToolResult.from_raw("load_skill", result), tool_result_extra={"turn": 1}
        )
    ]
    assert events[0] == "tool_result"
    assert events.index("tool_result") < events.index("skill_loaded")


# ──────────────────────────────── compact_history ──────────────────────────


def test_compact_history_keeps_recent_summarizes_old() -> None:
    long = json.dumps({"results": [{"id": f"p{i}", "description": "x" * 200} for i in range(5)]})
    history = [
        Message(role="user", content="hi"),
        Message(role="tool", tool_call_id="1", content=long),
        Message(role="assistant", content="ok"),
        Message(role="tool", tool_call_id="2", content=long),
        Message(role="tool", tool_call_id="3", content=long),
    ]
    out = compact_history(history, keep_full=2)
    assert history[1].content == long  # original list untouched
    assert len(out[1].content) < len(long)  # oldest tool result summarized
    assert out[3].content == long and out[4].content == long  # two most recent kept verbatim
    assert out[0].content == "hi"  # non-tool messages untouched


def test_compact_history_noop_when_few_tools() -> None:
    history = [Message(role="tool", tool_call_id="1", content="a" * 500)]
    assert compact_history(history, keep_full=2) is history


def test_a_loaded_recipe_survives_compaction():
    """Live runs 3 and 4 of 00_quickstart, same sequence both times: load_recipe, two
    run_python calls, load_recipe again. After two more tool results the recipe source
    had been compacted to its first 117 characters, the analyst no longer had the
    function, reloaded it — and in run 4 rewrote the formula from memory instead of
    calling it. A recipe is a few kilobytes and is the one tool result the next
    `run_python` is written against; it must stay verbatim.
    """
    recipe = json.dumps(
        {
            "name": "theta_bn",
            "code": "# name: theta_bn\n# description: Compute the shock normal angle\n"
            + "def theta_bn(B_up, B_dn):\n    ...\n" * 40,
            "metadata": {"name": "theta_bn", "description": "Compute the shock normal angle"},
        }
    )
    run = json.dumps({"stdout": "n_points 1200", "figure_paths": [], "exports": {}})
    history = [
        Message(role="user", content="compute theta_Bn"),
        Message(role="tool", tool_call_id="1", content=recipe),
        Message(role="assistant", content="probing the data first"),
        Message(role="tool", tool_call_id="2", content=run),
        Message(role="tool", tool_call_id="3", content=run),
    ]
    out = compact_history(history, keep_full=2)
    assert out[1].content == recipe


def test_a_recipe_is_recognised_by_its_tool_name_first():
    """Tool messages now carry the tool's name; a recipe is `name == "load_recipe"`,
    whatever its payload looks like. Histories persisted before the field existed have
    no name and still fall back to the shape (the test above)."""
    named_recipe = Message(role="tool", tool_call_id="1", name="load_recipe", content="{}")
    named_other = Message(
        role="tool",
        tool_call_id="2",
        name="run_python",
        content=json.dumps({"name": "x", "code": "y", "metadata": {}}),
    )
    assert te._is_recipe_source(named_recipe)
    assert not te._is_recipe_source(named_other)


def test_both_loops_record_the_tool_name_on_history_messages(monkeypatch, tmp_path):
    import asyncio

    from helioai.core import agent_loop
    from helioai.core.llm.base import ToolCall
    from helioai.core.session import SessionStore

    store = SessionStore(tmp_path / "sessions.db")
    monkeypatch.setattr(agent_loop, "store", store)
    responses = [
        Message(
            role="assistant", tool_calls=[ToolCall(id="c1", name="list_recipes", arguments={})]
        ),
        Message(role="assistant", content="done"),
    ]

    class _LLM:
        async def chat(self, messages, tools, **k):
            return responses.pop(0)

    async def run():
        async for _ in agent_loop.stream_chat(_LLM(), "web", "s1", "go", restricted=False):
            pass

    asyncio.run(run())
    tool_messages = [m for m in store.get_or_create("web", "s1") if m.role == "tool"]
    assert [m.name for m in tool_messages] == ["list_recipes"]
    assert (
        SessionStore(tmp_path / "sessions.db").get_or_create("web", "s1")[2].name == "list_recipes"
    )


def test_compaction_keeps_the_traceback_of_a_failed_run():
    """Losing stderr two turns later is why one typo was retried three times."""
    payload = json.dumps(
        {
            "error": "NameError: name 'nai' is not defined",
            "stdout": "",
            "stderr": "Traceback:\n  File \"your code\", line 25\nNameError: name 'nai' is not defined",
            "code_path": "/w/code_3.py",
            "n_lines": 40,
        }
    )
    summary = _summarize_tool_result(payload, max_chars=300)
    assert "NameError" in summary
    assert "line 25" in summary, summary


def test_failed_run_python_still_yields_the_code_artifact():
    payload = {
        "error": "ZeroDivisionError: division by zero",
        "code_path": "/w/code_1.py",
        "n_lines": 12,
    }
    arts = _extract_artifact("run_python", payload)
    assert [a["kind"] for a in arts] == ["code"]
    assert arts[0]["failed"] is True
    assert arts[0]["name"] == "code_1.py"


def test_other_tools_emit_nothing_on_error():
    arts = _extract_artifact("get_timeseries", {"error": "no data"})
    assert arts == []


def test_inject_run_python_args_no_network() -> None:
    from helioai.core.tool_exec import inject_run_python_args

    args_default = inject_run_python_args("run_python")
    assert "_no_net" not in args_default

    args_no_net = inject_run_python_args("run_python", no_network=True)
    assert args_no_net.get("_no_net") is True

    assert inject_run_python_args("other_tool", no_network=True) == {}


# ── host paths must not reach the model ────────────────────────────────────────


def test_history_result_hides_the_home_directory(monkeypatch, tmp_path):
    """The model cannot open a host path, so handing it one only teaches it to quote it.

    It surfaced on a published screen recording: the answer read
    "saved to /home/<user>/HelioAI/data/users/web/workspace/...".
    """
    monkeypatch.setattr(te.Path, "home", classmethod(lambda cls: tmp_path))
    result = json.dumps(
        {
            "figure_paths": [f"{tmp_path}/HelioAI/data/users/web/workspace/s1/fig_0_0.png"],
            "code_path": f"{tmp_path}/HelioAI/data/users/web/workspace/s1/code_0.py",
            "n_lines": 32,
        }
    )
    out = te._history_tool_result("run_python", result)
    assert str(tmp_path) not in out
    # Still identifiable: the model must be able to name the file it just made.
    assert "fig_0_0.png" in out
    assert "code_0.py" in out
    assert "32" in out


def test_redaction_leaves_paths_outside_home_alone(monkeypatch, tmp_path):
    """In a container the data dir is not under a home; nothing to hide there."""
    monkeypatch.setattr(te.Path, "home", classmethod(lambda cls: tmp_path))
    result = json.dumps({"figure_paths": ["/app/data/workspace/s1/fig.png"]})
    assert "/app/data/workspace/s1/fig.png" in te._history_tool_result("run_python", result)


def test_redaction_does_not_touch_science_text(monkeypatch, tmp_path):
    monkeypatch.setattr(te.Path, "home", classmethod(lambda cls: tmp_path))
    result = json.dumps({"stdout": "compression ratio 2.59, theta_Bn 47.3 deg"})
    out = te._history_tool_result("run_python", result)
    assert "2.59" in out
    assert "47.3" in out


# ──────────────────── compaction: what a stale result must still say ────────


def _stat(v, units=""):
    return {
        "units": units,
        "shape": [],
        "dtype": "float64",
        "min": v,
        "max": v,
        "mean": v,
        "std": 0.0,
        "n_finite": 1,
        "n_nan": 0,
        "sample": [v],
    }


def test_a_stale_run_python_still_holds_every_exported_number():
    """Run 4 of 00_quickstart: the analyst exported eleven values, probed the data twice
    more, and by then its own result read `"exports": "{11 keys}"`. Two turns later it
    was writing the answer from the two probes alone. The exports are the numbers the
    run exists to produce; they are kept as `name: value units`, like findings are.
    """
    payload = json.dumps(
        {
            "stdout": "theta_bn: 54.85 deg\n" + "diagnostic line\n" * 40,
            "figure_paths": ["/w/fig_3_0.png"],
            "exports": {
                "theta_bn_deg": _stat(54.85, "deg"),
                "Bmag_up_nT": _stat(9.79, "nT"),
                "Bmag_dn_nT": _stat(25.17, "nT"),
                "compression_ratio": _stat(2.57),
                "normal_Bx_GSM": _stat(-0.509),
                "normal_By_GSM": _stat(-0.753),
                "normal_Bz_GSM": _stat(0.417),
                "shock_time_ut": {"error": "could not convert string to float", "repr": "'04:00'"},
            },
            "cards": [],
            "code_path": "/w/code_3.py",
            "n_lines": 86,
        }
    )
    summary = _summarize_tool_result(payload, max_chars=1500)
    data = json.loads(summary)
    assert data["exports"]["theta_bn_deg"] == "54.85 deg"
    assert data["exports"]["compression_ratio"] == "2.57"
    assert data["exports"]["normal_Bz_GSM"] == "0.417"
    assert "shock_time_ut" not in data["exports"]
    assert data["stdout"].startswith("theta_bn: 54.85 deg")


def test_findings_are_never_cut_by_the_cap():
    """Run 4, one-shot: the analyst's report was 9 485 characters and the 300-character
    cap fell in the middle of the findings dict — the JSON was truncated after
    `"Bmag_up_nT": "9.79090`, and the compression ratio was gone. The code said the
    findings table must not degrade; the final slice degraded it anyway.
    """
    findings = {f"quantity_{i:02d}": {"value": float(i), "units": "nT"} for i in range(40)}
    payload = json.dumps(
        {
            "findings": findings,
            "summary": "x" * 5000,
            "n_iterations": 7,
            "artifacts": [],
            "error": None,
        }
    )
    summary = _summarize_tool_result(payload, max_chars=300)
    data = json.loads(summary)
    assert len(data["findings"]) == 40
    assert data["findings"]["quantity_39"] == "39.0 nT"


def test_a_sub_agent_report_keeps_enough_of_its_summary_to_be_a_memory():
    """The lead's only account of what a sub-agent did is its summary. At 117
    characters — "Done. **Dataset key:** `b3gsm` (parameter `cda/WI_H0_MFI/B3GSM`, Wind
    MFI 3-second GSM magnetic field vector, 2015-..." — the method, the windows and the
    caveats were all gone by the next cell of the notebook.
    """
    report = "Used windows 03:48–03:56 and 04:04–04:12; theta_bn() from the recipe. " * 30
    payload = json.dumps(
        {"findings": {}, "summary": report, "n_iterations": 5, "artifacts": [], "error": None}
    )
    summary = _summarize_tool_result(payload, max_chars=1500)
    data = json.loads(summary)
    assert len(data["summary"]) >= 900
    assert len(summary) <= 1500


def test_compaction_shrinks_text_to_fit_rather_than_cutting_the_json():
    payload = json.dumps(
        {
            "findings": {"Bd": {"value": 14.5, "units": "nT"}},
            "summary": "word " * 400,
            "error": None,
        }
    )
    summary = _summarize_tool_result(payload, max_chars=300)
    data = json.loads(summary)
    assert data["findings"]["Bd"] == "14.5 nT"
    assert len(summary) <= 300


# ── a turn's tool calls overlap, and their results keep the model's order ──────


@pytest.fixture
def three_slow_tools(monkeypatch):
    """Three registry tools that each take 0.2 s and record when they ran."""
    import asyncio
    import time

    from helioai.tools.registry import Tool, registry

    log: list[tuple[str, float]] = []

    def make(name):
        async def tool(**kwargs):
            log.append((name, time.monotonic()))
            await asyncio.sleep(0.2)
            return {"tool": name}

        return tool

    for name in ("slow_a", "slow_b", "slow_c"):
        monkeypatch.setitem(
            registry._tools,
            name,
            Tool(name=name, description="slow", parameters={"type": "object"}, func=make(name)),
        )
    return log


async def test_start_tool_calls_overlaps_registry_tools(three_slow_tools):
    import asyncio
    import time

    from helioai.core.llm.base import ToolCall
    from helioai.core.tool_exec import start_tool_calls

    calls = [
        ToolCall(id=f"c{i}", name=n, arguments={})
        for i, n in enumerate(("slow_a", "slow_b", "slow_c"))
    ]
    t0 = time.monotonic()
    started = start_tool_calls(calls)
    results = [await started[tc.id] for tc in calls]
    elapsed = time.monotonic() - t0

    assert [r.for_llm() for r in results] == [
        '{"tool": "slow_a"}',
        '{"tool": "slow_b"}',
        '{"tool": "slow_c"}',
    ]
    assert elapsed < 0.45, f"three 0.2 s tools took {elapsed:.2f} s — they ran one after another"
    await asyncio.sleep(0)


def test_start_tool_calls_leaves_the_sequential_and_unknown_calls_alone():
    import asyncio

    import helioai.tools.setup  # noqa: F401 — the registry is empty until this import
    from helioai.core.llm.base import ToolCall
    from helioai.core.tool_exec import start_tool_calls

    async def run():
        calls = [
            ToolCall(id="a", name="run_python", arguments={"code": "1"}),
            ToolCall(id="b", name="task", arguments={}),
            ToolCall(id="c", name="present_plan", arguments={}),
            ToolCall(id="d", name="no_such_tool", arguments={}),
            ToolCall(id="e", name="list_recipes", arguments={}),
        ]
        started = start_tool_calls(calls)
        assert set(started) == {"e"}
        await started["e"]

    asyncio.run(run())


def test_start_tool_calls_respects_a_sub_agents_whitelist():
    import asyncio

    import helioai.tools.setup  # noqa: F401
    from helioai.core.llm.base import ToolCall
    from helioai.core.tool_exec import start_tool_calls

    async def run():
        calls = [
            ToolCall(id="a", name="list_recipes", arguments={}),
            ToolCall(id="b", name="list_missions", arguments={}),
        ]
        started = start_tool_calls(calls, allowed={"list_recipes"})
        assert set(started) == {"a"}
        await started["a"]

    asyncio.run(run())


async def test_the_lead_keeps_event_order_when_calls_overlap(
    three_slow_tools, monkeypatch, tmp_path
):
    """Overlap must be invisible to the interfaces: tool_call/tool_result pairs and the
    `tool` messages stay in the order the model issued the calls."""
    from helioai.core import agent_loop
    from helioai.core.llm.base import Message, ToolCall
    from helioai.core.session import SessionStore

    monkeypatch.setattr(agent_loop, "store", SessionStore(tmp_path / "sessions.db"))
    responses = [
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="c1", name="slow_c", arguments={}),
                ToolCall(id="c2", name="slow_a", arguments={}),
                ToolCall(id="c3", name="slow_b", arguments={}),
            ],
        ),
        Message(role="assistant", content="done"),
    ]

    class _LLM:
        async def chat(self, messages, tools, **k):
            return responses.pop(0)

    events = [
        ev async for ev in agent_loop.stream_chat(_LLM(), "web", "s1", "go", restricted=False)
    ]
    names = [
        (e["event"], e["data"]["name"])
        for e in events
        if e["event"] in ("tool_call", "tool_result")
    ]
    assert names == [
        ("tool_call", "slow_c"),
        ("tool_result", "slow_c"),
        ("tool_call", "slow_a"),
        ("tool_result", "slow_a"),
        ("tool_call", "slow_b"),
        ("tool_result", "slow_b"),
    ]
    history = agent_loop.store.get_or_create("web", "s1")
    assert [m.tool_call_id for m in history if m.role == "tool"] == ["c1", "c2", "c3"]
