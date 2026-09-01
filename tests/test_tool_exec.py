"""Tests for the shared tool-execution helpers (helioai.core.tool_exec)."""

from __future__ import annotations

import json
from pathlib import Path

import helioai.core.tool_exec as te
from helioai.core.llm.base import Message
from helioai.core.tool_exec import (
    _extract_artifact,
    _summarize_tool_result,
    compact_history,
    emit_post_tool_events,
    inject_run_python_args,
)

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
    events = list(emit_post_tool_events("search_parameters", result, tool_result_extra={"turn": 2}))
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
    events = list(emit_post_tool_events("run_python", result, tool_result_extra={"turn": 1}))
    artifacts = [e for e in events if e["event"] == "artifact"]
    assert len(artifacts) == 1
    assert artifacts[0]["data"]["kind"] == "image"
    assert artifacts[0]["data"]["figure_paths"] == ["/tmp/ws/fig_0_0.png"]


def test_emit_run_python_exports_artifact() -> None:
    # The sandbox can compute everything and print nothing; without this artifact the
    # values never leave run_python's own result and the lead has no numbers to quote.
    exports = {"compression_ratio_density": {"shape": [], "mean": 2.37, "sample": [2.37]}}
    result = json.dumps({"stdout": "", "figure_paths": [], "exports": exports})
    events = list(emit_post_tool_events("run_python", result, tool_result_extra={"turn": 1}))
    artifacts = [e["data"] for e in events if e["event"] == "artifact"]
    assert [a["kind"] for a in artifacts] == ["exports"]
    assert artifacts[0]["values"] == exports


def test_emit_common_extra_on_artifact() -> None:
    result = json.dumps({"figure_paths": ["/tmp/ws/fig_0_0.png"], "stdout": ""})
    ctx = {"role": "data_analyst", "task_id": "abc123"}
    events = list(
        emit_post_tool_events(
            "run_python",
            result,
            tool_result_extra={"turn": 1, "sub_agent_ctx": ctx},
            common_extra={"sub_agent_ctx": ctx},
        )
    )
    artifact = next(e for e in events if e["event"] == "artifact")
    assert artifact["data"]["sub_agent_ctx"] == ctx


def test_emit_skill_loaded_for_load_skill() -> None:
    result = json.dumps({"name": "plotting", "body": "# procedure"})
    events = list(emit_post_tool_events("load_skill", result, tool_result_extra={"turn": 1}))
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
    events = list(emit_post_tool_events("load_recipe", result, tool_result_extra={"turn": 1}))
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
    events = list(emit_post_tool_events("run_python", result, tool_result_extra={"turn": 1}))
    recipes = [
        e for e in events if e["event"] == "artifact" and e["data"].get("kind") == "recipe_used"
    ]
    assert len(recipes) == 1
    assert recipes[0]["data"]["name"] == "MVAB"
    assert recipes[0]["data"]["reference"] == "Sonnerup & Scheible 1998"


def test_emit_no_skill_loaded_on_error() -> None:
    result = json.dumps({"error": "no such skill"})
    events = list(emit_post_tool_events("load_skill", result, tool_result_extra={"turn": 1}))
    assert not [e for e in events if e["event"] == "skill_loaded"]


def test_emit_event_order() -> None:
    result = json.dumps({"name": "plotting", "body": "x", "figure_paths": ["/tmp/f.png"]})
    events = [
        e["event"]
        for e in emit_post_tool_events("load_skill", result, tool_result_extra={"turn": 1})
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
    payload = json.dumps(
        {"error": "ZeroDivisionError: division by zero", "code_path": "/w/code_1.py", "n_lines": 12}
    )
    arts = _extract_artifact("run_python", payload)
    assert [a["kind"] for a in arts] == ["code"]
    assert arts[0]["failed"] is True
    assert arts[0]["name"] == "code_1.py"


def test_other_tools_emit_nothing_on_error():
    arts = _extract_artifact("get_timeseries", json.dumps({"error": "no data"}))
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
