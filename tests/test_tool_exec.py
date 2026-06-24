"""Tests for the shared tool-execution helpers (helioai.core.tool_exec)."""

from __future__ import annotations

import json

from helioai.core.tool_exec import (
    emit_post_tool_events,
    inject_run_python_args,
)


# ──────────────────────────────── inject_run_python_args ────────────────────


def test_inject_only_for_run_python(monkeypatch) -> None:
    import helioai.workspace as ws

    monkeypatch.setattr(ws, "get_session_dir", lambda: __import__("pathlib").Path("/tmp/ws"))
    monkeypatch.setattr(ws, "get_next_run_idx", lambda d: 3)

    # Returns only the trusted args, passed to call_tool(..., trusted=...).
    out = inject_run_python_args("run_python")
    assert out == {"_plot_dir": "/tmp/ws", "_run_idx": 3}


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
