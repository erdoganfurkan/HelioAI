"""Shared tool-execution helpers for the agent loops.

Both the lead loop (agent_loop.stream_chat) and the sub-agent loop
(sub_agents.stream_subagent) run the same tool-call mechanics: inject the
sandbox run dir for run_python, summarise the result, detect skill loads,
and extract renderable artifacts. Keeping that logic here — imported by
both loops — prevents the two copies from drifting apart (a real bug source,
see the session-13 _extract_artifact list/dict regression).

This module imports neither agent_loop nor sub_agents, so there is no cycle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


# Tools whose results contain large lists (per_event_stats, sample rows) that would
# flood the LLM context. All other tools pass through untouched so the LLM can reason
# on their content (search results, mission lists, catalog ids, etc.).
_HEAVY_TOOLS: frozenset[str] = frozenset({"get_events_timeseries", "get_catalog"})


def _history_tool_result(tool_name: str, result_text: str) -> str:
    """Return the string that gets appended to history for a completed tool call.

    Heavy tools (SEA, catalog preview) are summarized to avoid flooding the context.
    All other tools — especially search_parameters — are passed through verbatim so
    the LLM can see the actual candidates and make an informed choice.
    """
    if tool_name in _HEAVY_TOOLS:
        return _summarize_tool_result(result_text, max_chars=600)
    return result_text


def _summarize_tool_result(result_text: str, max_chars: int = 400) -> str:
    try:
        data = json.loads(result_text)
    except (ValueError, TypeError):
        return result_text[:max_chars]
    if not isinstance(data, dict):
        return str(data)[:max_chars] if not isinstance(data, list) else f"[list, {len(data)} items]"
    if data.get("error"):
        return f"error: {str(data['error'])[:max_chars]}"
    keep = {}
    for k, v in data.items():
        if k in {"figure_paths"}:
            keep[k] = [Path(p).name for p in v] if v else []
        elif isinstance(v, (str, int, float, bool, type(None))):
            keep[k] = v if not isinstance(v, str) or len(v) <= 120 else v[:117] + "..."
        elif isinstance(v, list):
            keep[k] = f"[{len(v)} items]"
        elif isinstance(v, dict):
            keep[k] = f"{{{len(v)} keys}}"
    return json.dumps(keep, ensure_ascii=False)[:max_chars]


def _extract_artifact(tool_name: str, result_text: str) -> list[dict]:
    """Extract renderable artifacts from tool results (plots, parameter cards)."""
    try:
        data = json.loads(result_text)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, dict) or "error" in data:
        return []

    artifacts: list[dict] = []

    # Python sandbox: figures + parameter cards emitted via param_card()
    if tool_name == "run_python":
        if data.get("figure_paths"):
            artifacts.append(
                {
                    "tool": tool_name,
                    "kind": "image",
                    "figure_paths": data["figure_paths"],
                    "stdout": data.get("stdout", ""),
                }
            )
        for card in data.get("cards", []):
            if isinstance(card, dict) and card.get("kind") == "parameter_card":
                artifacts.append({"tool": tool_name, **card})
        if data.get("code_path"):
            artifacts.append(
                {
                    "tool": tool_name,
                    "kind": "code",
                    "code_path": data["code_path"],
                    "name": Path(data["code_path"]).name,
                    "n_lines": data.get("n_lines"),
                }
            )

    # get_catalog result
    if tool_name == "get_catalog" and data.get("_kind") == "catalog_preview":
        artifacts.append(
            {
                "tool": tool_name,
                "kind": "catalog_preview",
                "catalog_id": data.get("catalog_id"),
                "name": data.get("name"),
                "type": data.get("type"),
                "nb_events_total": data.get("nb_events_total"),
                "columns": data.get("columns", []),
                "sample": (data.get("sample") or [])[:5],
                "survey_start": data.get("survey_start"),
                "survey_stop": data.get("survey_stop"),
            }
        )

    # get_timeseries called directly by main agent
    if tool_name == "get_timeseries" and "preview" in data:
        artifacts.append(
            {
                "tool": tool_name,
                "kind": "parameter_card",
                "param_id": data.get("param_id"),
                "name": data.get("name"),
                "mission": data.get("mission"),
                "instrument": data.get("instrument"),
                "units": data.get("units"),
                "cadence": data.get("cadence"),
                "components": data.get("components"),
                "n_points": data.get("n_points"),
                "start": data.get("start"),
                "stop": data.get("stop"),
            }
        )

    return artifacts


def inject_run_python_args(name: str, args: dict | None) -> dict:
    """Inject the per-run sandbox output dir into run_python arguments.

    No-op for any other tool. Returns a new dict (does not mutate the input).
    """
    args = args or {}
    if name != "run_python":
        return args
    import helioai.workspace as _ws

    sdir = _ws.get_session_dir()
    ridx = _ws.get_next_run_idx(sdir)
    return {**args, "_plot_dir": str(sdir), "_run_idx": ridx}


def emit_post_tool_events(
    name: str,
    result: str,
    *,
    tool_result_extra: dict | None = None,
    common_extra: dict | None = None,
) -> Iterator[dict]:
    """Yield the events that follow a completed tool call.

    Order is `tool_result` → (`skill_loaded` if load_skill) → `artifact`(s),
    matching what both loops emitted before this was factored out.

    - `tool_result_extra` is merged into the tool_result event data (e.g. {turn}).
    - `common_extra` is merged into skill_loaded and artifact event data
      (e.g. {sub_agent_ctx} for sub-agents).
    """
    tool_result_extra = tool_result_extra or {}
    common_extra = common_extra or {}

    yield {
        "event": "tool_result",
        "data": {
            "name": name,
            "summary": _summarize_tool_result(result),
            **tool_result_extra,
        },
    }

    if name == "load_skill":
        try:
            payload = json.loads(result)
            if payload.get("body") and not payload.get("error"):
                yield {
                    "event": "skill_loaded",
                    "data": {
                        "name": payload.get("name", ""),
                        **common_extra,
                    },
                }
        except (ValueError, TypeError):
            pass

    for art in _extract_artifact(name, result):
        yield {"event": "artifact", "data": {**art, **common_extra}}
