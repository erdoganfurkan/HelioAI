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
import re
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

from helioai import provenance

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


def _finding_str(entry) -> str:
    if not isinstance(entry, dict):
        return str(entry)[:60]
    out = f"{entry.get('value')} {entry.get('units') or ''}".strip()
    if entry.get("min") is not None:
        out += f" [{entry['min']}, {entry['max']}]"
    return out


def _summarize_tool_result(result_text: str, max_chars: int = 400) -> str:
    try:
        data = json.loads(result_text)
    except (ValueError, TypeError):
        return result_text[:max_chars]
    if not isinstance(data, dict):
        return str(data)[:max_chars] if not isinstance(data, list) else f"[list, {len(data)} items]"
    if data.get("error"):
        # The traceback has to survive compaction. Collapsing a failed run_python to its
        # `error` line alone meant the agent lost the diagnosis two tool calls later and
        # reproduced the same one-character typo three times in a single session.
        summary = f"error: {str(data['error'])[:max_chars]}"
        stderr = str(data.get("stderr") or "").strip()
        if stderr:
            summary += f"\n{stderr[-max_chars:]}"
        if isinstance(data.get("findings"), dict) and data["findings"]:
            # A run that hit its turn cap still measured things on the way there, and
            # dropping them here is how a lead ends up with nothing to report but prose.
            measured = ", ".join(f"{n}={_finding_str(d)}" for n, d in data["findings"].items())
            summary += f"\nmeasured before failing: {measured}"
        return summary
    keep = {}
    for k, v in data.items():
        if k == "findings" and isinstance(v, dict):
            # The one table that must not degrade into "{3 keys}": it holds the only
            # numbers the lead is allowed to publish as measurements.
            keep[k] = {n: _finding_str(d) for n, d in v.items()}
        elif k in {"figure_paths"}:
            keep[k] = [Path(p).name for p in v] if v else []
        elif isinstance(v, (str, int, float, bool, type(None))):
            keep[k] = v if not isinstance(v, str) or len(v) <= 120 else v[:117] + "..."
        elif isinstance(v, list):
            keep[k] = f"[{len(v)} items]"
        elif isinstance(v, dict):
            keep[k] = f"{{{len(v)} keys}}"
    return json.dumps(keep, ensure_ascii=False)[:max_chars]


def compact_history(messages: list, keep_full: int = 2) -> list:
    """Return a copy of `messages` where tool-result messages older than the last
    `keep_full` are summarized. The most recent results stay verbatim (the next LLM
    call usually needs them); older ones — already consumed — are trimmed so context
    does not grow unbounded over a long session. Persisted history is left untouched;
    only the per-call payload shrinks.

    ponytail: fixed window N=2; widen keep_full if a case regresses on stale results.
    """
    tool_idx = [i for i, m in enumerate(messages) if getattr(m, "role", None) == "tool"]
    if len(tool_idx) <= keep_full:
        return messages
    stale = set(tool_idx[:-keep_full])
    return [
        replace(m, content=_summarize_tool_result(m.content, max_chars=300))
        if i in stale and m.content
        else m
        for i, m in enumerate(messages)
    ]


def _extract_artifact(tool_name: str, result_text: str) -> list[dict]:
    """Extract renderable artifacts from tool results (plots, parameter cards)."""
    try:
        data = json.loads(result_text)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    if "error" in data:
        # A failed run leaves the offending script on disk. Surfacing it is the whole
        # point when something broke — hiding it left the user watching a bare
        # "exited with code 1" with no way to see what ran.
        if tool_name == "run_python" and data.get("code_path"):
            return [
                {
                    "tool": tool_name,
                    "kind": "code",
                    "code_path": data["code_path"],
                    "name": Path(data["code_path"]).name,
                    "n_lines": data.get("n_lines"),
                    "failed": True,
                }
            ]
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
        if data.get("exports"):
            artifacts.append(
                {
                    "tool": tool_name,
                    "kind": "exports",
                    "values": data["exports"],
                    "code_path": data.get("code_path"),
                }
            )
        for card in data.get("cards", []):
            if not isinstance(card, dict):
                continue
            if card.get("kind") == "parameter_card":
                artifacts.append({"tool": tool_name, **card})
            elif card.get("kind") == "method_used":
                artifacts.append(
                    {
                        "tool": tool_name,
                        "kind": "recipe_used",
                        "name": card.get("name", ""),
                        "reference": card.get("reference", ""),
                        "description": card.get("method", ""),
                    }
                )
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

    # load_recipe: surface the recipe + its scientific reference for provenance
    if tool_name == "load_recipe" and data.get("name"):
        meta = data.get("metadata") or {}
        artifacts.append(
            {
                "tool": tool_name,
                "kind": "recipe_used",
                "name": data["name"],
                "reference": meta.get("reference", ""),
                "description": meta.get("description", ""),
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
        card = {
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
        if (data.get("quality") or {}).get("notable"):
            card["quality"] = data["quality"]
        artifacts.append(card)

    return artifacts


def _code_path(result_text: str) -> str:
    try:
        data = json.loads(result_text)
        return data.get("code_path") or "" if isinstance(data, dict) else ""
    except (ValueError, TypeError):
        return ""


def inject_run_python_args(name: str) -> dict:
    """Trusted per-run sandbox args (_plot_dir/_run_idx) for run_python.

    Passed via `call_tool(..., trusted=...)` so they bypass the private-arg
    guard that rejects LLM/MCP-supplied `_*` overrides. Empty for any other tool.
    """
    if name != "run_python":
        return {}
    import helioai.workspace as _ws

    sdir = _ws.get_session_dir()
    ridx = _ws.get_next_run_idx(sdir)
    return {"_plot_dir": str(sdir), "_run_idx": ridx}


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
        if art.get("kind") == "exports":
            ctx = common_extra.get("sub_agent_ctx") or {}
            provenance.record(
                art.get("values") or {},
                code_path=_code_path(result),
                agent=ctx.get("role") or "lead",
                task_id=ctx.get("task_id"),
                turn=tool_result_extra.get("turn"),
            )
        yield {"event": "artifact", "data": {**art, **common_extra}}


def _flag_unknown_ids(text: str) -> tuple[str, list[str]]:
    """Append a correction when an answer quotes parameter ids that do not exist.

    Prompting alone does not stop this. Asked whether to use Cluster onboard
    moments or prime parameters, a `parameter_hunter` run searched correctly and
    then spliced two real products into one id that exists in neither:
    `csa/C3_PP_CIS/...` grafted the CDAWeb prime-parameter dataset onto the CSA
    onboard-moments path. A confidently wrong id is the most damaging possible
    answer, so it is checked against the index rather than trusted.

    The wrong ids are left in place and contradicted, not silently rewritten:
    the reader needs to see that the answer was unreliable, and guessing a
    replacement would repeat the original mistake.

    Args:
        text: The finished answer.

    Returns:
        The text (with a correction appended when needed) and the unknown ids.
    """
    from helioai.tools.rag import extract_ids, unknown_ids

    bogus = unknown_ids(extract_ids(text))
    if not bogus:
        return text, []
    listed = "\n".join(f"  - {i}" for i in bogus)
    return (
        f"{text}\n\n"
        f"⚠️ AUTOMATED CORRECTION — the following ids are NOT in the catalogue and "
        f"must not be used:\n{listed}\n"
        f"They were not returned by any search. Call `search_parameters` again and "
        f"copy the ids from its output verbatim.",
        bogus,
    )


# Words from a recipe's own `# outputs:` header that carry no diagnostic weight — they
# name the domain, not the specific calibrated quantity, and appear in almost any
# hand-written stand-in for the same computation regardless of whether it was ever
# pasted in and called.
_GENERIC_OUTPUT_WORDS = frozenset({"shock", "speed", "figure", "plot", "table", "array", "printed"})


def _exported_names(artifacts: list[dict]) -> set[str]:
    """Lower-cased names of every value a run actually exported."""
    return {
        name.lower()
        for art in artifacts
        if art.get("kind") == "exports"
        for name, stats in (art.get("values") or {}).items()
        if isinstance(stats, dict) and not stats.get("error")
    }


def _flag_recipe_bypass(text: str, history: list, artifacts: list[dict]) -> tuple[str, list[dict]]:
    """Append a note when a run's exports suggest a recipe was skipped, or read but
    not actually used.

    Provenance checks where a number came from; this checks whether it should have come
    from a calibrated recipe instead of memory. Both annotate, neither blocks — the same
    design as `_flag_unknown_ids`.

    Two independent signals, both read from data already in the run's history:

    - **never loaded** — exported values match a recipe's usual vocabulary
      (`RECIPE_SIGNATURES`), and `load_recipe` was never called for it. Matching against
      the recipe's OWN export names does not work here: a hand-written Rankine-Hugoniot on
      the St Patrick 2015 shock exported "density_compression_ratio" and
      "predicted_RH_compression", not the recipe's "r" / "r_predicted".
    - **loaded but shallow** — `load_recipe` WAS called, but none of that recipe's own
      declared outputs (its `# outputs:` header, already sitting in the tool result) show
      up among what was actually exported. This is the case the first signal misses: on
      one run, `rankine_hugoniot` and `shock_timing_2sc` were both loaded first, as
      instructed — and both were then read for their constants and formula shape and
      reimplemented anyway, with hand-picked averaging windows instead of the recipe's
      calibrated ones, and none of `transverse_separation_km` / `verdict` /
      `mom_residual` — the recipe's own consistency checks — ever computed. One task's own
      code left three inert strings named after the recipes as its only trace of having
      "used" them.

    Neither signal is proof, and the second under-fires on a recipe whose declared outputs
    share the field's ordinary vocabulary ("compression ratio", "spacecraft frame") with
    whatever a hand-written stand-in would also naturally call itself — it catches the
    recipes with distinctive output names, not all of them.

    Args:
        text: The finished answer.
        history: The run's message history, to check which recipes were loaded and
            read their declared outputs from the matching tool results.
        artifacts: The run's artifacts, to read what was exported.

    Returns:
        The text (with a note appended when needed) and the flags raised, each
        `{"recipe": name, "reason": "not_loaded" | "shallow_use"}`.
    """
    from helioai.tools.recipes import RECIPE_SIGNATURES

    loaded_calls: dict[str, str] = {}  # tool_call_id -> recipe name
    for m in history:
        for tc in m.tool_calls or []:
            if tc.name == "load_recipe":
                name = (tc.arguments or {}).get("name")
                if name:
                    loaded_calls[tc.id] = name
    loaded_names = set(loaded_calls.values())

    exported = _exported_names(artifacts)
    if not exported:
        return text, []

    flags: list[dict] = [
        {"recipe": recipe_name, "reason": "not_loaded"}
        for recipe_name, signatures in RECIPE_SIGNATURES.items()
        if recipe_name not in loaded_names
        and any(sig in name for name in exported for sig in signatures)
    ]

    tool_results = {m.tool_call_id: m.content for m in history if m.role == "tool"}
    for call_id, recipe_name in loaded_calls.items():
        raw = tool_results.get(call_id)
        if not raw:
            continue
        try:
            outputs_field = (json.loads(raw).get("metadata") or {}).get("outputs", "")
        except (ValueError, TypeError):
            continue
        tokens = {
            t
            for t in re.split(r"[^a-z0-9]+", outputs_field.lower())
            if len(t) > 4 and t not in _GENERIC_OUTPUT_WORDS
        }
        if tokens and not any(tok in name for name in exported for tok in tokens):
            flags.append({"recipe": recipe_name, "reason": "shallow_use"})

    if not flags:
        return text, []

    lines = [
        f"  - {f['recipe']}: never loaded"
        if f["reason"] == "not_loaded"
        else f"  - {f['recipe']}: loaded, but its declared outputs never appeared in what was exported"
        for f in flags
    ]
    return (
        f"{text}\n\n"
        "ℹ️ RECIPE CHECK — these exported values resemble a computation that has a "
        "calibrated recipe:\n" + "\n".join(lines) + "\n"
        "Recipes carry calibrated parameters and a self-test that hand-written code does "
        "not have — verify the numbers above against `load_recipe(name)` (read AND call its "
        "functions, not just its constants) before trusting them.",
        flags,
    )


def check_answer(
    text: str, history: list, artifacts: list[dict]
) -> tuple[str, list[str], list[dict]]:
    """Confront a finished answer with the catalogue and the recipe shelf.

    Both loops call this, which is the whole point of it living here. Both checks were
    written inside the sub-agent loop and stayed there, so a lead agent that did the
    physics itself — Acts III and IV of the showcase notebook, on a run where
    `load_recipe` was called zero times all session — was never checked at all. The
    detectors were not silent because the run was clean; they were silent because
    nothing called them.

    Returns:
        The text (annotated when a check fires), the unknown ids, and the recipe flags.
    """
    text, bogus = _flag_unknown_ids(text)
    text, bypassed = _flag_recipe_bypass(text, history, artifacts)
    return text, bogus, bypassed
