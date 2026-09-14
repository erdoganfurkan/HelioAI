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

import asyncio
import json
import re
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

from helioai import provenance
from helioai.core.event_display import describe_tool_result, finding_str
from helioai.core.events import artifact, make
from helioai.core.llm.base import ToolCall
from helioai.tools.results import ToolResult

# Tools whose results contain large lists (per_event_stats, sample rows) that would
# flood the LLM context. All other tools pass through untouched so the LLM can reason
# on their content (search results, mission lists, catalog ids, etc.).
_HEAVY_TOOLS: frozenset[str] = frozenset({"get_events_timeseries", "get_catalog"})


def _redact_host_paths(text: str) -> str:
    """Replace the operator's home directory with `~` in a tool result.

    The model cannot open a host path, so showing it one buys nothing and costs
    something: it quotes the path back in its answer, which then appears in the CLI,
    the web UI, an exported notebook and any screen recording of them. Only the home
    prefix goes — the file names stay, because "I saved fig_0_0.png" is a useful thing
    for the model to be able to say.

    Paths outside a home directory (a container's `/app/data`) are left alone.
    """
    home = str(Path.home()).rstrip("/")
    if not home or home == "/":
        return text
    return text.replace(home, "~")


def _history_tool_result(tool_name: str, result_text: str) -> str:
    """Return the string that gets appended to history for a completed tool call.

    Heavy tools (SEA, catalog preview) are summarized to avoid flooding the context.
    All other tools — especially search_parameters — are passed through verbatim so
    the LLM can see the actual candidates and make an informed choice.

    Either way the operator's home directory is redacted: this is the only place a
    tool result crosses into the model's context, so it is the one place the fix
    belongs. The events keep real paths — the CLI has to open the figure.
    """
    if tool_name in _HEAVY_TOOLS:
        return _redact_host_paths(_summarize_tool_result(result_text, max_chars=600))
    return _redact_host_paths(result_text)


def _export_str(stats) -> str | None:
    """One line for a sandbox export: `mean units`, plus `[min, max]` for an array.

    None for a failed export (a string handed to `export()`) — there is no number to
    remember, and the error text is already in stdout.
    """
    if not isinstance(stats, dict) or stats.get("error"):
        return None
    mean = stats.get("mean")
    if mean is None:
        return None
    out = f"{mean} {stats.get('units') or ''}".strip()
    if stats.get("shape") and stats.get("min") is not None:
        out += f" [{stats['min']}, {stats['max']}]"
    return out


# The numbers a result carries are kept whole and outside the cap; the prose is what
# shrinks to fit. Sizes are per field: a sub-agent's `summary` is the lead's only account
# of what was done and how, and `stdout` is where the analyst printed its diagnostics.
# Everything else that is text is a label and 120 characters is plenty for one.
_NUMBER_TABLES = ("findings", "exports")
_TEXT_ROOM = {"summary": 1000, "stdout": 400}
_LABEL_ROOM = 120
_MIN_TEXT_ROOM = 60
_STALE_RESULT_CHARS = 1500


def _clip(text: str, room: int) -> str:
    return text if len(text) <= room else text[: max(room - 3, 0)] + "..."


def _summarize_tool_result(result_text: str, max_chars: int = 400) -> str:
    """Shrink a tool result for the LLM payload once it is no longer the latest.

    What a stale result must still say was learned from four live runs of 00_quickstart:

    - **Every number, always.** `findings` (a sub-agent's measured values) and `exports`
      (a sandbox run's `export()` calls) are rendered one line each and never cut. At
      300 characters the analyst's own `run_python` result read `"exports": "{11 keys}"`
      two turns after computing θ_Bn, and the lead's copy of a 9 kB report was sliced
      in the middle of its findings dict — invalid JSON, compression ratio gone.
    - **Enough prose to be a memory.** `summary` is the lead's only account of what a
      sub-agent did and how; `stdout` is where the analyst printed its diagnostics.
      Both get their own room, and they are what shrinks when the cap is reached, so
      the cap cuts on a field boundary and the JSON stays valid.
    - **Errors keep their traceback** — losing stderr two turns later is why one typo
      was retried three times in a session.

    `max_chars` bounds everything except the numbers tables; when those alone exceed
    it, they are returned in full anyway.
    """
    try:
        data = json.loads(result_text)
    except (ValueError, TypeError):
        return result_text[:max_chars]
    if not isinstance(data, dict):
        return str(data)[:max_chars] if not isinstance(data, list) else f"[list, {len(data)} items]"
    if data.get("error"):
        summary = f"error: {str(data['error'])[:max_chars]}"
        stderr = str(data.get("stderr") or "").strip()
        if stderr:
            summary += f"\n{stderr[-max_chars:]}"
        if isinstance(data.get("findings"), dict) and data["findings"]:
            # A run that hit its turn cap still measured things on the way there, and
            # dropping them here is how a lead ends up with nothing to report but prose.
            measured = ", ".join(f"{n}={finding_str(d)}" for n, d in data["findings"].items())
            summary += f"\nmeasured before failing: {measured}"
        return summary

    keep: dict = {}
    text_fields: list[str] = []
    for k, v in data.items():
        if k == "findings" and isinstance(v, dict):
            keep[k] = {n: finding_str(d) for n, d in v.items()}
        elif k == "exports" and isinstance(v, dict):
            lines = {n: _export_str(s) for n, s in v.items()}
            keep[k] = {n: s for n, s in lines.items() if s is not None}
        elif k == "figure_paths":
            keep[k] = [Path(p).name for p in v] if v else []
        elif isinstance(v, str):
            keep[k] = _clip(v, _TEXT_ROOM.get(k, _LABEL_ROOM))
            if k in _TEXT_ROOM:
                text_fields.append(k)
        elif isinstance(v, (int, float, bool, type(None))):
            keep[k] = v
        elif isinstance(v, list):
            keep[k] = f"[{len(v)} items]"
        elif isinstance(v, dict):
            keep[k] = f"{{{len(v)} keys}}"

    def render() -> str:
        return json.dumps(keep, ensure_ascii=False)

    out = render()
    # The prose gives way first, longest field first, down to a floor that still says
    # what the field was about; the numbers are never touched.
    for k in sorted(text_fields, key=lambda f: -len(keep[f])):
        if len(out) <= max_chars:
            break
        room = max(len(keep[k]) - (len(out) - max_chars), _MIN_TEXT_ROOM)
        keep[k] = _clip(keep[k], room)
        out = render()
    if len(out) <= max_chars or any(k in keep for k in _NUMBER_TABLES):
        return out
    return out[:max_chars]


def _is_recipe_source(message) -> bool:
    """Whether a tool message is the payload `load_recipe` returns.

    Asked of the message's `name` first. Histories persisted before that field existed
    carry none, so for them the shape still identifies it: `name`, `code` and `metadata`
    together are produced by no other tool.
    """
    name = getattr(message, "name", None)
    if name is not None:
        return name == "load_recipe"
    try:
        data = json.loads(message.content)
    except (ValueError, TypeError):
        return False
    return isinstance(data, dict) and all(k in data for k in ("name", "code", "metadata"))


def compact_history(messages: list, keep_full: int = 2) -> list:
    """Return a copy of `messages` where tool-result messages older than the last
    `keep_full` are summarized. The most recent results stay verbatim (the next LLM
    call usually needs them); older ones — already consumed — are trimmed so context
    does not grow unbounded over a long session. Persisted history is left untouched;
    only the per-call payload shrinks.

    A loaded recipe is never summarized. It is the one result the model keeps writing
    code against for the rest of the task, and two `run_python` calls later it had been
    cut to its first line: in two live runs the analyst reloaded it, and once rewrote the
    formula from memory rather than call the function it could no longer see. A recipe
    is a few kilobytes; keeping it costs less than the extra turn.

    The cap per stale result is 1 500 characters, up from 300. Measured on the live
    runs of 00_quickstart: every tool result of a four-turn session together is 14 kB,
    less than the system prompt and tool schemas re-sent on every call, and the numbers
    are exempt from the cap anyway. What 300 bought was a lead that could not recall
    the previous cell's method and an analyst that had lost its own exports.

    ponytail: fixed window N=2; widen keep_full if a case regresses on stale results.

    Args:
        messages: The per-call payload, newest last.
        keep_full: How many of the most recent tool results to leave verbatim.
            Zero summarises every one of them, which is the degraded retry a
            context-length failure would want.

    Returns:
        A new list. `messages` is not modified, because it is also what gets
        persisted.
    """
    tool_idx = [i for i, m in enumerate(messages) if getattr(m, "role", None) == "tool"]
    if len(tool_idx) <= keep_full:
        return messages
    stale = set(tool_idx[:-keep_full])
    return [
        replace(m, content=_summarize_tool_result(m.content, max_chars=_STALE_RESULT_CHARS))
        if i in stale and m.content and not _is_recipe_source(m)
        else m
        for i, m in enumerate(messages)
    ]


def _extract_artifact(tool_name: str, payload: object) -> list[dict]:
    """Extract renderable artifacts from a tool's payload (plots, parameter cards).

    Args:
        tool_name: The tool that produced it.
        payload: `ToolResult.payload` — read as the dict it is; anything else has no
            artifacts. The JSON text used to be parsed here, a second time.
    """
    if not isinstance(payload, dict):
        return []
    data = payload
    if "error" in data:
        # A failed run leaves the offending script on disk. Surfacing it is the whole
        # point when something broke — hiding it left the user watching a bare
        # "exited with code 1" with no way to see what ran.
        if tool_name == "run_python" and data.get("code_path"):
            return [
                artifact(
                    "code",
                    tool=tool_name,
                    code_path=data["code_path"],
                    name=Path(data["code_path"]).name,
                    n_lines=data.get("n_lines"),
                    failed=True,
                )
            ]
        return []

    artifacts: list[dict] = []

    # Python sandbox: figures + parameter cards emitted via param_card()
    if tool_name == "run_python":
        if data.get("figure_paths"):
            artifacts.append(
                artifact(
                    "image",
                    tool=tool_name,
                    figure_paths=data["figure_paths"],
                    stdout=data.get("stdout", ""),
                )
            )
        if data.get("exports"):
            artifacts.append(
                artifact(
                    "exports",
                    tool=tool_name,
                    values=data["exports"],
                    code_path=data.get("code_path"),
                )
            )
        for card in data.get("cards", []):
            if not isinstance(card, dict):
                continue
            if card.get("kind") == "parameter_card":
                fields = {k: v for k, v in card.items() if k != "kind"}
                artifacts.append(artifact("parameter_card", tool=tool_name, **fields))
            elif card.get("kind") == "method_used":
                artifacts.append(
                    artifact(
                        "recipe_used",
                        tool=tool_name,
                        name=card.get("name", ""),
                        reference=card.get("reference", ""),
                        description=card.get("method", ""),
                    )
                )
        if data.get("code_path"):
            artifacts.append(
                artifact(
                    "code",
                    tool=tool_name,
                    code_path=data["code_path"],
                    name=Path(data["code_path"]).name,
                    n_lines=data.get("n_lines"),
                )
            )

    # load_recipe: surface the recipe + its scientific reference for provenance
    if tool_name == "load_recipe" and data.get("name"):
        meta = data.get("metadata") or {}
        artifacts.append(
            artifact(
                "recipe_used",
                tool=tool_name,
                name=data["name"],
                reference=meta.get("reference", ""),
                description=meta.get("description", ""),
            )
        )

    # get_catalog result
    if tool_name == "get_catalog" and data.get("_kind") == "catalog_preview":
        artifacts.append(
            artifact(
                "catalog_preview",
                tool=tool_name,
                catalog_id=data.get("catalog_id"),
                name=data.get("name"),
                type=data.get("type"),
                nb_events_total=data.get("nb_events_total"),
                columns=data.get("columns", []),
                sample=(data.get("sample") or [])[:5],
                survey_start=data.get("survey_start"),
                survey_stop=data.get("survey_stop"),
            )
        )

    # get_timeseries called directly by main agent
    if tool_name == "get_timeseries" and "preview" in data:
        card = artifact(
            "parameter_card",
            tool=tool_name,
            param_id=data.get("param_id"),
            name=data.get("name"),
            mission=data.get("mission"),
            instrument=data.get("instrument"),
            units=data.get("units"),
            cadence=data.get("cadence"),
            components=data.get("components"),
            n_points=data.get("n_points"),
            start=data.get("start"),
            stop=data.get("stop"),
        )
        if (data.get("quality") or {}).get("notable"):
            card["quality"] = data["quality"]
        artifacts.append(card)

    return artifacts


def _code_path(payload: object) -> str:
    return (payload.get("code_path") or "") if isinstance(payload, dict) else ""


def inject_run_python_args(name: str, *, no_network: bool = False) -> dict:
    """Trusted per-run sandbox args (_plot_dir/_run_idx/_no_net) for run_python.

    Passed via `call_tool(..., trusted=...)` so they bypass the private-arg
    guard that rejects LLM/MCP-supplied `_*` overrides. Empty for any other tool.

    Args:
        name: Tool about to be called. Anything but `run_python` gets nothing.
        no_network: Whether to deny the sandbox a network namespace.

    Returns:
        The trusted argument dict, empty for every other tool.
    """
    if name != "run_python":
        return {}
    import helioai.workspace as _ws

    sdir = _ws.get_session_dir()
    ridx = _ws.get_next_run_idx(sdir)
    args = {"_plot_dir": str(sdir), "_run_idx": ridx}
    if no_network:
        args["_no_net"] = True
    return args


# Tools that must run one at a time within a turn: run_python numbers its scripts from
# what is on disk (`get_next_run_idx`) and writes into the one session directory.
_SEQUENTIAL_TOOLS: frozenset[str] = frozenset({"run_python"})


def start_tool_calls(
    tool_calls: list[ToolCall] | None, *, allowed: set[str] | None = None
) -> dict[str, asyncio.Task]:
    """Start every parallel-safe registry call of a turn at once, keyed by call id.

    The prompt asks the model to batch its downloads in one turn, and the loops then
    ran them one after the other: a data_analyst's first turn — three or four
    `get_timeseries` — took the sum of their durations. Started here, they overlap;
    the caller still awaits each result in the model's order, so every event and
    every `tool` message keeps the order it had when the calls were sequential.

    Skipped, and left to the caller's sequential path: `run_python` (see
    `_SEQUENTIAL_TOOLS`), anything not in the registry (the `task` tool, the internal
    tools), and — for a sub-agent — anything outside its whitelist, which the caller
    refuses without dispatching.

    Args:
        tool_calls: The assistant's tool calls for this turn.
        allowed: A sub-agent's whitelist; None for the lead, who may call anything.

    Returns:
        `{tool_call.id: task}` for the calls that were started; each task resolves to
        a `ToolResult`. `registry.call_tool` never raises — a failure is a failed
        result — so awaiting a task is safe.
    """
    from helioai.tools.registry import registry

    started: dict[str, asyncio.Task] = {}
    for tc in tool_calls or []:
        if tc.name in _SEQUENTIAL_TOOLS or tc.name not in registry:
            continue
        if allowed is not None and tc.name not in allowed:
            continue
        started[tc.id] = asyncio.create_task(registry.call_tool(tc.name, tc.arguments))
    return started


def cancel_pending(started: dict[str, asyncio.Task]) -> None:
    """Cancel the calls a turn started and never awaited — a cancelled turn must not
    leave downloads running for nobody."""
    for task in started.values():
        if not task.done():
            task.cancel()


def emit_post_tool_events(
    name: str,
    result: ToolResult,
    *,
    tool_result_extra: dict | None = None,
    common_extra: dict | None = None,
) -> Iterator[dict]:
    """Yield the events that follow a completed tool call.

    Order is `tool_result` → (`skill_loaded` if load_skill) → `artifact`(s),
    matching what both loops emitted before this was factored out.

    Args:
        name: The tool that just ran.
        result: Its result; the payload is read for artifacts, the model's text for
            the summary the event carries.
        tool_result_extra: Merged into the `tool_result` event data, e.g. `{turn}`.
        common_extra: Merged into `skill_loaded` and `artifact` event data, e.g.
            `{sub_agent_ctx}` when a sub-agent is the caller.

    Yields:
        The events, in the order both loops emitted them before this was
        factored out: `tool_result`, then `skill_loaded`, then artifacts.
    """
    tool_result_extra = tool_result_extra or {}
    common_extra = common_extra or {}
    text = result.for_llm()
    payload = result.payload

    # `summary` is written for the model and stays untouched. `display` is the same
    # event told to a person, computed here so the CLI, the Jupyter magic and the
    # browser show identical words without three copies of the logic.
    yield make(
        "tool_result",
        name=name,
        summary=_summarize_tool_result(text),
        display=describe_tool_result(name, text),
        **tool_result_extra,
    )

    if name == "load_skill" and isinstance(payload, dict):
        if payload.get("body") and not payload.get("error"):
            yield make("skill_loaded", name=payload.get("name", ""), **common_extra)

    for art in _extract_artifact(name, payload):
        if art.get("kind") == "exports":
            ctx = common_extra.get("sub_agent_ctx") or {}
            provenance.record(
                art.get("values") or {},
                code_path=_code_path(payload),
                agent=ctx.get("role") or "lead",
                task_id=ctx.get("task_id"),
                turn=tool_result_extra.get("turn"),
            )
        yield make("artifact", **art, **common_extra)


def unknown_id_correction(bogus: list[str]) -> str:
    """The correction handed back when an answer quotes ids that are not in the catalogue.

    Shared between the retry that buys the model another turn and the annotation left
    on an answer that has run out of turns, so both say exactly the same thing.

    Args:
        bogus: The ids that are absent from the index.

    Returns:
        The correction text, without the answer it refers to.
    """
    listed = "\n".join(f"  - {i}" for i in bogus)
    return (
        f"⚠️ AUTOMATED CORRECTION — the following ids are NOT in the catalogue and "
        f"must not be used:\n{listed}\n"
        f"They were not returned by any search. Call `search_parameters` again and "
        f"copy the ids from its output verbatim."
    )


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
    return f"{text}\n\n{unknown_id_correction(bogus)}", bogus


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


_DEF_LINE = re.compile(r"^[ \t]*def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)


def _recipe_functions(code: str) -> set[str]:
    """Public function names a recipe defines — what "calling the recipe" means.

    Private helpers are left out: `_rh_core` is the recipe's business, `rh_jump` is
    the contract. A script-shaped recipe with no public `def` (solar_mach,
    superposed_epoch) has nothing to call and is not judged on this signal.
    """
    return {n for n in _DEF_LINE.findall(code) if not n.startswith("_")}


def _calls_any(code: str, names: set[str]) -> bool:
    """Whether `code` invokes one of `names` — a definition of it does not count."""
    body = _DEF_LINE.sub("", code)
    return any(re.search(rf"\b{re.escape(n)}\s*\(", body) for n in names)


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

    - **loaded but never called** — `load_recipe` WAS called, and no `run_python` that
      followed it in this history invokes one of the recipe's public functions. The
      second signal cannot see this case when the hand-written copy exports the
      recipe's own name: on the fourth live run of 00_quickstart the analyst loaded
      `theta_bn`, rewrote the coplanarity formula inline, exported `theta_bn` — and got
      54.85° from a 12-minute averaging window the recipe would not have chosen. The
      export name matched, the function never ran. The function names come from the
      recipe source already sitting in the tool result (`compact_history` keeps it
      verbatim), the calls from the `code` argument of the later `run_python` calls.
      A history with no `run_python` after the load is not judged: a lead that loads a
      recipe and delegates the computation has not bypassed anything.

    None of the signals is proof, and the second under-fires on a recipe whose declared
    outputs share the field's ordinary vocabulary ("compression ratio", "spacecraft
    frame") with whatever a hand-written stand-in would also naturally call itself — it
    catches the recipes with distinctive output names, not all of them. The third is
    textual: a run that redefines a function under the recipe's own name passes it.
    One flag per recipe; "never called" is the more specific finding and wins.

    Args:
        text: The finished answer.
        history: The run's message history, to check which recipes were loaded, read
            their declared outputs and source from the matching tool results, and see
            what the later `run_python` calls actually invoked.
        artifacts: The run's artifacts, to read what was exported.

    Returns:
        The text (with a note appended when needed) and the flags raised, each
        `{"recipe": name, "reason": "not_loaded" | "shallow_use" | "not_called"}`.
    """
    from helioai.tools.recipes import RECIPE_SIGNATURES

    loaded_calls: dict[str, str] = {}  # tool_call_id -> recipe name
    load_positions: dict[str, int] = {}  # tool_call_id -> index in history
    python_calls: list[tuple[int, str]] = []  # (index in history, code)
    for i, m in enumerate(history):
        for tc in m.tool_calls or []:
            if tc.name == "load_recipe":
                name = (tc.arguments or {}).get("name")
                if name:
                    loaded_calls[tc.id] = name
                    load_positions[tc.id] = i
            elif tc.name == "run_python":
                python_calls.append((i, str((tc.arguments or {}).get("code") or "")))
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
            payload = json.loads(raw)
            outputs_field = (payload.get("metadata") or {}).get("outputs", "")
            functions = _recipe_functions(str(payload.get("code") or ""))
        except (ValueError, TypeError, AttributeError):
            continue
        later_code = [code for i, code in python_calls if i > load_positions[call_id]]
        if functions and later_code and not any(_calls_any(c, functions) for c in later_code):
            flags.append({"recipe": recipe_name, "reason": "not_called"})
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

    reasons = {
        "not_loaded": "never loaded",
        "shallow_use": "loaded, but its declared outputs never appeared in what was exported",
        "not_called": "loaded, but none of its functions was called by any later run_python — "
        "the computation was rewritten by hand",
    }
    lines = [f"  - {f['recipe']}: {reasons[f['reason']]}" for f in flags]
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

    Args:
        text: The finished answer.
        history: The turn's messages, read for evidence a recipe was loaded and
            then bypassed.
        artifacts: What the run exported, used to tell a computed value from a
            quoted one.

    Returns:
        The text (annotated when a check fires), the unknown ids, and the recipe flags.
    """
    text, bogus = _flag_unknown_ids(text)
    text, bypassed = _flag_recipe_bypass(text, history, artifacts)
    return text, bogus, bypassed
