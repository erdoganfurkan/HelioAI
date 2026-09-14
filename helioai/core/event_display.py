"""One-line, human-facing descriptions of tool calls and their results.

`tool_exec._summarize_tool_result` writes for the *model*: dense, complete, JSON, and
deliberately hard to degrade — its comments record why a traceback has to survive
compaction and why `findings` must not collapse to `{3 keys}`. Every interface was
showing that same string to a *person*, truncated mid-key, which is how a session that
worked read like one that crashed.

So this module writes for the reader instead, and leaves the model's briefing alone.
It is computed once at emission and travels on the event as `display`, because the web
UI is JavaScript and cannot call into Python — without that, the same formatting would
have to exist three times and drift, exactly as the event handlers already had.
"""

from __future__ import annotations

import json

_MAX = 160


def _clip(text: str, limit: int = _MAX) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _window(args: dict) -> str:
    start, stop = args.get("start"), args.get("stop")
    if not start:
        return ""
    return f" {str(start)[:16]}→{str(stop)[:16]}" if stop else f" from {str(start)[:16]}"


def finding_str(entry) -> str:
    """`value units [min, max]` for one sub-agent finding, the way the model sees it."""
    if not isinstance(entry, dict):
        return str(entry)[:60]
    out = f"{entry.get('value')} {entry.get('units') or ''}".strip()
    if entry.get("min") is not None:
        out += f" [{entry['min']}, {entry['max']}]"
    return out


def describe_findings(findings: dict | None, limit: int = 8) -> list[str]:
    """One `name = value units` line per measured value, for a person.

    The same text in the CLI, the notebook and the browser, computed once here. Capped
    because a sub-agent that exported a diagnostic table would otherwise print it whole;
    the cap says how many were left out.

    Args:
        findings: The `findings` table of a `sub_agent_end` event.
        limit: Lines to show before summarising the rest.

    Returns:
        Lines ready to print, empty when there is nothing measured.
    """
    if not findings:
        return []
    items = list(findings.items())
    lines = [f"{name} = {finding_str(entry)}" for name, entry in items[:limit]]
    if len(items) > limit:
        lines.append(f"… and {len(items) - limit} more")
    return lines


def describe_tool_call(name: str, arguments: dict | None) -> str:
    """Describe what the agent is about to do, in a reader's terms.

    Args:
        name: Registered tool name.
        arguments: Decoded arguments, possibly empty when the model emitted bad JSON.

    Returns:
        A single line with no newlines, safe to print or drop into HTML text.
    """
    args = arguments or {}

    if name == "run_python":
        # The old renderers printed 40 characters of source, which reliably cut a
        # comment in half and left an unbalanced quote on screen. Size is the only
        # thing a reader can use here; the code itself has its own panel.
        n_lines = len(str(args.get("code", "")).splitlines())
        return f"{n_lines} lines of Python" if n_lines else "Python"

    if name == "run_recipe":
        bound = args.get("inputs") or {}
        names = ", ".join(str(k) for k in bound) if isinstance(bound, dict) else ""
        return _clip(f"{args.get('name', '')}({names})")

    if name == "search_parameters":
        queries = args.get("queries")
        if isinstance(queries, list) and queries:
            head = _clip(str(queries[0]), 60)
            body = f"{len(queries)} queries — {head}…" if len(queries) > 1 else f'"{head}"'
        else:
            body = f'"{_clip(str(args.get("query", "")), 80)}"'
        provider = args.get("provider")
        return f"{body} ({provider})" if provider else body

    if name in ("get_timeseries", "get_events_timeseries"):
        pid = args.get("param_id", "")
        catalog = args.get("catalog") or args.get("catalog_id")
        suffix = f" over {catalog}" if catalog else _window(args)
        return _clip(f"{pid}{suffix}")

    if name in ("get_catalog", "save_catalog", "load_recipe", "load_skill"):
        return _clip(str(args.get("name") or args.get("catalog_id") or ""))

    if name == "find_papers":
        return _clip(f'"{args.get("query", "")}"')

    if name == "task":
        role = args.get("agent_role", "")
        desc = _clip(str(args.get("description", "")), 70)
        return f"{role} — {desc}" if desc else str(role)

    if not args:
        return ""
    return _clip(", ".join(f"{k}={_clip(str(v), 40)}" for k, v in args.items()))


def _describe_error(data: dict) -> str:
    error = _clip(str(data.get("error", "")), 200)
    return error if error.lower().startswith("error") else f"error: {error}"


def describe_tool_result(name: str, result: str) -> str:
    """Describe what came back, keeping the few facts a reader acts on.

    Args:
        name: Registered tool name.
        result: The tool's raw result payload, usually JSON.

    Returns:
        A single line with no newlines. Errors lead with the exception, because that is
        what the reader needs first and what the old renderer buried in a dict.
    """
    if not result:
        return ""
    try:
        data = json.loads(result)
    except (ValueError, TypeError):
        return _clip(result)
    if not isinstance(data, dict):
        return _clip(str(data))

    if data.get("error"):
        return _describe_error(data)

    if name == "get_timeseries":
        bits = [str(data.get("param_id") or data.get("name") or "")]
        for key, label in (("mission", ""), ("cadence", ""), ("n_points", " points")):
            value = data.get(key)
            if value not in (None, ""):
                bits.append(f"{value}{label}")
        return _clip(" · ".join(b for b in bits if b))

    if name == "search_parameters":
        results = data.get("results")
        if isinstance(results, list):
            top = ""
            if results and isinstance(results[0], dict):
                top = str(results[0].get("id", ""))
            return _clip(f"{len(results)} hits" + (f", top {top}" if top else ""))

    if name in ("run_python", "run_recipe"):
        bits = []
        n_fig = data.get("n_figures") or len(data.get("figure_paths") or [])
        if n_fig:
            bits.append(f"{n_fig} figure(s)")
        exports = data.get("exports")
        if isinstance(exports, dict) and exports:
            bits.append(f"{len(exports)} exported")
        stdout = str(data.get("stdout") or "").strip()
        if stdout:
            bits.append(_clip(stdout.splitlines()[0], 80))
        return _clip(" · ".join(bits)) if bits else "ok"

    if name == "find_papers":
        papers = data.get("papers") or data.get("results")
        if isinstance(papers, list):
            return f"{len(papers)} paper(s)"

    # Unknown or simple tools: the scalar fields, spelled out rather than dumped as a
    # dict. A new tool therefore reads acceptably before anyone touches this file.
    bits = [
        f"{k} {v}"
        for k, v in data.items()
        if isinstance(v, (str, int, float, bool)) and str(v) and len(str(v)) <= 60
    ]
    return _clip(" · ".join(bits[:4])) if bits else "ok"


def describe_verdict(data: dict) -> tuple[str, list[str]]:
    """One line of counts and one line per claim worth a look, for a `verdict` event.

    The claims the model named are judged by name against the ledger; a contradicted
    one is the strongest signal this system has — the value was computed, and the answer
    states another — so it is spelled out with both numbers. Unsourced claims are listed
    with the source the model gave them (`literature`, `asserted`, or an export the
    session never wrote), matched ones only counted.

    Args:
        data: The event payload.

    Returns:
        The summary line and the detail lines, without any styling.
    """
    summary = (
        f"claims — {data.get('matched', 0)} backed, {data.get('contradicted', 0)} contradicted, "
        f"{data.get('unsourced', 0)} unsourced"
    )
    lines: list[str] = []
    for c in data.get("claims") or []:
        stated = f"{c.get('value')}{' ' + c['units'] if c.get('units') else ''}"
        if c.get("status") == "contradicted":
            recorded = (
                f"{c.get('ledger')}{' ' + c['ledger_units'] if c.get('ledger_units') else ''}"
            )
            lines.append(
                f"contradicted: {c.get('name')} stated {stated}, the session computed {recorded}"
            )
        elif c.get("status") == "unsourced":
            note = f" ({c['note']})" if c.get("note") else ""
            source = c.get("source") or "asserted"
            lines.append(f"unsourced: {c.get('name')} = {stated} — source: {source}{note}")
    return summary, lines[:8]


def describe_plan_report(data: dict) -> str:
    """One line on how the turn followed its plan, for a `plan_report` event.

    Reads as a sentence a person can act on — "3/4 planned tools used, not
    get_timeseries; unplanned: find_papers" — rather than a ratio alone. A plan that
    named no tool is said so; nothing here judges whether the deviation was right.

    Args:
        data: The event payload.

    Returns:
        The line, without any styling.
    """
    planned, executed = data.get("planned") or [], data.get("executed") or []
    missed, unplanned = data.get("missed_tools") or [], data.get("unplanned_tools") or []
    if not planned:
        used = ", ".join(executed) if executed else "none"
        return f"plan named no tools — used: {used}"
    text = f"plan — {len(planned) - len(missed)}/{len(planned)} planned tools used"
    if missed:
        text += f", not {', '.join(missed)}"
    if unplanned:
        text += f"; unplanned: {', '.join(unplanned)}"
    return text
