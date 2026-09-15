#!/usr/bin/env python
"""Live benchmark harness: the same questions, N runs each, one configuration at a time.

A single run says nothing about a configuration. The maintainer's database holds 19 runs
of *one* question — "find an interplanetary shock in WIND data around 2004-11-07 and
compute θ_Bn" — and the reported θ_Bn ranges from 44° to 80°, because the day holds two
shocks (10:03 UT, weak and clean; 17:59 UT, strong, the SSC driver) and the agent picks
either. Comparing two branches on one run each therefore compares noise. This script
runs a fixed question set several times under whatever configuration the environment
selects (provider, `HELIOAI_ROLE_MODELS`, experiment flags, git HEAD), records every
session id in a manifest, and scores the sessions afterwards from the SQLite journal so
the comparison is repeatable and needs no second LLM call.

Two subcommands:

    run    drive `stream_chat` on each question × repetition (spends real tokens; the
           maintainer runs this, never CI) and append one entry per run to a manifest
    score  open the session database read-only, compute per-session metrics, print a
           table and a per-question aggregate; optionally write CSV/JSON

`score` handles the two session shapes in the database: sessions journaled by the
event stream (`events` table — preferred, every tool call of every agent is visible)
and sessions from the `main` branch, which predate the journal and only have
`messages`. In the second shape a delegated role's own tool calls are not recorded —
only its `task` result is — so `n_search`/`n_download` count the lead's calls alone
and the delegation's iteration count comes from the `task` result JSON. Each score
says which shape it used.

Usage:
    python scripts/bench_live.py run --n 3 --label "branch-A" --out bench_A.json
    python scripts/bench_live.py score --manifest bench_A.json --csv bench_A.csv
    python scripts/bench_live.py score --session web:<session_id> --session web:<other>
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sqlite3
import statistics
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = Path(__file__).resolve().parent / "bench_questions.json"
BENCH_USER = "bench"

SEARCH_TOOLS = frozenset({"search_parameters", "list_missions"})
DOWNLOAD_TOOLS = frozenset({"get_timeseries", "get_events_timeseries"})

# Environment knobs worth recording next to each run: anything that changes the agent's
# behaviour without a code change. `HELIOAI_EXPERIMENTS` is recorded whether or not the
# current branch reads it, so manifests from different branches line up.
RECORDED_ENV = (
    "HELIOAI_EXPERIMENTS",
    "HELIOAI_LLM_PROVIDER",
    "HELIOAI_ROLE_MODELS",
    "HELIOAI_MAX_ITERATIONS",
)

# Phrases that show the answer weighed a candidate it did not take. Tune here: the
# `mentions_alternatives` metric is exactly "any of these matched".
ALTERNATIVE_MARKERS: tuple[str, ...] = (
    r"\breject(?:ed|ing|s)?\b",
    r"\bexcluded?\b",
    r"\bdiscarded\b",
    r"\bruled out\b",
    r"\bdid not (?:use|take|select|choose|pick)\b",
    r"\binstead of\b",
    r"\bother candidates?\b",
    r"\balternative candidates?\b",
    r"\bsecond (?:structure|jump|shock|candidate|crossing)\b",
    r"\bnot (?:the|a) shock\b",
)
_ALTERNATIVES_RE = re.compile("|".join(ALTERNATIVE_MARKERS), re.IGNORECASE)

_THETA_LABEL = r"(?:θ|\\?theta)\s*_?\s*\{?\s*Bn\s*\}?"
_THETA_SEP = (
    r"\s*\**\s*(?:=|≈|≃|≅|~|∼|:|\b(?:is|of|was)\b)\s*\**\s*"
    r"(?:≈|~|about|roughly|approximately|around)?\s*"
)
_THETA_VALUE = r"(?P<val>[-+]?\d+(?:\.\d+)?)\s*(?:°|º|˚|deg(?:rees)?)?"
_THETA_RE = re.compile(_THETA_LABEL + _THETA_SEP + _THETA_VALUE, re.IGNORECASE)
_SIGMA_RE = re.compile(
    r"(?:±|\+/-|(?:std|σ|sigma|uncertainty|error|spread)\s*(?:of|=|:)?\s*±?)\s*"
    r"(?P<sig>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_SIGMA_WINDOW = 80

_TIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}):(\d{2})(?!\d)")
_BENCH_SESSION_RE = re.compile(r"^bench-(?P<qid>.+)-[0-9a-f]{8}$")
_CAPPED_LEAD_RE = re.compile(r"exceeded \d+ iterations")


def parse_theta(text: str) -> tuple[float | None, float | None]:
    """The first θ_Bn statement in an answer, as (value, sigma) in degrees.

    Only a labelled assignment counts — `θ_Bn = 52.2°`, `θBn ≈ 56.1 °`,
    `theta_Bn: 49.09 deg`, `\\theta_{Bn} = 62.2° ± 1.1°` — so a mention such as
    "computed with the `theta_bn` recipe (Colburn & Sonett 1966)" never yields 1966.
    Values outside 0–180° are skipped as false positives. The uncertainty is whatever
    `±`, `std`, `σ` or `spread` figure follows within a short window of the value.

    Args:
        text: The answer text.

    Returns:
        `(theta, sigma)`; either is None when not stated.
    """
    for m in _THETA_RE.finditer(text):
        val = float(m.group("val"))
        if not 0.0 <= val <= 180.0:
            continue
        window = text[m.end() : m.end() + _SIGMA_WINDOW]
        s = _SIGMA_RE.search(window)
        return val, (float(s.group("sig")) if s else None)
    return None, None


def parse_times(text: str) -> list[str]:
    """Sorted distinct `YYYY-MM-DDTHH:MM` stamps an answer names.

    Accepts `T` or a space between date and time and drops seconds, so
    `2004-11-07T10:03:43 UT` and `2004-11-07 10:03` are the same minute. A bare date
    is not a time and is ignored.

    Args:
        text: The answer text.

    Returns:
        Minute-resolution stamps, sorted, each once.
    """
    out: set[str] = set()
    for date, hh, mm in _TIME_RE.findall(text):
        if int(hh) < 24 and int(mm) < 60:
            out.add(f"{date}T{hh}:{mm}")
    return sorted(out)


def mentions_alternatives(text: str) -> bool:
    """Whether the answer explains a candidate it rejected (see `ALTERNATIVE_MARKERS`)."""
    return bool(_ALTERNATIVES_RE.search(text))


@dataclass
class SessionScore:
    """What one session did and what it answered, from the database alone.

    `shape` records which tables the score came from — `"events"` when the journal
    exists, `"messages"` for a `main`-branch session, `"empty"` when the session is
    absent — because the two shapes see different things (see the module docstring)
    and a comparison across them must say so.
    """

    user_id: str
    session_id: str
    shape: str
    duration_s: float | None = None
    lead_turns: int = 0
    delegations: list[dict] = field(default_factory=list)
    n_search: int = 0
    n_download: int = 0
    n_run_python: int = 0
    n_run_recipe: int = 0
    n_corrections: int = 0
    tokens: dict[str, dict[str, int]] = field(default_factory=dict)
    param_ids: list[str] = field(default_factory=list)
    answer: str = ""
    answer_words: int = 0
    times_utc: list[str] = field(default_factory=list)
    theta_bn: float | None = None
    theta_bn_sigma: float | None = None
    mentions_alternatives: bool = False
    verdict: dict | None = None
    provenance: dict | None = None
    errors: list[str] = field(default_factory=list)
    capped_lead: bool = False
    question_id: str | None = None
    question_text: str = ""

    @property
    def total_tokens(self) -> tuple[int, int]:
        """`(prompt, completion)` summed over every agent."""
        return (
            sum(t.get("prompt", 0) for t in self.tokens.values()),
            sum(t.get("completion", 0) for t in self.tokens.values()),
        )

    def key_answer(self) -> str:
        """The one value a repetition is compared on: θ_Bn when stated, else the first
        time named, else nothing — what "did the runs agree" means for each family."""
        if self.theta_bn is not None:
            return f"θ={self.theta_bn:g}"
        if self.times_utc:
            return self.times_utc[0]
        return "-"

    def to_row(self) -> dict[str, str]:
        """Flatten into the columns of the printed table and the CSV."""
        prompt, completion = self.total_tokens
        return {
            "question": self.question_id or "-",
            "session": self.session_id[:14],
            "shape": self.shape,
            "s": f"{self.duration_s:.0f}" if self.duration_s is not None else "-",
            "turns": str(self.lead_turns) + ("!" if self.capped_lead else ""),
            "deleg": ";".join(
                f"{d['role']}:{d['n_iterations']}" + ("!" if d.get("capped") else "")
                for d in self.delegations
            )
            or "-",
            "srch": str(self.n_search),
            "dl": str(self.n_download),
            "py": str(self.n_run_python),
            "rcp": str(self.n_run_recipe),
            "corr": str(self.n_corrections),
            "tok_p": _kilo(prompt),
            "tok_c": _kilo(completion),
            "params": str(len(self.param_ids)),
            "theta": f"{self.theta_bn:g}" if self.theta_bn is not None else "-",
            "sigma": f"{self.theta_bn_sigma:g}" if self.theta_bn_sigma is not None else "-",
            "times": ",".join(t[5:] for t in self.times_utc) or "-",
            "alt": "y" if self.mentions_alternatives else "n",
            "verdict": _triple(self.verdict),
            "prov": _triple(self.provenance),
            "words": str(self.answer_words),
            "err": str(len(self.errors)),
        }


def _kilo(n: int) -> str:
    return f"{n / 1000:.0f}k" if n else "-"


def _triple(d: dict | None) -> str:
    if not d:
        return "-"
    return f"{d.get('matched', 0)}/{d.get('contradicted', 0)}/{d.get('unsourced', 0)}"


def _loads(raw: str | None, default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def _load_tokens(conn: sqlite3.Connection, user_id: str, session_id: str) -> dict:
    rows = conn.execute(
        "SELECT agent, SUM(prompt_tokens), SUM(completion_tokens), SUM(cached_tokens) "
        "FROM usage WHERE user_id = ? AND session_id = ? GROUP BY agent",
        (user_id, session_id),
    ).fetchall()
    return {
        agent: {"prompt": int(p), "completion": int(c), "cached": int(k)} for agent, p, c, k in rows
    }


def _count_tool(score: SessionScore, name: str) -> None:
    if name in SEARCH_TOOLS:
        score.n_search += 1
    elif name in DOWNLOAD_TOOLS:
        score.n_download += 1
    elif name == "run_python":
        score.n_run_python += 1
    elif name == "run_recipe":
        score.n_run_recipe += 1


def _add_param(score: SessionScore, art: dict) -> None:
    pid = art.get("param_id") if art.get("kind") == "parameter_card" else None
    if pid and pid not in score.param_ids:
        score.param_ids.append(pid)


def _score_events(score: SessionScore, events: list[tuple[str, dict, float]]) -> None:
    lead_turn_max = 0
    done_seen = None
    for kind, data, _ts in events:
        if kind == "tool_call":
            _count_tool(score, data.get("name", ""))
            if "sub_agent_ctx" not in data:
                lead_turn_max = max(lead_turn_max, int(data.get("turn") or 0))
        elif kind == "artifact":
            _add_param(score, data)
        elif kind == "sub_agent_end":
            score.delegations.append(
                {
                    "role": data.get("role", "?"),
                    "n_iterations": int(data.get("n_iterations") or 0),
                    "capped": bool(data.get("capped")),
                }
            )
        elif kind == "correction":
            score.n_corrections += 1
        elif kind == "provenance":
            score.provenance = {k: data.get(k, 0) for k in ("matched", "contradicted", "unsourced")}
        elif kind == "verdict":
            score.verdict = {k: data.get(k, 0) for k in ("matched", "contradicted", "unsourced")}
        elif kind == "error":
            msg = str(data.get("message", ""))
            score.errors.append(msg)
            if _CAPPED_LEAD_RE.search(msg):
                score.capped_lead = True
        elif kind == "done":
            done_seen = int(data.get("n_iterations") or 0)
        elif kind == "user" and not score.question_text:
            score.question_text = str(data.get("text", ""))
    score.lead_turns = done_seen if done_seen is not None else lead_turn_max
    stamps = [ts for _k, _d, ts in events if ts is not None]
    if len(stamps) >= 2:
        score.duration_s = (max(stamps) - min(stamps)) * 86400.0


def _score_messages(score: SessionScore, messages: list[tuple], span: tuple | None) -> None:
    by_call_id = {tcid: content for _r, content, _tc, tcid, _o in messages if tcid}
    for role, content, tool_calls, _tcid, origin in messages:
        if role == "user" and not score.question_text and origin is None:
            score.question_text = content
        if role == "user" and origin == "correction":
            score.n_corrections += 1
        if role != "assistant":
            continue
        score.lead_turns += 1
        for tc in _loads(tool_calls, []):
            name = tc.get("name", "")
            _count_tool(score, name)
            if name != "task":
                continue
            args = tc.get("arguments") or {}
            result = _loads(by_call_id.get(tc.get("id")), {})
            if not isinstance(result, dict):
                continue
            err = str(result.get("error") or "")
            score.delegations.append(
                {
                    "role": args.get("agent_role", "?"),
                    "n_iterations": int(result.get("n_iterations") or 0),
                    "capped": bool(result.get("capped")) or "iterations" in err,
                }
            )
            for art in result.get("artifacts") or []:
                if isinstance(art, dict):
                    _add_param(score, art)
    if span and span[0] is not None and span[1] is not None:
        score.duration_s = (span[1] - span[0]) * 86400.0


def score_session(conn: sqlite3.Connection, user_id: str, session_id: str) -> SessionScore:
    """Score one session from an open (read-only is fine) connection to `sessions.db`.

    Prefers the event journal; falls back to the message history for sessions recorded
    before the journal existed. The answer is always the last non-empty assistant
    message, which both shapes have.

    Args:
        conn: A `sqlite3` connection to the session database.
        user_id: Owner of the session (`"web"`, `"cli"`, `"bench"`…).
        session_id: The session.

    Returns:
        A `SessionScore`; `shape == "empty"` when the session has neither events nor
        messages.
    """
    events = [
        (kind, _loads(data, {}), ts)
        for kind, data, ts in conn.execute(
            "SELECT kind, data, ts FROM events WHERE user_id = ? AND session_id = ? ORDER BY seq",
            (user_id, session_id),
        ).fetchall()
    ]
    messages = conn.execute(
        "SELECT role, content, tool_calls, tool_call_id, origin FROM messages "
        "WHERE user_id = ? AND session_id = ? ORDER BY seq",
        (user_id, session_id),
    ).fetchall()
    span = conn.execute(
        "SELECT created_at, updated_at FROM sessions WHERE user_id = ? AND session_id = ?",
        (user_id, session_id),
    ).fetchone()

    shape = "events" if events else ("messages" if messages else "empty")
    score = SessionScore(user_id=user_id, session_id=session_id, shape=shape)
    if shape == "events":
        _score_events(score, events)
        if not score.question_text:
            score.question_text = next((c for r, c, *_ in messages if r == "user" and c), "")
    elif shape == "messages":
        _score_messages(score, messages, span)

    answer = next((c for r, c, *_ in reversed(messages) if r == "assistant" and c.strip()), "")
    if not answer and events:
        answer = next((d.get("text", "") for k, d, _ in reversed(events) if k == "reply"), "")
    score.answer = answer
    score.answer_words = len(answer.split())
    score.times_utc = parse_times(answer)
    score.theta_bn, score.theta_bn_sigma = parse_theta(answer)
    score.mentions_alternatives = mentions_alternatives(answer)
    score.tokens = _load_tokens(conn, user_id, session_id)
    m = _BENCH_SESSION_RE.match(session_id)
    if m:
        score.question_id = m.group("qid")
    return score


def aggregate(scores: Iterable[SessionScore]) -> list[dict]:
    """Collapse repetitions of the same question into one row.

    Grouped by `question_id`, or by the question text when a session was not run
    through this harness (the maintainer's web sessions), so any set of runs of the
    same question aggregates. Reports what a single run cannot: how many distinct
    answers the runs gave, and the spread of θ_Bn where it was stated.

    Args:
        scores: Any iterable of `SessionScore`.

    Returns:
        One dict per question with `question`, `n`, `distinct`, `theta_mean`,
        `theta_std`, `theta_n`, `alt_frac`, `err_n`, `turns_mean`, `tokens_mean`,
        `seconds_mean`; means are None when nothing contributed.
    """
    groups: dict[str, list[SessionScore]] = {}
    for s in scores:
        key = s.question_id or (s.question_text.strip()[:60] or s.session_id)
        groups.setdefault(key, []).append(s)
    out: list[dict] = []
    for key, group in groups.items():
        thetas = [s.theta_bn for s in group if s.theta_bn is not None]
        seconds = [s.duration_s for s in group if s.duration_s is not None]
        tokens = [sum(s.total_tokens) for s in group if s.tokens]
        out.append(
            {
                "question": key,
                "n": len(group),
                "distinct": len({s.key_answer() for s in group}),
                "theta_mean": statistics.fmean(thetas) if thetas else None,
                "theta_std": statistics.stdev(thetas) if len(thetas) > 1 else None,
                "theta_n": len(thetas),
                "alt_frac": sum(s.mentions_alternatives for s in group) / len(group),
                "err_n": sum(1 for s in group if s.errors),
                "turns_mean": statistics.fmean(s.lead_turns for s in group),
                "tokens_mean": statistics.fmean(tokens) if tokens else None,
                "seconds_mean": statistics.fmean(seconds) if seconds else None,
            }
        )
    return out


def format_table(rows: list[dict[str, str]]) -> str:
    """Align a list of same-keyed string dicts into a monospace table."""
    if not rows:
        return "(no rows)"
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    body = ["  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols) for r in rows]
    return "\n".join([line, sep, *body])


def _fmt(x: float | None, spec: str = ".1f") -> str:
    return "-" if x is None else format(x, spec)


def aggregate_rows(agg: list[dict]) -> list[dict[str, str]]:
    """Render `aggregate` output for `format_table`."""
    return [
        {
            "question": str(a["question"])[:40],
            "n": str(a["n"]),
            "distinct": str(a["distinct"]),
            "theta_mean": _fmt(a["theta_mean"]),
            "theta_std": _fmt(a["theta_std"]),
            "theta_n": str(a["theta_n"]),
            "alt": f"{a['alt_frac']:.0%}",
            "err": str(a["err_n"]),
            "turns": _fmt(a["turns_mean"]),
            "tokens": _kilo(int(a["tokens_mean"])) if a["tokens_mean"] else "-",
            "seconds": _fmt(a["seconds_mean"], ".0f"),
        }
        for a in agg
    ]


def git_head() -> str:
    """Short HEAD of the checkout the script lives in; empty when git is unavailable."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def provider_model() -> tuple[str, str]:
    """`(provider, model)` as `settings` resolves them — Azure routes by deployment."""
    from helioai.config import settings

    provider = settings.llm.provider
    cfg = getattr(settings.llm, provider, None)
    model = getattr(cfg, "deployment", "") or getattr(cfg, "model", "") if cfg else ""
    return provider, model


def manifest_entry(
    *,
    label: str,
    question_id: str,
    session_id: str,
    rep: int,
    result: dict,
    head: str = "",
    provider: str = "",
    model: str = "",
    env: dict[str, str] | None = None,
    timestamp: str | None = None,
) -> dict:
    """One manifest record, built from values alone so it can be tested without a run.

    Args:
        label: Free text naming the configuration under test.
        question_id: From the question file.
        session_id: The session the run wrote to.
        rep: 1-based repetition index.
        result: What the runner returned: `seconds`, `n_events`, `last_kind`,
            `error_seen`, `exception`.
        head: `git rev-parse --short HEAD`.
        provider: LLM provider name.
        model: Model or deployment.
        env: Values of `RECORDED_ENV`.
        timestamp: ISO UTC; now when omitted.

    Returns:
        A JSON-serialisable dict.
    """
    return {
        "label": label,
        "git_head": head,
        "provider": provider,
        "model": model,
        "env": env if env is not None else {k: os.environ.get(k, "") for k in RECORDED_ENV},
        "timestamp": timestamp or datetime.now(UTC).isoformat(timespec="seconds"),
        "question_id": question_id,
        "user_id": BENCH_USER,
        "session_id": session_id,
        "rep": rep,
        "seconds": result.get("seconds"),
        "n_events": result.get("n_events"),
        "last_kind": result.get("last_kind"),
        "error_seen": bool(result.get("error_seen")),
        "exception": result.get("exception"),
    }


def append_manifest(path: Path, entry: dict) -> list[dict]:
    """Append one entry to a JSON-list manifest, creating the file when absent.

    Rewritten whole after every run rather than at the end of the batch: a batch that
    dies on question 4 keeps the three sessions it already paid for.

    Args:
        path: Manifest file.
        entry: From `manifest_entry`.

    Returns:
        The full list after the append.
    """
    entries = _loads(path.read_text(encoding="utf-8"), []) if path.exists() else []
    if not isinstance(entries, list):
        raise ValueError(f"{path} is not a JSON list")
    entries.append(entry)
    path.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
    return entries


def mint_session_id(question_id: str) -> str:
    """`bench-<qid>-<8 hex>`; `score` reads the question id back out of it."""
    return f"bench-{question_id}-{uuid.uuid4().hex[:8]}"


def load_questions(path: Path, only: list[str] | None = None) -> list[dict]:
    """Read the question file, optionally keeping only the listed ids."""
    questions = json.loads(path.read_text(encoding="utf-8"))
    if only:
        wanted = set(only)
        unknown = wanted - {q["id"] for q in questions}
        if unknown:
            raise SystemExit(f"unknown question id(s): {sorted(unknown)}")
        questions = [q for q in questions if q["id"] in wanted]
    return questions


def run_batch(
    questions: list[dict],
    *,
    n: int,
    label: str,
    out: Path,
    runner: Callable[[str, str], dict],
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Run every question `n` times, sequentially, appending to the manifest as it goes.

    Sequential on purpose: the sandbox and the speasy cache are shared, and two agents
    downloading the same interval at once is the one thing the harness must not
    measure. An exception in `runner` is recorded in the entry and the batch goes on.

    Args:
        questions: `{"id", "text", ...}` dicts.
        n: Repetitions per question.
        label: Configuration label written to each entry.
        out: Manifest path.
        runner: `(session_id, question_text) -> result dict`; the live one drives
            `stream_chat`, a test passes a stub.
        log: Progress sink, one line per run.

    Returns:
        The entries this batch appended.
    """
    head, (provider, model) = git_head(), _provider_model_safe()
    env = {k: os.environ.get(k, "") for k in RECORDED_ENV}
    appended: list[dict] = []
    for q in questions:
        for rep in range(1, n + 1):
            sid = mint_session_id(q["id"])
            t0 = time.monotonic()
            try:
                result = runner(sid, q["text"])
            except Exception as exc:
                result = {
                    "seconds": time.monotonic() - t0,
                    "n_events": 0,
                    "last_kind": None,
                    "error_seen": True,
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            entry = manifest_entry(
                label=label,
                question_id=q["id"],
                session_id=sid,
                rep=rep,
                result=result,
                head=head,
                provider=provider,
                model=model,
                env=env,
            )
            append_manifest(out, entry)
            appended.append(entry)
            flag = "  ERROR" if entry["error_seen"] else ""
            log(
                f"{q['id']:<22} {rep}/{n}  {sid}  {entry['seconds'] or 0:6.1f}s  "
                f"{entry['n_events']:>4} events  last={entry['last_kind']}{flag}"
            )
    return appended


def _provider_model_safe() -> tuple[str, str]:
    try:
        return provider_model()
    except Exception:
        return "", ""


async def _drive(session_id: str, text: str) -> dict:
    """One question end to end, as `helioai/interfaces/cli.py::_run_query` does it."""
    import helioai.tools.setup  # noqa: F401
    from helioai.core.agent_loop import stream_chat
    from helioai.core.llm.factory import build_llm_client

    llm = build_llm_client()
    n_events = 0
    last_kind = None
    error_seen = False
    t0 = time.monotonic()
    try:
        async for ev in stream_chat(llm, BENCH_USER, session_id, text, restricted=True):
            n_events += 1
            last_kind = ev["event"]
            if last_kind == "error":
                error_seen = True
    finally:
        await llm.aclose()
    return {
        "seconds": time.monotonic() - t0,
        "n_events": n_events,
        "last_kind": last_kind,
        "error_seen": error_seen,
        "exception": None,
    }


def live_runner(session_id: str, text: str) -> dict:
    """The runner `run` uses: a fresh event loop and client per question, as the CLI."""
    return asyncio.run(_drive(session_id, text))


def cmd_run(args: argparse.Namespace) -> int:
    from helioai.logging_config import setup_logging
    from helioai.tools.mcp_client import discover_and_register

    setup_logging("WARNING")
    asyncio.run(discover_and_register())
    questions = load_questions(Path(args.questions), args.only)
    print(f"{len(questions)} question(s) × {args.n} → {args.out}", flush=True)
    run_batch(
        questions,
        n=args.n,
        label=args.label,
        out=Path(args.out),
        runner=live_runner,
        log=lambda s: print(s, flush=True),
    )
    return 0


def open_readonly(path: Path) -> sqlite3.Connection:
    """A read-only connection: `score` must never migrate or touch a live database."""
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _default_db() -> Path:
    from helioai.core.session import DEFAULT_DB

    return DEFAULT_DB


def _parse_session_arg(raw: str) -> tuple[str, str]:
    user_id, sep, session_id = raw.partition(":")
    if not sep or not session_id:
        raise SystemExit(f"--session expects <user_id>:<session_id>, got {raw!r}")
    return user_id, session_id


def score_targets(
    conn: sqlite3.Connection,
    manifest: list[dict] | None,
    sessions: list[tuple[str, str]],
) -> list[SessionScore]:
    """Score every session a manifest and/or explicit `--session` arguments name."""
    scores: list[SessionScore] = []
    for entry in manifest or []:
        s = score_session(conn, entry.get("user_id", BENCH_USER), entry["session_id"])
        s.question_id = entry.get("question_id") or s.question_id
        scores.append(s)
    for user_id, session_id in sessions:
        scores.append(score_session(conn, user_id, session_id))
    return scores


def cmd_score(args: argparse.Namespace) -> int:
    manifest = None
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    sessions = [_parse_session_arg(s) for s in args.session or []]
    if not manifest and not sessions:
        raise SystemExit("nothing to score: pass --manifest and/or --session")
    db = Path(args.db) if args.db else _default_db()
    conn = open_readonly(db)
    try:
        scores = score_targets(conn, manifest, sessions)
    finally:
        conn.close()

    rows = [s.to_row() for s in scores]
    print(format_table(rows))
    agg = aggregate(scores)
    print()
    print(format_table(aggregate_rows(agg)))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if args.json:
        payload = {"sessions": [asdict(s) for s in scores], "aggregate": agg}
        Path(args.json).write_text(
            json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The two subcommands; kept as a function so `--help` is testable."""
    p = argparse.ArgumentParser(
        prog="bench_live.py",
        description=__doc__.split("\n\n", 1)[0],
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="drive the agent on each question × N (spends tokens)")
    r.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    r.add_argument("--n", type=int, default=3, help="repetitions per question")
    r.add_argument("--only", nargs="*", metavar="ID", help="question ids to run")
    r.add_argument("--label", default="", help="name of the configuration under test")
    r.add_argument("--out", default="bench_manifest.json", help="manifest to append to")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("score", help="score sessions from the database, read-only")
    s.add_argument("--manifest", help="manifest written by `run`")
    s.add_argument(
        "--session",
        action="append",
        metavar="USER:SESSION",
        help="score this session too (repeatable); e.g. web:<uuid>",
    )
    s.add_argument("--db", help="sessions.db (default: the configured one)")
    s.add_argument("--csv", help="write the per-session table here")
    s.add_argument("--json", help="write per-session scores and the aggregate here")
    s.set_defaults(func=cmd_score)
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
