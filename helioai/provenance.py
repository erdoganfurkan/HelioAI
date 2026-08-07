"""Session provenance ledger — what was actually *computed*, as opposed to written.

Layout:  <session_dir>/data/provenance.json   — {"values": [entry, ...]}

Every `export()` made inside a `run_python` run is appended here with the script that
produced it and the agent that ran it. Nothing else in HelioAI distinguishes a number
that came out of a computation from a number the model wrote down: the sandbox exports
travel up as a tool result, become an `exports` artifact, and are then gone. A published
value with no entry in this ledger has no origin — that is the whole question the file
exists to answer.

Like the datastore, all I/O errors are swallowed: a ledger must never break a calculation.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

log = logging.getLogger(__name__)

DATA_SUBDIR = "data"
LEDGER_FILE = "provenance.json"

_RUN_IDX_RE = re.compile(r"code_(\d+)\.py$")


def _ledger_path(session_dir: Path) -> Path:
    return session_dir / DATA_SUBDIR / LEDGER_FILE


def _read_ledger_file(session_dir: Path) -> dict:
    f = _ledger_path(session_dir)
    if f.exists():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("values"), list):
                return data
        except Exception:
            pass
    return {"values": []}


def read_ledger(session_dir: Path) -> dict:
    """Return the provenance ledger for a session directory.

    Same contract as `datastore.read_manifest`: an absent or corrupt file reads as an
    empty ledger rather than raising, so callers never need a try/except.
    """
    return _read_ledger_file(Path(session_dir))


def record(
    values: dict,
    *,
    code_path: str,
    agent: str | None = None,
    task_id: str | None = None,
    turn: int | None = None,
) -> None:
    """Append the exports of one sandbox run to that session's ledger.

    `code_path` doubles as the session locator — the sandbox writes `code_<n>.py` at the
    root of the session directory — which keeps this callable from `tool_exec` without
    reaching for the workspace contextvars a second time.
    """
    if not values or not code_path:
        return
    try:
        code = Path(code_path)
        session_dir = code.parent
        m = _RUN_IDX_RE.search(code.name)
        run_idx = int(m.group(1)) if m else None
        created = str(int(time.time()))

        ledger = _read_ledger_file(session_dir)
        for name, stats in values.items():
            if not isinstance(stats, dict) or stats.get("error"):
                continue
            ledger["values"].append(
                {
                    "name": name,
                    "mean": stats.get("mean"),
                    "min": stats.get("min"),
                    "max": stats.get("max"),
                    "std": stats.get("std"),
                    "units": stats.get("units", ""),
                    "code_path": str(code),
                    "run_idx": run_idx,
                    "agent": agent,
                    "task_id": task_id,
                    "turn": turn,
                    "created": created,
                }
            )
        (session_dir / DATA_SUBDIR).mkdir(parents=True, exist_ok=True)
        _ledger_path(session_dir).write_text(
            json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        log.debug("provenance record failed: %s", e)


def find_value(session_dir: Path, name: str) -> dict | None:
    """Most recent ledger entry exported under `name`, or None."""
    for entry in reversed(read_ledger(session_dir)["values"]):
        if entry.get("name") == name:
            return entry
    return None


def match_number(session_dir: Path, x: float, rtol: float = 1e-3) -> list[dict]:
    """Ledger entries whose mean/min/max/std is within `rtol` of `x`.

    Any of the four statistics counts: a reply quoting "peaked at 14.5 nT" is sourced by
    the max of an exported array just as much as by its mean.
    """
    hits = []
    for entry in read_ledger(session_dir)["values"]:
        for key in ("mean", "min", "max", "std"):
            v = entry.get(key)
            if isinstance(v, (int, float)) and abs(v - x) <= rtol * max(abs(x), abs(v), 1e-12):
                hits.append(entry)
                break
    return hits
