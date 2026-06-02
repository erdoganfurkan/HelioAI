"""Workspace — stable output directory for sandbox figures and data.

Figures go to  workspace/<session_label>/fig_N_M.png
Code files go to workspace/<session_label>/code_N.py

The session label is a human-readable slug derived from the first user message,
propagated via a contextvar set by stream_chat at the start of each request.
"""

from __future__ import annotations

import re
import shutil
import time
from contextvars import ContextVar
from pathlib import Path

_current_session: ContextVar[str | None] = ContextVar("helioai_current_session", default=None)
_current_label: ContextVar[str | None] = ContextVar("helioai_workspace_label", default=None)


def _root() -> Path:
    from helioai.config import settings
    p = Path(settings.workspace.workspace_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_session(session_id: str) -> object:
    """Bind the session contextvar. Returns token for later reset."""
    return _current_session.set(session_id)


def reset_session(token: object) -> None:
    _current_session.reset(token)  # type: ignore[arg-type]


def set_label(label: str) -> object:
    """Bind the workspace label contextvar. Returns token for later reset."""
    return _current_label.set(label)


def reset_label(token: object) -> None:
    _current_label.reset(token)  # type: ignore[arg-type]


def make_session_label(first_message: str, session_id: str) -> str:
    """Build a human-readable slug for the session workspace folder.

    Example: "Plot IMF Bz from ACE" + session abc123... → "plot-imf-bz-from_abc123"
    """
    words = re.sub(r"[^a-z0-9\s]", "", first_message.lower().strip()).split()
    slug = "-".join(words[:4]) if words else "session"
    return f"{slug[:25]}_{session_id[:6]}"


def get_session_dir() -> Path:
    """Return the workspace directory for the current session.

    Uses _current_label if set, falls back to _current_session UUID, then tmpdir.
    Creates the directory if it does not exist.
    """
    label = _current_label.get()
    if label:
        d = _root() / label
        d.mkdir(parents=True, exist_ok=True)
        return d
    session_id = _current_session.get()
    if session_id:
        d = _root() / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d
    import tempfile
    return Path(tempfile.mkdtemp(prefix="helioai_"))


def get_next_run_idx(session_dir: Path) -> int:
    """Return the next available run index for a session directory.

    Scans for code_N.py files and returns max(N)+1, or 0 if none exist.
    """
    existing = list(session_dir.glob("code_*.py"))
    if not existing:
        return 0
    indices = []
    for p in existing:
        parts = p.stem.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            indices.append(int(parts[1]))
    return max(indices) + 1 if indices else 0


def get_run_dir_for_sandbox() -> str:
    """Backward-compat: return session dir path as string (used by sandbox fallback)."""
    return str(get_session_dir())


def is_under_workspace(path: str | Path) -> bool:
    """True if path is safely under the workspace root (no path traversal)."""
    try:
        p = Path(path).resolve()
        root = _root().resolve()
        return str(p).startswith(str(root) + "/") or p == root
    except (ValueError, OSError):
        return False


def cleanup_old_runs(ttl_seconds: int | None = None) -> int:
    """Purge session dirs older than ttl_seconds. Returns count removed."""
    from helioai.config import settings
    if ttl_seconds is None:
        ttl_seconds = settings.workspace.ttl_seconds
    root = _root()
    cutoff = time.time() - ttl_seconds
    removed = 0
    for session_dir in root.iterdir():
        if session_dir.is_dir() and session_dir.stat().st_mtime < cutoff:
            shutil.rmtree(session_dir, ignore_errors=True)
            removed += 1
    return removed
