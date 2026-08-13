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
from functools import lru_cache
from pathlib import Path

_current_session: ContextVar[str | None] = ContextVar("helioai_current_session", default=None)
_current_label: ContextVar[str | None] = ContextVar("helioai_workspace_label", default=None)
_current_user: ContextVar[str | None] = ContextVar("helioai_current_user", default=None)

DEFAULT_USER = "web"


def current_user() -> str:
    """Return the user owning the current context, or the default user."""
    return _current_user.get() or DEFAULT_USER


def set_user(user_id: str) -> object:
    """Bind the user contextvar. Returns token for later reset."""
    return _current_user.set(user_id)


def reset_user(token: object) -> None:
    """Restore the user contextvar from a token returned by `set_user`."""
    _current_user.reset(token)  # type: ignore[arg-type]


def _users_root() -> Path:
    from helioai.config import settings

    return Path(settings.data_dir) / "users"


def user_home(user: str) -> Path:
    """A user's private storage home: <data>/users/<user>/ (not created here)."""
    return _users_root() / user


def _root() -> Path:
    p = user_home(current_user()) / "workspace"
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_session(session_id: str) -> object:
    """Bind the session contextvar. Returns token for later reset."""
    return _current_session.set(session_id)


def reset_session(token: object) -> None:
    """Restore the session contextvar from a token returned by `set_session`."""
    _current_session.reset(token)  # type: ignore[arg-type]


def set_label(label: str) -> object:
    """Bind the workspace label contextvar. Returns token for later reset."""
    return _current_label.set(label)


def reset_label(token: object) -> None:
    """Restore the label contextvar from a token returned by `set_label`."""
    _current_label.reset(token)  # type: ignore[arg-type]


def safe_id(value: str, fallback: str = "session") -> str:
    """Reduce an identifier to something that cannot escape its parent directory.

    Session ids are caller-supplied — a web request body, an MCP client, a CLI
    flag — and end up as path components here, in the export filename, and in the
    `rmtree` behind `DELETE /api/sessions/{id}`. Everything the project mints is a
    uuid4, so stripping to `[A-Za-z0-9_-]` is lossless in practice and turns
    `../..` into the fallback rather than a parent directory.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", value)[:64]
    return cleaned or fallback


def make_session_label(first_message: str, session_id: str) -> str:
    """Build a human-readable slug for the session workspace folder.

    Example: "Plot IMF Bz from ACE" + session abc123... → "plot-imf-bz-from_abc123"
    """
    words = re.sub(r"[^a-z0-9\s]", "", first_message.lower().strip()).split()
    slug = "-".join(words[:4]) if words else "session"
    return f"{slug[:25]}_{safe_id(session_id)[:6]}"


def get_session_dir() -> Path:
    """Return the workspace directory for the current session.

    Uses _current_label if set, falls back to _current_session UUID, then tmpdir.
    Creates the directory if it does not exist.
    """
    label = _current_label.get()
    if label:
        d = _root() / safe_id(label)
        d.mkdir(parents=True, exist_ok=True)
        return d
    session_id = _current_session.get()
    if session_id:
        d = _root() / safe_id(session_id)
        d.mkdir(parents=True, exist_ok=True)
        return d
    return _no_session_dir()


@lru_cache(maxsize=1)
def _no_session_dir() -> Path:
    """One scratch dir per process for calls made outside any session.

    Minting a fresh mkdtemp per call left ~200 orphaned directories per notebook
    run — nothing ever deleted them — and it also meant two calls in the same
    no-session context wrote to two different places.

    ponytail: leaked once per process instead of never; a session always has its
    own directory, so this only catches the fallback path.
    """
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
    """True if path is safely under the per-user storage root (no traversal).

    `is_relative_to` rather than a string prefix: comparing against `str(root) + "/"`
    hard-coded the POSIX separator, so on Windows the check never matched and `/figure`
    and `/code` returned 404 for every legitimate path. Fail-closed, so it was a dead
    web UI rather than a hole — but dead all the same.
    """
    try:
        p = Path(path).resolve()
        return p.is_relative_to(_users_root().resolve())
    except (ValueError, OSError):
        return False


def cleanup_old_runs(ttl_seconds: int | None = None) -> int:
    """Purge session dirs older than ttl_seconds across all users. Returns count removed."""
    from helioai.config import settings

    if ttl_seconds is None:
        ttl_seconds = settings.workspace.ttl_seconds
    users_root = _users_root()
    if not users_root.exists():
        return 0
    cutoff = time.time() - ttl_seconds
    removed = 0
    for home in users_root.iterdir():
        ws = home / "workspace"
        if not ws.is_dir():
            continue
        for session_dir in ws.iterdir():
            if session_dir.is_dir() and session_dir.stat().st_mtime < cutoff:
                shutil.rmtree(session_dir, ignore_errors=True)
                removed += 1
    return removed
