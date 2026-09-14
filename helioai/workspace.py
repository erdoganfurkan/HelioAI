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
    """Return the user owning the current context, or the default user.

    Returns:
        The bound user id, or `DEFAULT_USER` outside any bound context — the CLI
        and the Jupyter magic never bind one, so unowned callers still get a
        real storage home rather than an error.
    """
    return _current_user.get() or DEFAULT_USER


def set_user(user_id: str) -> object:
    """Bind the user that owns storage for the current context.

    A contextvar rather than a global: every task spawned from here inherits the
    binding, which is what keeps a sub-agent writing into the same user's tree as
    the lead that spawned it, while a concurrent web request keeps its own.

    Args:
        user_id: Owner of every path derived until the binding is reset.

    Returns:
        An opaque token to hand back to `reset_user`. Resetting by token rather
        than by re-setting a previous value is what makes nesting safe.
    """
    return _current_user.set(user_id)


def reset_user(token: object) -> None:
    """Restore the user binding that was in force before `set_user`.

    Args:
        token: The value `set_user` returned. A token from another contextvar,
            or one already reset, raises rather than silently rebinding.
    """
    _current_user.reset(token)  # type: ignore[arg-type]


def _users_root() -> Path:
    from helioai.config import settings

    return Path(settings.data_dir) / "users"


def user_home(user: str) -> Path:
    """A user's private storage home: `<data>/users/<user>/`.

    The directory is not created — callers that only need to read must not
    materialise a home for a user that does not exist.

    Args:
        user: Owner id. Not sanitised here; callers that accept it from a
            request pass it through `safe_id` first.

    Returns:
        The path, whether or not anything exists at it.

    Example:
        >>> user_home("cli")
        PosixPath('.../data/users/cli')
    """
    return _users_root() / user


def _root() -> Path:
    p = user_home(current_user()) / "workspace"
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_session(session_id: str) -> object:
    """Bind the session whose workspace the current context writes into.

    Args:
        session_id: Session id, used as a directory name when no label is bound.

    Returns:
        An opaque token for `reset_session`.
    """
    return _current_session.set(session_id)


def reset_session(token: object) -> None:
    """Restore the session binding in force before `set_session`.

    Args:
        token: The value `set_session` returned.
    """
    _current_session.reset(token)  # type: ignore[arg-type]


def set_label(label: str) -> object:
    """Bind the human-readable folder name for the current session.

    Takes precedence over the session id in `get_session_dir`, so a workspace is
    findable by what was asked rather than by a uuid.

    Args:
        label: Directory name, sanitised on use rather than here.

    Returns:
        An opaque token for `reset_label`.
    """
    return _current_label.set(label)


def reset_label(token: object) -> None:
    """Restore the label binding in force before `set_label`.

    Args:
        token: The value `set_label` returned.
    """
    _current_label.reset(token)  # type: ignore[arg-type]


def safe_id(value: str, fallback: str = "session") -> str:
    """Reduce an identifier to something that cannot escape its parent directory.

    Session ids are caller-supplied — a web request body, an MCP client, a CLI
    flag — and end up as path components here, in the export filename, and in the
    `rmtree` behind `DELETE /api/sessions/{id}`. Everything the project mints is a
    uuid4, so stripping to `[A-Za-z0-9_-]` is lossless in practice and turns
    `../..` into the fallback rather than a parent directory.

    Args:
        value: Caller-supplied identifier, trusted for nothing.
        fallback: Returned when nothing survives the filter, so the result is
            never the empty string — which would resolve to the parent directory.

    Returns:
        At most 64 characters of `[A-Za-z0-9_-]`, or `fallback`.

    Example:
        >>> safe_id("../../etc/passwd"), safe_id("sess-abc-123456")
        ('etcpasswd', 'sess-abc-123456')
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", value)[:64]
    return cleaned or fallback


def make_session_label(first_message: str, session_id: str) -> str:
    """Build a human-readable slug for the session workspace folder.

    The session id suffix is what keeps two identical questions from sharing one
    directory; the words are only there so a human can find it.

    Args:
        first_message: The question that opened the session.
        session_id: Session id, truncated to six characters as a discriminator.

    Returns:
        A slug of at most four words plus the id suffix.

    Example:
        >>> make_session_label("Plot IMF Bz from ACE", "abc123def456")
        'plot-imf-bz-from_abc123'
    """
    words = re.sub(r"[^a-z0-9\s]", "", first_message.lower().strip()).split()
    slug = "-".join(words[:4]) if words else "session"
    return f"{slug[:25]}_{safe_id(session_id)[:6]}"


def session_dir_for(user: str, session_id: str, label: str | None = None) -> Path:
    """The workspace directory of a session, from its ids rather than from the ambience.

    The same rule `get_session_dir` applies to the bound contextvars — the label when
    the session has one, the id otherwise — so a `RunContext` built from ids and a
    caller reading the contextvars land in the same directory.

    Args:
        user: Owner of the session.
        session_id: The conversation.
        label: Its human-readable directory name, when already minted.

    Returns:
        The directory, created if missing.
    """
    root = user_home(user) / "workspace"
    d = root / safe_id(label or session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_session_dir() -> Path:
    """Return the workspace directory for the current session.

    Prefers the bound label, falls back to the bound session id, and lands in a
    temporary directory when neither is bound — an unbound caller still gets a
    writable place rather than an exception, because `run_python` must not fail
    for want of a session.

    Returns:
        The directory, created if missing.
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

    Derived from what is on disk rather than from a counter, so the numbering
    survives a restart and stays right when a session is resumed.

    Args:
        session_dir: Directory holding the `code_N.py` files of past runs.

    Returns:
        `max(N) + 1`, or 0 when the session has run nothing yet.
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
    """The current session directory as a string, for the sandbox fallback.

    Returns:
        `get_session_dir()` as a plain string — the fallback path builds a bwrap
        argument list, which takes no `Path`.
    """
    return str(get_session_dir())


def is_under_workspace(path: str | Path) -> bool:
    """True if path is safely under the per-user storage root (no traversal).

    `is_relative_to` rather than a string prefix: comparing against `str(root) + "/"`
    hard-coded the POSIX separator, so on Windows the check never matched and `/figure`
    and `/code` returned 404 for every legitimate path. Fail-closed, so it was a dead
    web UI rather than a hole — but dead all the same.

    Args:
        path: Candidate path, resolved before comparison so symlinks and `..`
            cannot point outside from within.

    Returns:
        True when the resolved path sits under the users root. False on any
        resolution error, which keeps an unreadable path from being served.
    """
    try:
        p = Path(path).resolve()
        return p.is_relative_to(_users_root().resolve())
    except (ValueError, OSError):
        return False


def cleanup_old_runs(ttl_seconds: int | None = None) -> int:
    """Purge session directories older than the TTL, for every user.

    Called on CLI startup, when the notebook magic loads and when the MCP server
    starts, and every hour by `cleanup_periodically` under the web server — the one
    process that never restarts, and so never reached this until it did. Removal
    failures are ignored rather than raised: housekeeping must not stop a user from
    asking a question. A user's home (profile, catalogs, speasy seed) is never touched;
    only what is under `workspace/`.

    Args:
        ttl_seconds: Age above which a session directory is deleted, measured on
            its mtime. Defaults to `settings.workspace.ttl_seconds`.

    Returns:
        How many directories were removed.
    """
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


async def cleanup_periodically(period_seconds: float = 3600.0) -> None:
    """Run `cleanup_old_runs` every `period_seconds` until the task is cancelled.

    The web server used to clean up once, at startup, and then run for weeks: a demo
    machine had 76 session directories of 269 MB each, every one of them past the
    TTL. The sweep runs in a worker thread so a slow disk never stalls a stream, and
    a failing sweep is logged and retried at the next tick rather than ending the task.

    Args:
        period_seconds: Time between sweeps; the first sweep happens after one period,
            since the caller already swept at startup.
    """
    import asyncio

    from helioai.logging_config import get_logger

    while True:
        await asyncio.sleep(period_seconds)
        try:
            removed = await asyncio.to_thread(cleanup_old_runs)
            if removed:
                get_logger(__name__).info("workspace_cleanup", removed=removed)
        except Exception:
            get_logger(__name__).warning("workspace_cleanup_failed", exc_info=True)
