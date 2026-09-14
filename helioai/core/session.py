"""Conversation history keyed by (user_id, session_id), persisted to SQLite."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC
from pathlib import Path

from helioai.config import settings
from helioai.core.llm.base import Message, ToolCall

DEFAULT_DB = Path(os.environ.get("HELIOAI_SESSION_DB", str(settings.data_dir / "sessions.db")))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    user_id       TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    created_at    REAL NOT NULL DEFAULT (julianday('now')),
    updated_at    REAL NOT NULL DEFAULT (julianday('now')),
    workspace_dir TEXT,
    PRIMARY KEY (user_id, session_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    seq          INTEGER NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT NOT NULL DEFAULT '',
    tool_calls   TEXT,
    tool_call_id TEXT,
    origin       TEXT,
    FOREIGN KEY (user_id, session_id)
        REFERENCES sessions(user_id, session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session_seq
    ON messages(user_id, session_id, seq);
"""

# Additive only, each one tried on its own: a column that already exists raises and
# is skipped, so an old database gains what it lacks and a new one is left alone.
_MIGRATIONS = (
    "ALTER TABLE sessions ADD COLUMN workspace_dir TEXT",
    "ALTER TABLE messages ADD COLUMN origin TEXT",
)


def _dump_tool_calls(tcs: list[ToolCall] | None) -> str | None:
    if not tcs:
        return None
    return json.dumps([{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tcs])


def _load_tool_calls(raw: str | None) -> list[ToolCall] | None:
    if not raw:
        return None
    data = json.loads(raw)
    return [ToolCall(id=d["id"], name=d["name"], arguments=d.get("arguments") or {}) for d in data]


SessionKey = tuple[str, str]


class SessionStore:
    """Conversation history keyed by (user_id, session_id), persisted to SQLite.

    Histories are cached in memory per key and written back whole on `save`.
    Tests use a real database on `tmp_path` rather than a mock: a mocked store
    passed happily through a schema migration that broke production.

    Example:
        >>> store = SessionStore(tmp_path / "sessions.db")
        >>> history = store.get_or_create("cli", "sess-1")   # [] on first call
        >>> history.append(Message(role="user", content="hello"))
        >>> store.save("cli", "sess-1", history)
        >>> store.get_or_create("cli", "sess-1")[0].role
        'user'
    """

    def __init__(self, db_path: Path = DEFAULT_DB) -> None:
        self._db_path = db_path
        self._cache: dict[SessionKey, list[Message]] = {}
        self._lock = threading.Lock()
        self._turn_locks: dict[SessionKey, asyncio.Lock] = {}
        # Nothing touches the disk until the first query: the module-level `store`
        # is built at import, and creating directories on `import helioai` is the
        # kind of side effect a packager, a linter run or a docs build trips over.
        self._schema_ready = False
        # Its own lock: `save` already holds `_lock` when it connects, and `_lock` is
        # not re-entrant — taking it again here would deadlock the first save.
        self._schema_lock = threading.Lock()

    def _init_schema(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        try:
            conn.executescript(_SCHEMA)
            for statement in _MIGRATIONS:
                try:
                    conn.execute(statement)
                except Exception:
                    pass
            conn.commit()
        finally:
            conn.close()
        self._schema_ready = True

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if not self._schema_ready:
            with self._schema_lock:
                if not self._schema_ready:
                    self._init_schema()
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        try:
            yield conn
        finally:
            conn.close()

    def turn_lock(self, user_id: str, session_id: str) -> asyncio.Lock:
        """The lock a caller must hold for the whole of one conversational turn.

        `get_or_create` hands every caller the same list, and a turn is a
        read-modify-write of it that spans several awaits: without this, two turns on
        one session — two browser tabs, two MCP calls — interleave their appends and
        `save` persists the mix. `_lock` only serialises the SQL, never the turn.

        One `asyncio.Lock` per key, created on first use. A lock in Python ≥ 3.10 binds
        to an event loop only on its first *contended* acquisition, so the CLI and the
        Jupyter magic — a fresh `asyncio.run` per question, never two turns on one
        session at once — reuse it across loops safely, while the web and MCP servers
        run one loop. If that assumption ever breaks, asyncio raises a `RuntimeError`
        naming the loop mismatch instead of silently corrupting anything.

        Args:
            user_id: Owner of the session.
            session_id: Session identifier.

        Returns:
            The same lock object for the same key, for the life of the store.
        """
        key: SessionKey = (user_id, session_id)
        with self._lock:
            lock = self._turn_locks.get(key)
            if lock is None:
                lock = self._turn_locks[key] = asyncio.Lock()
            return lock

    def is_busy(self, user_id: str, session_id: str) -> bool:
        """Whether a turn is currently running for this session.

        Read without taking the lock — this is the web layer's fast refusal (409), not
        a guarantee; the guarantee is `turn_lock` itself.
        """
        lock = self._turn_locks.get((user_id, session_id))
        return bool(lock and lock.locked())

    def get_or_create(self, user_id: str, session_id: str) -> list[Message]:
        """Return the cached history for a session, loading it from disk if needed."""
        key: SessionKey = (user_id, session_id)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            history = self._load(user_id, session_id)
            self._cache[key] = history
            return history

    def _load(self, user_id: str, session_id: str) -> list[Message]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, tool_calls, tool_call_id, origin "
                "FROM messages WHERE user_id = ? AND session_id = ? ORDER BY seq",
                (user_id, session_id),
            ).fetchall()
        return [
            Message(
                role=role,
                content=content or "",
                tool_calls=_load_tool_calls(tool_calls),
                tool_call_id=tool_call_id,
                origin=origin,
            )
            for role, content, tool_calls, tool_call_id, origin in rows
        ]

    def save(self, user_id: str, session_id: str, history: list[Message]) -> None:
        """Replace a session's stored history.

        Args:
            user_id: Owner of the session.
            session_id: Session identifier.
            history: Full message list; it replaces whatever was stored.
        """
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions(user_id, session_id) VALUES(?, ?) "
                "ON CONFLICT(user_id, session_id) DO UPDATE SET updated_at = julianday('now')",
                (user_id, session_id),
            )
            conn.execute(
                "DELETE FROM messages WHERE user_id = ? AND session_id = ?", (user_id, session_id)
            )
            conn.executemany(
                "INSERT INTO messages(user_id, session_id, seq, role, content, tool_calls, "
                "tool_call_id, origin) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        user_id,
                        session_id,
                        i,
                        m.role,
                        m.content or "",
                        _dump_tool_calls(m.tool_calls),
                        m.tool_call_id,
                        m.origin,
                    )
                    for i, m in enumerate(history)
                ],
            )
            conn.commit()

    def reset(self, user_id: str, session_id: str) -> None:
        """Delete a session and its messages, and drop it from the cache."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM sessions WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )
            conn.commit()
        self._cache.pop((user_id, session_id), None)
        self._turn_locks.pop((user_id, session_id), None)

    def set_workspace_dir(self, user_id: str, session_id: str, workspace_dir: str) -> None:
        """Record which workspace directory a session's artifacts live in."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET workspace_dir = ? WHERE user_id = ? AND session_id = ?",
                (workspace_dir, user_id, session_id),
            )
            conn.commit()

    def get_workspace_dir(self, user_id: str, session_id: str) -> str | None:
        """Return a session's workspace directory label, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT workspace_dir FROM sessions WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            ).fetchone()
        return row[0] if row else None

    def workspace_dirs(self, user_id: str) -> set[str]:
        """All workspace dir labels owned by a user (for path-ownership checks)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT workspace_dir FROM sessions "
                "WHERE user_id = ? AND workspace_dir IS NOT NULL",
                (user_id,),
            ).fetchall()
        return {r[0] for r in rows}

    def all_sessions(self, user_id: str) -> list[str]:
        """Return a user's session ids, most recently updated first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id FROM sessions WHERE user_id = ? "
                "ORDER BY updated_at DESC, rowid DESC",
                (user_id,),
            ).fetchall()
        return [r[0] for r in rows]

    def list_summaries(self, user_id: str, limit: int = 50) -> list[dict]:
        """Summarise a user's recent sessions for the history view.

        Args:
            user_id: Owner of the sessions.
            limit: Maximum number of sessions to return.

        Returns:
            Dicts with session_id, updated_at, first_message, n_messages and
            workspace_dir, most recent first.
        """
        from datetime import datetime

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.session_id, s.updated_at AS jd,
                       (SELECT content FROM messages WHERE user_id = s.user_id
                          AND session_id = s.session_id AND role = 'user'
                          ORDER BY seq LIMIT 1) AS first_user,
                       (SELECT COUNT(*) FROM messages WHERE user_id = s.user_id
                          AND session_id = s.session_id) AS n_messages,
                       s.workspace_dir
                FROM sessions s
                WHERE s.user_id = ?
                ORDER BY s.updated_at DESC, s.rowid DESC LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        out: list[dict] = []
        for session_id, jd, first_user, n_messages, workspace_dir in rows:
            preview = (first_user or "").strip().replace("\n", " ")
            if len(preview) > 80:
                preview = preview[:77] + "..."
            unix_ts = (jd - 2440587.5) * 86400
            iso = datetime.fromtimestamp(unix_ts, tz=UTC).isoformat().replace("+00:00", "Z")
            out.append(
                {
                    "session_id": session_id,
                    "first_message": preview,
                    "n_messages": n_messages,
                    "updated_at": iso,
                    "workspace_dir": workspace_dir,
                }
            )
        return out


def strip_orphan_tool_calls(history: list[Message]) -> list[Message]:
    """Remove assistant tool_calls that have no matching tool response.

    An interrupted generation (e.g. client disconnect mid-tool) can leave an
    assistant message with tool_calls but no corresponding tool messages in the
    history.  Sending such a sequence to the LLM API causes a 400 error.

    For each orphaned tool_call id:
    - If the assistant message has content too, keep the message but drop the
      orphaned tool_calls list entry (or clear it entirely if all are orphaned).
    - If the assistant message has no content and all its tool_calls are
      orphaned, drop the message entirely.

    Args:
        history: Messages in order, as loaded from the session store.

    Returns:
        A new list; the input is not modified. Messages without tool calls pass
        through untouched, so a clean history is returned equal to its input.
    """
    answered: set[str] = {m.tool_call_id for m in history if m.tool_call_id}
    cleaned: list[Message] = []
    for m in history:
        if not m.tool_calls:
            cleaned.append(m)
            continue
        live_tcs = [tc for tc in m.tool_calls if tc.id in answered]
        if len(live_tcs) == len(m.tool_calls):
            cleaned.append(m)
        elif live_tcs:
            cleaned.append(replace(m, tool_calls=live_tcs))
        elif m.content:
            cleaned.append(replace(m, tool_calls=None))
        # else: drop the message entirely (no content, no answered tool_calls)
    return cleaned


store = SessionStore()
