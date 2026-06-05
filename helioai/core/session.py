"""Conversation history keyed by (user_id, session_id), persisted to SQLite."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path

from helioai.core.llm.base import Message, ToolCall

_SERVICE_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB = Path(os.environ.get("HELIOAI_SESSION_DB", str(_SERVICE_ROOT / "data" / "sessions.db")))

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
    FOREIGN KEY (user_id, session_id)
        REFERENCES sessions(user_id, session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session_seq
    ON messages(user_id, session_id, seq);
"""

_MIGRATE = "ALTER TABLE sessions ADD COLUMN workspace_dir TEXT"


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
    def __init__(self, db_path: Path = DEFAULT_DB) -> None:
        self._db_path = db_path
        self._cache: dict[SessionKey, list[Message]] = {}
        self._lock = threading.Lock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            try:
                conn.execute(_MIGRATE)
            except Exception:
                pass
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def get_or_create(self, user_id: str, session_id: str) -> list[Message]:
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
                "SELECT role, content, tool_calls, tool_call_id "
                "FROM messages WHERE user_id = ? AND session_id = ? ORDER BY seq",
                (user_id, session_id),
            ).fetchall()
        return [
            Message(
                role=role,
                content=content or "",
                tool_calls=_load_tool_calls(tool_calls),
                tool_call_id=tool_call_id,
            )
            for role, content, tool_calls, tool_call_id in rows
        ]

    def save(self, user_id: str, session_id: str, history: list[Message]) -> None:
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
                "INSERT INTO messages(user_id, session_id, seq, role, content, tool_calls, tool_call_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        user_id,
                        session_id,
                        i,
                        m.role,
                        m.content or "",
                        _dump_tool_calls(m.tool_calls),
                        m.tool_call_id,
                    )
                    for i, m in enumerate(history)
                ],
            )
            conn.commit()

    def reset(self, user_id: str, session_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM sessions WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )
            conn.commit()
        self._cache.pop((user_id, session_id), None)

    def set_workspace_dir(self, user_id: str, session_id: str, workspace_dir: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET workspace_dir = ? WHERE user_id = ? AND session_id = ?",
                (workspace_dir, user_id, session_id),
            )
            conn.commit()

    def get_workspace_dir(self, user_id: str, session_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT workspace_dir FROM sessions WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            ).fetchone()
        return row[0] if row else None

    def all_sessions(self, user_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [r[0] for r in rows]

    def list_summaries(self, user_id: str, limit: int = 50) -> list[dict]:
        from datetime import datetime, timezone

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
                ORDER BY s.updated_at DESC LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        out: list[dict] = []
        for session_id, jd, first_user, n_messages, workspace_dir in rows:
            preview = (first_user or "").strip().replace("\n", " ")
            if len(preview) > 80:
                preview = preview[:77] + "..."
            unix_ts = (jd - 2440587.5) * 86400
            iso = (
                datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            )
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
            cleaned.append(Message(role=m.role, content=m.content, tool_calls=live_tcs))
        elif m.content:
            cleaned.append(Message(role=m.role, content=m.content))
        # else: drop the message entirely (no content, no answered tool_calls)
    return cleaned


store = SessionStore()
