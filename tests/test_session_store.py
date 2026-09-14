"""Tests for helioai.core.session.SessionStore."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from helioai.core.llm.base import Message, ToolCall
from helioai.core.session import SessionStore, strip_orphan_tool_calls


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "sessions.db"


def test_get_or_create_returns_same_list(db: Path) -> None:
    s = SessionStore(db)
    h1 = s.get_or_create("alice", "t1")
    h2 = s.get_or_create("alice", "t1")
    assert h1 is h2


def test_connect_closes_connection_after_use(db: Path, monkeypatch) -> None:
    """Real bug: `with self._connect() as conn` only relies on sqlite3.Connection's
    own context manager, which commits/rolls back but never calls close() — every
    call leaked a connection."""
    closed: list[bool] = []

    class _TrackedConnection(sqlite3.Connection):
        def close(self):
            closed.append(True)
            super().close()

    real_connect = sqlite3.connect

    def spy_connect(*args, **kwargs):
        kwargs["factory"] = _TrackedConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", spy_connect)

    s = SessionStore(db)
    s.save("alice", "t1", [Message(role="user", content="hi")])

    assert closed, "SessionStore must explicitly close() its sqlite3 connections"


def test_save_and_reload_round_trips_all_fields(db: Path) -> None:
    s1 = SessionStore(db)
    h = s1.get_or_create("alice", "t1")
    h.append(Message(role="user", content="hello"))
    h.append(
        Message(
            role="assistant",
            tool_calls=[ToolCall(id="search::1", name="search_parameters", arguments={"q": "imf"})],
        )
    )
    h.append(Message(role="tool", tool_call_id="search::1", content='{"hits":3}'))
    h.append(Message(role="assistant", content="Found 3 results."))
    s1.save("alice", "t1", h)

    s2 = SessionStore(db)
    h2 = s2.get_or_create("alice", "t1")
    assert len(h2) == 4
    assert h2[0].role == "user" and h2[0].content == "hello"
    assert h2[1].role == "assistant" and h2[1].tool_calls
    assert h2[1].tool_calls[0].name == "search_parameters"
    assert h2[1].tool_calls[0].arguments == {"q": "imf"}
    assert h2[2].role == "tool" and h2[2].tool_call_id == "search::1"
    assert h2[3].content == "Found 3 results."


def test_save_replaces_not_appends(db: Path) -> None:
    s = SessionStore(db)
    h = s.get_or_create("alice", "t1")
    h.append(Message(role="user", content="first"))
    s.save("alice", "t1", h)
    s.save("alice", "t1", h)
    s2 = SessionStore(db)
    assert len(s2.get_or_create("alice", "t1")) == 1


def test_reset_clears_session(db: Path) -> None:
    s = SessionStore(db)
    h = s.get_or_create("alice", "t1")
    h.append(Message(role="user", content="hi"))
    s.save("alice", "t1", h)
    assert "t1" in s.all_sessions("alice")

    s.reset("alice", "t1")
    assert "t1" not in s.all_sessions("alice")
    assert s.get_or_create("alice", "t1") == []


def test_isolated_sessions_same_user(db: Path) -> None:
    s = SessionStore(db)
    a = s.get_or_create("alice", "a")
    b = s.get_or_create("alice", "b")
    a.append(Message(role="user", content="a1"))
    b.append(Message(role="user", content="b1"))
    s.save("alice", "a", a)
    s.save("alice", "b", b)

    s2 = SessionStore(db)
    assert s2.get_or_create("alice", "a")[0].content == "a1"
    assert s2.get_or_create("alice", "b")[0].content == "b1"


def test_user_isolation_same_session_id(db: Path) -> None:
    s = SessionStore(db)
    ha = s.get_or_create("alice", "shared")
    hb = s.get_or_create("bob", "shared")
    assert ha is not hb

    ha.append(Message(role="user", content="alice msg"))
    hb.append(Message(role="user", content="bob msg"))
    s.save("alice", "shared", ha)
    s.save("bob", "shared", hb)

    s2 = SessionStore(db)
    assert s2.get_or_create("alice", "shared")[0].content == "alice msg"
    assert s2.get_or_create("bob", "shared")[0].content == "bob msg"

    s2.reset("alice", "shared")
    assert s2.all_sessions("alice") == []
    assert s2.all_sessions("bob") == ["shared"]


def test_list_summaries_orders_by_recency(db: Path) -> None:
    s = SessionStore(db)
    for sid in ("old", "newer", "newest"):
        h = s.get_or_create("alice", sid)
        h.append(Message(role="user", content=f"hello from {sid}"))
        s.save("alice", sid, h)

    ids = [r["session_id"] for r in s.list_summaries("alice")]
    assert ids[0] == "newest"
    assert set(ids) == {"old", "newer", "newest"}


def test_list_summaries_preview_uses_first_user_message(db: Path) -> None:
    s = SessionStore(db)
    h = s.get_or_create("alice", "t1")
    h.append(Message(role="user", content="What is the IMF on Jan 1 2024?"))
    h.append(Message(role="assistant", content="Let me check."))
    h.append(Message(role="user", content="(follow-up)"))
    s.save("alice", "t1", h)

    row = s.list_summaries("alice")[0]
    assert row["first_message"] == "What is the IMF on Jan 1 2024?"
    assert row["n_messages"] == 3


def test_list_summaries_truncates_long_preview(db: Path) -> None:
    s = SessionStore(db)
    h = s.get_or_create("alice", "t1")
    h.append(Message(role="user", content="X" * 200))
    s.save("alice", "t1", h)
    row = s.list_summaries("alice")[0]
    assert len(row["first_message"]) == 80
    assert row["first_message"].endswith("...")


def test_list_summaries_strips_newlines(db: Path) -> None:
    s = SessionStore(db)
    h = s.get_or_create("alice", "t1")
    h.append(Message(role="user", content="first line\nsecond line"))
    s.save("alice", "t1", h)
    assert "\n" not in s.list_summaries("alice")[0]["first_message"]


def test_list_summaries_respects_limit(db: Path) -> None:
    s = SessionStore(db)
    for i in range(5):
        h = s.get_or_create("alice", f"s{i}")
        h.append(Message(role="user", content=f"msg {i}"))
        s.save("alice", f"s{i}", h)
    assert len(s.list_summaries("alice", limit=3)) == 3


# ── strip_orphan_tool_calls ──────────────────────────────────────────────────


def test_strip_clean_history_unchanged() -> None:
    history = [
        Message(role="user", content="hello"),
        Message(role="assistant", tool_calls=[ToolCall(id="tc1", name="search", arguments={})]),
        Message(role="tool", tool_call_id="tc1", content="result"),
        Message(role="assistant", content="Done."),
    ]
    result = strip_orphan_tool_calls(history)
    assert len(result) == 4
    assert result[1].tool_calls is not None


def test_strip_drops_fully_orphaned_assistant_message() -> None:
    history = [
        Message(role="user", content="hi"),
        Message(role="assistant", tool_calls=[ToolCall(id="orphan", name="search", arguments={})]),
        # No tool response — simulates cancelled generation
    ]
    result = strip_orphan_tool_calls(history)
    assert len(result) == 1
    assert result[0].role == "user"


def test_strip_keeps_content_when_orphaned_tool_calls() -> None:
    history = [
        Message(
            role="assistant",
            content="I'll search for that.",
            tool_calls=[ToolCall(id="orphan", name="search", arguments={})],
        ),
    ]
    result = strip_orphan_tool_calls(history)
    assert len(result) == 1
    assert result[0].content == "I'll search for that."
    assert not result[0].tool_calls


def test_strip_partial_orphan_keeps_answered() -> None:
    history = [
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="tc1", name="search", arguments={}),
                ToolCall(id="orphan", name="get_data", arguments={}),
            ],
        ),
        Message(role="tool", tool_call_id="tc1", content="result"),
    ]
    result = strip_orphan_tool_calls(history)
    assert len(result) == 2
    assert len(result[0].tool_calls) == 1
    assert result[0].tool_calls[0].id == "tc1"


def test_turn_lock_is_one_object_per_session(db: Path) -> None:
    """The lock only serialises anything if every caller for a key gets the same one."""
    store = SessionStore(db)
    assert store.turn_lock("u", "s1") is store.turn_lock("u", "s1")
    assert store.turn_lock("u", "s1") is not store.turn_lock("u", "s2")
    assert store.turn_lock("u", "s1") is not store.turn_lock("v", "s1")


async def test_is_busy_reflects_a_held_turn_lock(db: Path) -> None:
    store = SessionStore(db)
    assert not store.is_busy("u", "s1")
    async with store.turn_lock("u", "s1"):
        assert store.is_busy("u", "s1")
        assert not store.is_busy("u", "s2")
    assert not store.is_busy("u", "s1")


def test_reset_forgets_the_turn_lock(db: Path) -> None:
    store = SessionStore(db)
    lock = store.turn_lock("u", "s1")
    store.reset("u", "s1")
    assert store.turn_lock("u", "s1") is not lock


def test_origin_round_trips(db: Path) -> None:
    """`origin` is what tells an injected correction from the user's own question once
    the history is read back; losing it on save would re-credit the person with it."""
    store = SessionStore(db)
    h = store.get_or_create("u", "s")
    h.append(Message(role="user", content="plot Bz"))
    h.append(Message(role="assistant", content="Use cda/BOGUS."))
    h.append(Message(role="user", content="⚠️ AUTOMATED CORRECTION …", origin="correction"))
    store.save("u", "s", h)

    reloaded = SessionStore(db).get_or_create("u", "s")
    assert [m.origin for m in reloaded] == [None, None, "correction"]


def test_a_database_from_before_the_origin_column_is_migrated_on_open(db: Path) -> None:
    """Existing installs carry a `messages` table without `origin`; opening it must add
    the column silently and read the old rows as human messages."""
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE sessions (
            user_id TEXT NOT NULL, session_id TEXT NOT NULL,
            created_at REAL NOT NULL DEFAULT (julianday('now')),
            updated_at REAL NOT NULL DEFAULT (julianday('now')),
            workspace_dir TEXT, PRIMARY KEY (user_id, session_id));
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
            session_id TEXT NOT NULL, seq INTEGER NOT NULL, role TEXT NOT NULL,
            content TEXT NOT NULL DEFAULT '', tool_calls TEXT, tool_call_id TEXT);
        INSERT INTO sessions(user_id, session_id) VALUES ('u', 'old');
        INSERT INTO messages(user_id, session_id, seq, role, content)
            VALUES ('u', 'old', 0, 'user', 'legacy question');
        """
    )
    conn.commit()
    conn.close()

    store = SessionStore(db)
    history = store.get_or_create("u", "old")
    assert [(m.role, m.content, m.origin) for m in history] == [("user", "legacy question", None)]
    history.append(Message(role="user", content="note", origin="correction"))
    store.save("u", "old", history)
    assert SessionStore(db).get_or_create("u", "old")[-1].origin == "correction"


def test_strip_orphans_keeps_origin() -> None:
    h = [
        Message(
            role="assistant",
            content="text",
            tool_calls=[ToolCall(id="x", name="t", arguments={})],
            origin=None,
        ),
        Message(role="user", content="fix it", origin="correction"),
    ]
    cleaned = strip_orphan_tool_calls(h)
    assert cleaned[0].tool_calls is None and cleaned[0].content == "text"
    assert cleaned[1].origin == "correction"


def test_constructing_a_store_touches_no_disk_until_first_use(tmp_path: Path) -> None:
    """`store = SessionStore()` runs at import time; creating directories then made
    `import helioai` itself write to the filesystem."""
    db = tmp_path / "nested" / "sessions.db"
    store = SessionStore(db)
    assert not db.parent.exists()
    store.get_or_create("u", "s")
    assert db.exists()


def test_save_on_a_fresh_store_does_not_deadlock(tmp_path: Path) -> None:
    """`save` holds the store lock while it connects, and the first connection has to
    create the schema — that must not re-take the same lock."""
    store = SessionStore(tmp_path / "sessions.db")
    store.save("u", "s", [Message(role="user", content="hi")])
    assert SessionStore(tmp_path / "sessions.db").get_or_create("u", "s")[0].content == "hi"
