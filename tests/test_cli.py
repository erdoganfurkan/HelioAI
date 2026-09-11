"""Tests for CLI history commands and routing."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def reset_session_id():
    import helioai.interfaces.cli as cli

    original = cli._SESSION_ID
    yield
    cli._SESSION_ID = original


def _make_store(summaries=None, all_ids=None):
    store = MagicMock()
    store.list_summaries.return_value = summaries or []
    store.all_sessions.return_value = all_ids or []
    return store


# --- _show_history ---


def test_show_history_empty(capsys, monkeypatch):
    import helioai.core.session as s

    monkeypatch.setattr(s, "store", _make_store())
    from helioai.interfaces.cli import _show_history

    _show_history()
    assert "No sessions found" in capsys.readouterr().out


def test_show_history_with_sessions(capsys, monkeypatch):
    import helioai.core.session as s

    monkeypatch.setattr(
        s,
        "store",
        _make_store(
            [
                {
                    "session_id": "abc12345-rest",
                    "updated_at": "2026-05-31T10:00:00Z",
                    "n_messages": 4,
                    "first_message": "solar wind density",
                },
            ]
        ),
    )
    from helioai.interfaces.cli import _show_history

    _show_history()
    out = capsys.readouterr().out
    assert "abc12345" in out
    assert "solar wind density" in out
    assert "4" in out


# --- _pick_session ---


def test_pick_session_empty_store(capsys, monkeypatch):
    import helioai.core.session as s

    monkeypatch.setattr(s, "store", _make_store())
    from helioai.interfaces.cli import _pick_session

    result = _pick_session()
    assert result is None
    assert "No previous sessions" in capsys.readouterr().out


def test_pick_session_by_number(monkeypatch):
    summaries = [
        {
            "session_id": "session-aaa",
            "updated_at": "2026-05-31T10:00:00Z",
            "n_messages": 3,
            "first_message": "first query",
        },
        {
            "session_id": "session-bbb",
            "updated_at": "2026-05-30T09:00:00Z",
            "n_messages": 2,
            "first_message": "second query",
        },
    ]
    import helioai.core.session as s

    monkeypatch.setattr(s, "store", _make_store(summaries, ["session-aaa", "session-bbb"]))
    monkeypatch.setattr("builtins.input", lambda _: "2")
    from helioai.interfaces.cli import _pick_session

    assert _pick_session() == "session-bbb"


def test_pick_session_by_prefix(monkeypatch):
    summaries = [
        {
            "session_id": "abc12345-0000",
            "updated_at": "2026-05-31T10:00:00Z",
            "n_messages": 1,
            "first_message": "query",
        }
    ]
    import helioai.core.session as s

    monkeypatch.setattr(s, "store", _make_store(summaries, ["abc12345-0000"]))
    monkeypatch.setattr("builtins.input", lambda _: "abc12345")
    from helioai.interfaces.cli import _pick_session

    assert _pick_session() == "abc12345-0000"


def test_pick_session_skip_with_enter(monkeypatch):
    summaries = [
        {
            "session_id": "x",
            "updated_at": "2026-05-31T10:00:00Z",
            "n_messages": 1,
            "first_message": "q",
        }
    ]
    import helioai.core.session as s

    monkeypatch.setattr(s, "store", _make_store(summaries, ["x"]))
    monkeypatch.setattr("builtins.input", lambda _: "")
    from helioai.interfaces.cli import _pick_session

    assert _pick_session() is None


def test_pick_session_no_match(monkeypatch):
    summaries = [
        {
            "session_id": "abc",
            "updated_at": "2026-05-31T10:00:00Z",
            "n_messages": 1,
            "first_message": "q",
        }
    ]
    import helioai.core.session as s

    monkeypatch.setattr(s, "store", _make_store(summaries, ["abc"]))
    monkeypatch.setattr("builtins.input", lambda _: "zzz")
    from helioai.interfaces.cli import _pick_session

    assert _pick_session() is None


# --- main() dispatch ---


def test_main_history_command(monkeypatch):
    called = []
    monkeypatch.setattr("helioai.interfaces.cli._show_history", lambda: called.append(True))
    monkeypatch.setattr(sys, "argv", ["helioai", "history"])
    from helioai.interfaces.cli import main

    main()
    assert called == [True]


def test_main_session_flag_sets_id(monkeypatch):
    import helioai.interfaces.cli as cli

    monkeypatch.setattr("helioai.interfaces.cli._interactive", lambda **kw: None)
    monkeypatch.setattr(sys, "argv", ["helioai", "--session", "my-session-id"])
    cli.main()
    assert cli._SESSION_ID == "my-session-id"


def test_main_resume_calls_interactive(monkeypatch):
    interactive_calls = []
    monkeypatch.setattr("helioai.interfaces.cli._pick_session", lambda: None)
    monkeypatch.setattr(
        "helioai.interfaces.cli._interactive", lambda **kw: interactive_calls.append(True)
    )
    monkeypatch.setattr(sys, "argv", ["helioai", "--resume"])
    from helioai.interfaces.cli import main

    main()
    assert interactive_calls == [True]


def test_main_resume_sets_session_when_picked(monkeypatch):
    import helioai.interfaces.cli as cli

    monkeypatch.setattr("helioai.interfaces.cli._pick_session", lambda: "picked-session-id")
    monkeypatch.setattr("helioai.interfaces.cli._interactive", lambda **kw: None)
    monkeypatch.setattr(sys, "argv", ["helioai", "--resume"])
    cli.main()
    assert cli._SESSION_ID == "picked-session-id"


# --- --help ---


@pytest.fixture
def tripwires(monkeypatch):
    """Fail loudly if `--help` does any of the work a real invocation does."""
    import asyncio

    import helioai.interfaces.cli as cli
    import helioai.workspace as ws

    def boom(name):
        def _fail(*a, **kw):
            raise AssertionError(f"--help must not call {name}")

        return _fail

    monkeypatch.setattr(ws, "set_user", boom("workspace.set_user"))
    monkeypatch.setattr(ws, "cleanup_old_runs", boom("workspace.cleanup_old_runs"))
    monkeypatch.setattr(cli, "_run_query", boom("_run_query"))
    monkeypatch.setattr(cli, "_interactive", boom("_interactive"))
    monkeypatch.setattr(asyncio, "run", boom("asyncio.run"))


@pytest.mark.parametrize("argv", [["--help"], ["-h"], ["serve", "--help"], ["index", "--help"]])
def test_help_prints_usage_without_running_anything(argv, tripwires, capsys, monkeypatch):
    """`--help` is a command, not a question.

    Measured on 2026-09-02: it fell through to the default branch, which treats
    an unrecognised argument as a query — one LLM iteration and a workspace, to
    be told what `--help` means. The absence of side effects is the real subject
    here, not the wording of the text.
    """
    from helioai.interfaces.cli import main

    monkeypatch.setattr(sys, "argv", ["helioai", *argv])
    main()

    out = capsys.readouterr().out
    assert "Usage:" in out
    assert "helioai index" in out


def test_help_text_is_the_module_documentation(capsys, monkeypatch, tripwires):
    """One string, so the help cannot drift from what the docs render."""
    import helioai.interfaces.cli as cli

    monkeypatch.setattr(sys, "argv", ["helioai", "--help"])
    cli.main()

    assert capsys.readouterr().out.strip() == (cli.__doc__ or "").strip()


def test_a_query_mentioning_help_is_still_a_query(monkeypatch):
    """`--help` inside a quoted question is one argument, not the flag."""
    import helioai.interfaces.cli as cli

    seen = {}
    monkeypatch.setattr(cli, "_run_query", lambda q, **kw: seen.setdefault("q", q))
    monkeypatch.setattr(cli.asyncio, "run", lambda coro: coro)
    monkeypatch.setattr(sys, "argv", ["helioai", "what does the --help flag of speasy do"])
    cli.main()

    assert seen["q"] == "what does the --help flag of speasy do"
