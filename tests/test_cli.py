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


# ── migrate-storage: the split data directory ─────────────────────────────────


def _split_layout(tmp_path, monkeypatch):
    """An older install with HELIOAI_DATA_DIR set: sessions under `configured/`,
    index / catalogues / profile under the default tree the variable did not reach."""
    import helioai.config as cfg
    from helioai.config import settings

    legacy = tmp_path / "default"
    configured = tmp_path / "configured"
    (legacy / "chroma").mkdir(parents=True)
    (legacy / "chroma" / "chroma.sqlite3").write_text("index")
    (legacy / "catalogs").mkdir()
    (legacy / "catalogs" / "shocks.json").write_text("[]")
    (legacy / "profile.md").write_text("I study shocks")
    configured.mkdir()

    monkeypatch.setattr(cfg, "_default_data_dir", lambda: legacy)
    monkeypatch.setattr(settings, "data_dir", configured)
    monkeypatch.setattr(settings.rag, "chroma_dir", configured / "chroma")
    monkeypatch.setattr(settings.catalogs, "catalogs_dir", configured / "catalogs")
    monkeypatch.setattr(settings.profile, "profile_path", configured / "profile.md")
    return legacy, configured


def test_migrate_storage_moves_the_split_tree_under_the_configured_data_dir(tmp_path, monkeypatch):
    from helioai.interfaces import cli

    legacy, configured = _split_layout(tmp_path, monkeypatch)
    moved = cli._migrate_split_data_dir()

    assert moved == 3
    assert (configured / "chroma" / "chroma.sqlite3").read_text() == "index"
    assert (configured / "catalogs" / "shocks.json").exists()
    assert not (legacy / "chroma").exists()
    assert not (legacy / "profile.md").exists()


def test_migrate_storage_never_overwrites_and_is_idempotent(tmp_path, monkeypatch):
    from helioai.interfaces import cli

    legacy, configured = _split_layout(tmp_path, monkeypatch)
    (configured / "chroma").mkdir()
    (configured / "chroma" / "chroma.sqlite3").write_text("rebuilt")

    assert cli._migrate_split_data_dir() == 2
    assert (configured / "chroma" / "chroma.sqlite3").read_text() == "rebuilt"
    assert (legacy / "chroma" / "chroma.sqlite3").read_text() == "index"
    assert cli._migrate_split_data_dir() == 0


def test_migrate_storage_is_a_no_op_when_the_data_dir_is_the_default(tmp_path, monkeypatch):
    import helioai.config as cfg
    from helioai.interfaces import cli

    (tmp_path / "chroma").mkdir()
    monkeypatch.setattr(cfg, "_default_data_dir", lambda: tmp_path)
    assert cli._migrate_split_data_dir() == 0
    assert (tmp_path / "chroma").exists()


# ── argparse router: flags with missing values fail cleanly, subcommand flags parse ──


def test_a_flag_without_its_value_is_an_argument_error_not_a_crash(monkeypatch, tripwires):
    import helioai.interfaces.cli as cli

    monkeypatch.setattr(sys, "argv", ["helioai", "--session"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_serve_web_parses_host_and_port_wherever_they_come(monkeypatch):
    import helioai.interfaces.cli as cli
    import helioai.interfaces.web.app as web_app
    import helioai.workspace as ws

    seen = {}
    monkeypatch.setattr(ws, "set_user", lambda u: None)
    monkeypatch.setattr(web_app, "serve_web", lambda host, port: seen.update(host=host, port=port))
    monkeypatch.setattr(
        sys, "argv", ["helioai", "serve", "--port", "9000", "--web", "--host", "0.0.0.0"]
    )
    cli.main()
    assert seen == {"host": "0.0.0.0", "port": 9000}


def test_serve_web_without_a_port_value_is_an_argument_error(monkeypatch):
    import helioai.interfaces.cli as cli
    import helioai.workspace as ws

    monkeypatch.setattr(ws, "set_user", lambda u: None)
    monkeypatch.setattr(sys, "argv", ["helioai", "serve", "--web", "--port"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_history_delete_needs_an_id(monkeypatch, capsys):
    import helioai.interfaces.cli as cli
    import helioai.workspace as ws

    monkeypatch.setattr(ws, "set_user", lambda u: None)
    monkeypatch.setattr(sys, "argv", ["helioai", "history", "delete"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "session id" in capsys.readouterr().err


def test_index_rebuild_flag(monkeypatch):
    import helioai.interfaces.cli as cli
    import helioai.workspace as ws

    seen = {}
    monkeypatch.setattr(ws, "set_user", lambda u: None)
    monkeypatch.setattr(cli, "_run_index", lambda rebuild: seen.update(rebuild=rebuild))
    monkeypatch.setattr(sys, "argv", ["helioai", "index", "--rebuild"])
    cli.main()
    assert seen == {"rebuild": True}


def test_global_options_may_follow_the_question(monkeypatch):
    import helioai.interfaces.cli as cli
    import helioai.workspace as ws

    seen = {}
    monkeypatch.setattr(ws, "set_user", lambda u: None)
    monkeypatch.setattr(ws, "cleanup_old_runs", lambda: None)
    monkeypatch.setattr(cli, "_run_query", lambda q, **kw: seen.setdefault("q", q))
    monkeypatch.setattr(cli.asyncio, "run", lambda coro: coro)
    monkeypatch.setattr(sys, "argv", ["helioai", "plot", "IMF", "Bz", "--session", "abc"])
    cli.main()
    assert seen["q"] == "plot IMF Bz" and cli._SESSION_ID == "abc"
