"""Tests for the CLI's event renderer and session commands.

`_render_event` is the entire visible output of the interactive CLI: every
streamed agent event passes through it, and a missing branch means the user
silently sees nothing at all. It was untested because interfaces/cli.py sat on
the coverage omit list.

ANSI colour codes are stripped before asserting so the tests pin what the user
reads, not how it is coloured.
"""

from __future__ import annotations

import re

import pytest

from helioai.interfaces import cli

ANSI = re.compile(r"\033\[[0-9;]*m")


def render(capsys, event: str, data: dict) -> str:
    cli._render_event({"event": event, "data": data})
    return ANSI.sub("", capsys.readouterr().out)


# ── reply / error / done ───────────────────────────────────────────────────────


def test_reply_is_printed(capsys):
    assert "theta_Bn is 47 degrees" in render(capsys, "reply", {"text": "theta_Bn is 47 degrees"})


def test_error_is_printed_with_a_marker(capsys):
    out = render(capsys, "error", {"message": "speasy timed out"})
    assert "speasy timed out" in out
    assert "✗" in out


def test_done_reports_the_iteration_count(capsys):
    assert "3" in render(capsys, "done", {"n_iterations": 3})


def test_done_without_a_count_does_not_crash(capsys):
    assert "0" in render(capsys, "done", {})


# ── tool call / result ─────────────────────────────────────────────────────────


def test_tool_call_shows_name_and_arguments(capsys):
    out = render(capsys, "tool_call", {"name": "get_timeseries", "arguments": {"id": "amda/imf"}})
    assert "get_timeseries" in out
    assert "amda/imf" in out


def test_tool_call_without_arguments(capsys):
    assert "list_missions" in render(capsys, "tool_call", {"name": "list_missions"})


def test_long_argument_values_are_truncated(capsys):
    """A whole catalog pasted into the terminal would bury the conversation."""
    out = render(capsys, "tool_call", {"name": "t", "arguments": {"blob": "x" * 500}})
    assert len(out) < 200


def test_tool_result_shows_the_summary(capsys):
    out = render(capsys, "tool_result", {"name": "get_timeseries", "summary": "1440 points"})
    assert "1440 points" in out


def test_nested_sub_agent_events_are_indented_further(capsys):
    """Sub-agent activity must be visually distinguishable from the lead agent's."""
    flat = render(capsys, "tool_call", {"name": "t", "arguments": {}})
    nested = render(capsys, "tool_call", {"name": "t", "arguments": {}, "sub_agent_ctx": "x"})
    assert len(nested) - len(nested.lstrip()) > len(flat) - len(flat.lstrip())


# ── sub-agent lifecycle ────────────────────────────────────────────────────────


def test_sub_agent_start_names_the_role(capsys):
    assert "data_analyst" in render(capsys, "sub_agent_start", {"role": "data_analyst"})


def test_sub_agent_end_success_shows_the_summary(capsys):
    out = render(capsys, "sub_agent_end", {"role": "data_analyst", "summary": "found 3 shocks"})
    assert "found 3 shocks" in out
    assert "✓" in out


def test_sub_agent_end_failure_shows_the_error(capsys):
    out = render(capsys, "sub_agent_end", {"role": "librarian", "error": "ADS token missing"})
    assert "ADS token missing" in out
    assert "✗" in out


def test_skill_loaded_names_the_skill(capsys):
    assert "plotting" in render(capsys, "skill_loaded", {"name": "plotting"})


# ── artifacts ──────────────────────────────────────────────────────────────────


def test_image_artifact_lists_figure_paths(capsys, monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(cli, "_open_file", opened.append)

    out = render(
        capsys, "artifact", {"kind": "image", "figure_paths": ["/tmp/a.png", "/tmp/b.png"]}
    )
    assert "2 figure" in out
    assert "/tmp/a.png" in out
    assert opened == ["/tmp/a.png", "/tmp/b.png"], "each figure should be opened in the viewer"


def test_image_artifact_also_prints_stdout(capsys, monkeypatch):
    monkeypatch.setattr(cli, "_open_file", lambda _: None)
    out = render(capsys, "artifact", {"kind": "image", "figure_paths": [], "stdout": "beta = 1.7"})
    assert "beta = 1.7" in out


def test_data_preview_artifact_shows_parameter_and_point_count(capsys):
    out = render(
        capsys,
        "artifact",
        {"kind": "data_preview", "param_id": "amda/imf_bz", "n_points": 1440},
    )
    assert "amda/imf_bz" in out
    assert "1440" in out


def test_data_preview_truncates_a_long_preview(capsys):
    out = render(
        capsys,
        "artifact",
        {"kind": "data_preview", "param_id": "p", "n_points": 1, "preview": "\n".join("l" * 20)},
    )
    assert out.count("\n") <= 8


def test_unknown_event_is_ignored_silently(capsys):
    """Forward compatibility: a new server-side event must not crash an old CLI."""
    assert render(capsys, "some_future_event", {"whatever": 1}) == ""


def test_unknown_artifact_kind_is_ignored(capsys):
    assert render(capsys, "artifact", {"kind": "hologram"}) == ""


# ── _open_file ─────────────────────────────────────────────────────────────────


def test_open_file_never_raises(monkeypatch):
    """Opening a viewer is best-effort; a headless box must not break the run."""

    def boom(*a, **kw):
        raise OSError("no display")

    monkeypatch.setattr("subprocess.Popen", boom)
    cli._open_file("/tmp/whatever.png")


# ── _delete_session ────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path, monkeypatch):
    from helioai.core.session import SessionStore

    s = SessionStore(tmp_path / "sessions.db")
    monkeypatch.setattr("helioai.core.session.store", s)
    return s


def test_delete_session_reports_when_nothing_matches(store, capsys):
    cli._delete_session("deadbeef")
    assert "No session matching" in capsys.readouterr().out


def test_delete_session_removes_history_and_workspace(store, tmp_path, monkeypatch, capsys):
    from helioai.core.llm.base import Message

    store.save(cli._USER_ID, "abc12345", [Message(role="user", content="hi")])
    store.set_workspace_dir(cli._USER_ID, "abc12345", "run_1")

    ws_root = tmp_path / "workspace"
    (ws_root / "run_1").mkdir(parents=True)
    (ws_root / "run_1" / "figure.png").write_bytes(b"x")
    monkeypatch.setattr("helioai.workspace._root", lambda: ws_root)

    cli._delete_session("abc123")

    assert "deleted" in capsys.readouterr().out
    assert store.get_or_create(cli._USER_ID, "abc12345") == []
    assert not (ws_root / "run_1").exists()


# ── shared display layer ───────────────────────────────────────────────────────


def test_tool_call_prefers_the_shared_display_string(capsys):
    """Emitted once for all three interfaces; the renderer must not re-derive it."""
    out = render(
        capsys,
        "tool_call",
        {"name": "run_python", "arguments": {"code": "x=1\ny=2"}, "display": "108 lines of Python"},
    )
    assert "108 lines of Python" in out
    assert "x=1" not in out


def test_tool_result_prefers_display_over_the_model_summary(capsys):
    out = render(
        capsys,
        "tool_result",
        {
            "name": "get_timeseries",
            "summary": '{"param_id": "cda/A/B", "dataset_note": "use load_data"}',
            "display": "cda/A/B · WI · 3 s · 1800 points",
        },
    )
    assert "1800 points" in out
    assert "dataset_note" not in out


def test_long_stdout_is_capped_and_says_how_much_was_hidden(capsys, monkeypatch):
    """A run that prints 40 samples must not push the answer off the screen."""
    monkeypatch.setattr(cli, "_open_file", lambda _: None)
    stdout = "\n".join(f"03:58:{i:02d}  |B| = 10.0 nT" for i in range(40))
    out = render(capsys, "artifact", {"kind": "image", "figure_paths": [], "stdout": stdout})
    assert "03:58:00" in out
    assert "03:58:39" not in out
    assert "more lines" in out
