"""Tests for `helioai mcp-install` — the config emitted for external MCP clients.

These run against real files on tmp_path rather than mocks: the whole point of the
command is that it produces something another program can actually load, and a
mocked open() proves nothing about that.
"""

from __future__ import annotations

import json

import pytest

from helioai.interfaces.cli import (
    _MCP_CLIENTS,
    _mcp_config_path,
    _mcp_server_command,
    _mcp_snippet,
    _run_mcp_install,
    _write_json_config,
)


def test_server_command_is_absolute_and_named_helioai_mcp():
    """A relative command is useless: the client launches it from its own cwd."""
    cmd = _mcp_server_command()
    assert cmd.endswith("helioai-mcp")
    assert cmd.startswith("/") or ":" in cmd


def test_every_advertised_client_renders_a_snippet():
    for client in _MCP_CLIENTS:
        text = _mcp_snippet(client)
        assert "helioai-mcp" in text
        assert text.strip()


def test_claude_code_snippet_uses_the_native_command():
    """Claude Code ships `claude mcp add`; reimplementing its config write would
    be a worse version of a tool the user already has."""
    text = _mcp_snippet("claude-code")
    assert "claude mcp add" in text
    assert "helioai" in text


def test_json_clients_emit_parseable_json():
    for client in ("claude-desktop", "claude-code-project"):
        text = _mcp_snippet(client)
        start = text.index("{")
        payload = json.loads(text[start:])
        assert payload["mcpServers"]["helioai"]["command"].endswith("helioai-mcp")


def test_codex_snippet_is_toml_and_is_never_written():
    """No TOML writer in the stdlib — emitting a snippet beats corrupting a config."""
    text = _mcp_snippet("codex")
    assert "[mcp_servers.helioai]" in text
    assert _mcp_config_path("codex") is not None


def test_unknown_client_is_refused():
    with pytest.raises(ValueError, match="unknown"):
        _mcp_snippet("emacs")


def test_write_creates_the_file_with_a_loadable_config(tmp_path):
    target = tmp_path / "nested" / "claude_desktop_config.json"
    _write_json_config(target)
    payload = json.loads(target.read_text())
    assert payload["mcpServers"]["helioai"]["command"].endswith("helioai-mcp")


def test_write_preserves_other_servers_and_unrelated_keys(tmp_path):
    """Merging, not overwriting: these files hold the user's other servers."""
    target = tmp_path / "config.json"
    target.write_text(
        json.dumps(
            {
                "mcpServers": {"other": {"command": "/usr/bin/other-mcp"}},
                "theme": "dark",
            }
        )
    )
    _write_json_config(target)
    payload = json.loads(target.read_text())
    assert payload["mcpServers"]["other"]["command"] == "/usr/bin/other-mcp"
    assert payload["theme"] == "dark"
    assert "helioai" in payload["mcpServers"]


def test_write_is_idempotent(tmp_path):
    target = tmp_path / "config.json"
    _write_json_config(target)
    first = target.read_text()
    _write_json_config(target)
    assert target.read_text() == first


def test_write_refuses_a_corrupt_config_instead_of_clobbering_it(tmp_path):
    """Overwriting a file we failed to parse would silently delete real servers."""
    target = tmp_path / "config.json"
    target.write_text("{ this is not json")
    with pytest.raises(ValueError, match="could not be parsed"):
        _write_json_config(target)
    assert target.read_text() == "{ this is not json"


def test_run_prints_all_clients_by_default(capsys):
    _run_mcp_install([])
    out = capsys.readouterr().out
    for client in _MCP_CLIENTS:
        assert client in out


def test_run_with_client_prints_only_that_one(capsys):
    _run_mcp_install(["--client", "codex"])
    out = capsys.readouterr().out
    assert "[mcp_servers.helioai]" in out
    assert "claude mcp add" not in out


def test_run_with_unknown_client_reports_the_valid_ones(capsys):
    _run_mcp_install(["--client", "emacs"])
    out = capsys.readouterr().out
    assert "emacs" in out
    assert "codex" in out


def test_run_write_refuses_codex(capsys):
    """TOML is emitted, never written — say so rather than pretend it worked."""
    _run_mcp_install(["--client", "codex", "--write"])
    out = capsys.readouterr().out
    assert "cannot" in out.lower() or "paste" in out.lower()
