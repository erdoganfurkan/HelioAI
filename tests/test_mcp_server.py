"""Tests for the MCP server handlers (no subprocess, no network I/O)."""

from __future__ import annotations

import json

import pytest
from mcp import MCPError, types
from mcp.types import CallToolRequestParams, ReadResourceRequestParams
from starlette.testclient import TestClient

import helioai.tools.setup  # noqa: F401
from helioai import mcp_server as ms


async def test_list_tools_count():
    result = await ms._list_tools(None, None)
    assert len(result.tools) >= 10


async def test_list_tools_contains_core():
    result = await ms._list_tools(None, None)
    names = {t.name for t in result.tools}
    assert "search_parameters" in names
    assert "run_python" in names
    assert "list_missions" in names


async def test_list_tools_schemas_are_objects():
    result = await ms._list_tools(None, None)
    for tool in result.tools:
        assert isinstance(tool.input_schema, dict)
        assert tool.input_schema.get("type") == "object"


async def test_list_tools_returns_mcp_tool_instances():
    result = await ms._list_tools(None, None)
    for tool in result.tools:
        assert isinstance(tool, types.Tool)


async def test_call_tool_unknown_returns_error():
    result = await ms._call_tool(None, CallToolRequestParams(name="does_not_exist", arguments={}))
    assert len(result.content) == 1
    assert isinstance(result.content[0], types.TextContent)
    assert result.is_error is False
    body = json.loads(result.content[0].text)
    assert "error" in body


async def test_call_tool_list_missions_returns_json():
    result = await ms._call_tool(None, CallToolRequestParams(name="list_missions", arguments={}))
    body = json.loads(result.content[0].text)
    assert "providers" in body
    assert isinstance(body["providers"], list)
    assert len(body["providers"]) > 0


async def test_call_tool_returns_text_content():
    result = await ms._call_tool(None, CallToolRequestParams(name="list_missions", arguments={}))
    assert all(isinstance(c, types.TextContent) for c in result.content)


async def test_call_tool_plasma_beta_correct_value():
    params = CallToolRequestParams(
        name="plasma_beta", arguments={"B_nT": 5.0, "n_cm3": 10.0, "T_eV": 10.0}
    )
    result = await ms._call_tool(None, params)
    body = json.loads(result.content[0].text)
    assert "beta" in body
    assert isinstance(body["beta"], float)
    assert body["beta"] > 0


async def test_call_tool_gyrofrequency_proton():
    result = await ms._call_tool(
        None, CallToolRequestParams(name="gyrofrequency", arguments={"B_nT": 10.0})
    )
    body = json.loads(result.content[0].text)
    assert "frequency_Hz" in body
    assert body["frequency_Hz"] > 0


async def test_call_tool_none_arguments():
    result = await ms._call_tool(None, CallToolRequestParams(name="list_missions", arguments=None))
    body = json.loads(result.content[0].text)
    assert "providers" in body


async def test_list_resources_includes_recipes_and_skills():
    result = await ms._list_resources(None, None)
    uris = [r.uri for r in result.resources]
    assert any(u.startswith("recipe://") for u in uris)
    assert any(u.startswith("skill://") for u in uris)


async def test_read_resource_recipe_returns_python_source():
    listing = await ms._list_resources(None, None)
    uri = next(r.uri for r in listing.resources if r.uri.startswith("recipe://"))
    result = await ms._read_resource(None, ReadResourceRequestParams(uri=uri))
    assert result.contents[0].mime_type == "text/x-python"
    assert result.contents[0].uri == uri


async def test_read_resource_skill_returns_markdown():
    listing = await ms._list_resources(None, None)
    uri = next(r.uri for r in listing.resources if r.uri.startswith("skill://"))
    result = await ms._read_resource(None, ReadResourceRequestParams(uri=uri))
    assert result.contents[0].mime_type == "text/markdown"


async def test_read_resource_unknown_recipe_raises():
    with pytest.raises(MCPError):
        await ms._read_resource(None, ReadResourceRequestParams(uri="recipe://does-not-exist"))


async def test_read_resource_unknown_scheme_raises():
    with pytest.raises(MCPError):
        await ms._read_resource(None, ReadResourceRequestParams(uri="bogus://x"))


def test_build_http_app_has_mcp_route():
    from starlette.applications import Starlette

    app = ms.build_http_app()
    assert isinstance(app, Starlette)
    paths = [str(r.path) for r in app.routes]
    assert any("/mcp" in p for p in paths)


def test_http_app_open_when_token_unset(monkeypatch):
    monkeypatch.setattr(ms.settings.mcp, "token", "")
    with TestClient(ms.build_http_app()) as client:
        assert client.post("/mcp", json={}).status_code != 401


def test_http_app_requires_token_when_configured(monkeypatch):
    monkeypatch.setattr(ms.settings.mcp, "token", "s3cret")
    with TestClient(ms.build_http_app()) as client:
        assert client.post("/mcp", json={}).status_code == 401
        assert (
            client.post("/mcp", json={}, headers={"Authorization": "Bearer wrong"}).status_code
            == 401
        )
        assert (
            client.post("/mcp", json={}, headers={"Authorization": "Bearer s3cret"}).status_code
            != 401
        )


def test_init_options_returns_initialization_options():
    from mcp.server.models import InitializationOptions

    opts = ms._init_options()
    assert isinstance(opts, InitializationOptions)


def test_arg_helper_found():
    assert ms._arg(["--host", "0.0.0.0", "--port", "9000"], "--host", "127.0.0.1") == "0.0.0.0"


def test_arg_helper_default():
    assert ms._arg(["--http"], "--port", "8765") == "8765"


def test_arg_helper_flag_at_end():
    assert ms._arg(["--host"], "--host", "127.0.0.1") == "127.0.0.1"


class _RecordingLogger:
    def __init__(self):
        self.events = []

    def warning(self, event, **kw):
        self.events.append(("warning", event))

    def error(self, event, **kw):
        self.events.append(("error", event))


def test_http_refused_without_token_on_public_interface(monkeypatch):
    """--http exposes run_python; on a public interface with no token it must refuse to start.

    Over stdio that is the MCP contract — the client owns the process. Over HTTP on a
    public interface with no auth it is remote code execution, so this is now a hard
    startup failure rather than a warning someone might miss.
    """
    rec = _RecordingLogger()
    monkeypatch.setattr(ms, "get_logger", lambda _name: rec)
    monkeypatch.setattr(ms, "serve_http", lambda host, port: pytest.fail("must not start"))
    monkeypatch.setattr(ms.settings.mcp, "token", "")
    monkeypatch.setattr(ms.sys, "argv", ["helioai-mcp", "--http", "--host", "0.0.0.0"])

    with pytest.raises(SystemExit):
        ms.main()

    assert ("error", "mcp_http_refused_without_auth") in rec.events


def test_http_on_loopback_stays_quiet(monkeypatch):
    """The default bind is the supported one — logging on it trains people to ignore logs."""
    rec = _RecordingLogger()
    monkeypatch.setattr(ms, "get_logger", lambda _name: rec)
    monkeypatch.setattr(ms, "serve_http", lambda host, port: None)
    monkeypatch.setattr(ms.sys, "argv", ["helioai-mcp", "--http"])

    ms.main()

    assert rec.events == []


def test_http_on_public_interface_with_token_starts(monkeypatch):
    rec = _RecordingLogger()
    monkeypatch.setattr(ms, "get_logger", lambda _name: rec)
    started = []
    monkeypatch.setattr(ms, "serve_http", lambda host, port: started.append((host, port)))
    monkeypatch.setattr(ms.settings.mcp, "token", "secret")
    monkeypatch.setattr(ms.sys, "argv", ["helioai-mcp", "--http", "--host", "0.0.0.0"])

    ms.main()

    assert started == [("0.0.0.0", 8765)]
