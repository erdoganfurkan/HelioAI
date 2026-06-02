"""Tests for the MCP server handlers (no subprocess, no network I/O)."""

from __future__ import annotations

import json


import helioai.tools.setup  # noqa: F401
from helioai import mcp_server as ms
from mcp import types


async def test_list_tools_count():
    tools = await ms._list_tools()
    assert len(tools) >= 10


async def test_list_tools_contains_core():
    tools = await ms._list_tools()
    names = {t.name for t in tools}
    assert "search_parameters" in names
    assert "run_python" in names
    assert "list_missions" in names


async def test_list_tools_schemas_are_objects():
    tools = await ms._list_tools()
    for tool in tools:
        assert isinstance(tool.inputSchema, dict)
        assert tool.inputSchema.get("type") == "object"


async def test_list_tools_returns_mcp_tool_instances():
    tools = await ms._list_tools()
    for tool in tools:
        assert isinstance(tool, types.Tool)


async def test_call_tool_unknown_returns_error():
    result = await ms._call_tool("does_not_exist", {})
    assert len(result) == 1
    assert isinstance(result[0], types.TextContent)
    body = json.loads(result[0].text)
    assert "error" in body


async def test_call_tool_list_missions_returns_json():
    result = await ms._call_tool("list_missions", {})
    assert len(result) == 1
    body = json.loads(result[0].text)
    assert "providers" in body
    assert isinstance(body["providers"], list)
    assert len(body["providers"]) > 0


async def test_call_tool_returns_text_content():
    result = await ms._call_tool("list_missions", {})
    assert all(isinstance(c, types.TextContent) for c in result)


async def test_call_tool_plasma_beta_correct_value():
    result = await ms._call_tool("plasma_beta", {"B_nT": 5.0, "n_cm3": 10.0, "T_eV": 10.0})
    body = json.loads(result[0].text)
    assert "beta" in body
    assert isinstance(body["beta"], float)
    assert body["beta"] > 0


async def test_call_tool_gyrofrequency_proton():
    result = await ms._call_tool("gyrofrequency", {"B_nT": 10.0})
    body = json.loads(result[0].text)
    assert "frequency_Hz" in body
    assert body["frequency_Hz"] > 0


async def test_call_tool_none_arguments():
    result = await ms._call_tool("list_missions", None)
    assert len(result) == 1
    body = json.loads(result[0].text)
    assert "providers" in body


def test_build_http_app_has_mcp_route():
    app = ms.build_http_app()
    from starlette.applications import Starlette

    assert isinstance(app, Starlette)
    paths = [str(r.path) for r in app.routes]
    assert any("/mcp" in p for p in paths)


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
