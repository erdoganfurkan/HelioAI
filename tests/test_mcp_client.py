"""Tests for the generic MCP client (helioai/tools/mcp_client.py)."""

from __future__ import annotations

import sys
import textwrap
from contextlib import asynccontextmanager

import mcp.types as types
import pytest
from mcp.client import Client
from mcp.server import Server

from helioai.config import settings
from helioai.tools import mcp_client
from helioai.tools.registry import registry


def _toy_server(big: bool = False) -> Server:
    async def _list(ctx, params):
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name="echo",
                    description="Echo text back",
                    input_schema={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                )
            ]
        )

    async def _call(ctx, params):
        args = params.arguments or {}
        text = "x" * 10_000 if big else f"echo:{args.get('text', '')}"
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)], is_error=False
        )

    return Server("toy", on_list_tools=_list, on_call_tool=_call)


def _fake_session(server: Server):
    @asynccontextmanager
    async def fake(spec):
        async with Client(server) as client:
            yield client

    return fake


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(mcp_client, "_discovered", False)
    saved = dict(registry._tools)
    yield
    registry._tools.clear()
    registry._tools.update(saved)


@pytest.fixture
def toy_config(monkeypatch):
    monkeypatch.setattr(settings.mcp, "servers_json", '{"toy": {"url": "http://unused"}}')


async def test_discover_registers_prefixed_tool(toy_config, monkeypatch):
    monkeypatch.setattr(mcp_client, "_session", _fake_session(_toy_server()))
    names = await mcp_client.discover_and_register()
    assert names == ["toy_echo"]
    assert "toy_echo" in registry
    out = await registry.call_tool("toy_echo", {"text": "hi"})
    assert out == "echo:hi"


async def test_result_truncated(toy_config, monkeypatch):
    monkeypatch.setattr(mcp_client, "_session", _fake_session(_toy_server(big=True)))
    await mcp_client.discover_and_register()
    out = await registry.call_tool("toy_echo", {"text": "hi"})
    assert len(out) < 10_000
    assert "truncated" in out


async def test_collision_skipped(toy_config, monkeypatch):
    async def dummy() -> str:
        return "original"

    registry.register(name="toy_echo", description="pre-existing", parameters={})(dummy)
    monkeypatch.setattr(mcp_client, "_session", _fake_session(_toy_server()))
    names = await mcp_client.discover_and_register()
    assert names == []
    assert await registry.call_tool("toy_echo", {}) == "original"


async def test_invalid_json_noop(monkeypatch):
    monkeypatch.setattr(settings.mcp, "servers_json", "not-json{")
    assert await mcp_client.discover_and_register() == []


async def test_empty_config_noop(monkeypatch):
    monkeypatch.setattr(settings.mcp, "servers_json", "")
    assert await mcp_client.discover_and_register() == []


async def test_dead_server_warns_not_raises(monkeypatch):
    monkeypatch.setattr(settings.mcp, "servers_json", '{"dead": {"command": "/bin/false"}}')
    assert await mcp_client.discover_and_register() == []


async def test_idempotent(toy_config, monkeypatch):
    monkeypatch.setattr(mcp_client, "_session", _fake_session(_toy_server()))
    first = await mcp_client.discover_and_register()
    second = await mcp_client.discover_and_register()
    assert first == ["toy_echo"]
    assert second == []


def test_safe_name():
    assert mcp_client._safe_name("a b", "c.d/e") == "a_b_c_d_e"
    assert len(mcp_client._safe_name("x" * 80, "y")) == 64


async def test_real_stdio_roundtrip(monkeypatch, tmp_path):
    server_script = tmp_path / "echo_server.py"
    server_script.write_text(
        textwrap.dedent(
            """
            from mcp.server.mcpserver import MCPServer

            mcp = MCPServer("stdio-toy")

            @mcp.tool()
            def shout(text: str) -> str:
                return text.upper()

            mcp.run()
            """
        ),
        encoding="utf-8",
    )
    spec = {"stdio": {"command": sys.executable, "args": [str(server_script)]}}
    monkeypatch.setattr(settings.mcp, "servers_json", __import__("json").dumps(spec))
    names = await mcp_client.discover_and_register()
    assert names == ["stdio_shout"]
    out = await registry.call_tool("stdio_shout", {"text": "hello"})
    assert out == "HELLO"
