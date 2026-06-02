"""MCP server for HelioAI — exposes all registered tools via stdio or HTTP streamable transport.

Usage:
    helioai serve              # stdio (Claude Desktop / claude CLI)
    helioai serve --http       # HTTP streamable on 127.0.0.1:8765
    helioai serve --http --host 0.0.0.0 --port 9000
    helioai-mcp                # direct entry point (stdio only)
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from typing import Any

import helioai.tools.setup  # noqa: F401 — registers all tools at import time
from helioai.logging_config import setup_logging
from helioai.tools.registry import registry

from mcp import types
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.lowlevel import NotificationOptions

server = Server("helioai")


@server.list_tools()
async def _list_tools() -> list[types.Tool]:
    return [
        types.Tool(name=t.name, description=t.description, inputSchema=t.parameters)
        for t in registry.list_tool_defs()
    ]


@server.call_tool()
async def _call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
    result = await registry.call_tool(name, arguments or {})
    return [types.TextContent(type="text", text=result)]


def _init_options() -> InitializationOptions:
    return server.create_initialization_options(
        notification_options=NotificationOptions()
    )


async def serve_stdio() -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read, write):
        await server.run(read, write, _init_options())


def build_http_app():
    from starlette.applications import Starlette
    from starlette.routing import Mount
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    manager = StreamableHTTPSessionManager(app=server, json_response=False, stateless=False)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with manager.run():
            yield

    return Starlette(routes=[Mount("/mcp", app=manager.handle_request)], lifespan=lifespan)


def serve_http(host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(build_http_app(), host=host, port=port)


def _arg(args: list[str], flag: str, default: str) -> str:
    try:
        return args[args.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def main() -> None:
    setup_logging("WARNING")
    args = sys.argv[1:]
    if "--http" in args:
        host = _arg(args, "--host", "127.0.0.1")
        port = int(_arg(args, "--port", "8765"))
        serve_http(host, port)
    else:
        asyncio.run(serve_stdio())


if __name__ == "__main__":
    main()
