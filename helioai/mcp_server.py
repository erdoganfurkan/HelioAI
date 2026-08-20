"""MCP server for HelioAI — exposes registered tools and read-only resources (recipes,
skills) via stdio or HTTP streamable transport.

Usage:
    helioai serve              # stdio (Claude Desktop / claude CLI)
    helioai serve --http       # HTTP streamable on 127.0.0.1:8765
    helioai serve --http --host 0.0.0.0 --port 9000   # requires HELIOAI_MCP_TOKEN
    helioai-mcp                # direct entry point (stdio only)

Skills are listed from a process-lifetime-cached index (skills_loader._discover is
lru_cache'd): a skill added or edited after this process started is invisible until
restart. Recipes re-glob the filesystem on every call and need no restart.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import sys
from urllib.parse import urlparse

from mcp import MCPError
from mcp.server import NotificationOptions, Server, ServerRequestContext
from mcp.server.models import InitializationOptions
from mcp.types import (
    INVALID_PARAMS,
    CallToolRequestParams,
    CallToolResult,
    ListResourcesResult,
    ListToolsResult,
    PaginatedRequestParams,
    ReadResourceRequestParams,
    ReadResourceResult,
    Resource,
    TextContent,
    TextResourceContents,
    Tool,
)

import helioai.tools.setup  # noqa: F401 — registers all tools at import time
from helioai.config import settings
from helioai.core.skills_loader import SkillError, list_skills, load_skill
from helioai.logging_config import get_logger, setup_logging
from helioai.tools.recipes import list_recipes, load_recipe
from helioai.tools.registry import registry

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


async def _list_tools(
    ctx: ServerRequestContext, params: PaginatedRequestParams | None
) -> ListToolsResult:
    return ListToolsResult(
        tools=[
            Tool(name=t.name, description=t.description, input_schema=t.parameters)
            for t in registry.list_tool_defs()
        ]
    )


async def _call_tool(ctx: ServerRequestContext, params: CallToolRequestParams) -> CallToolResult:
    # registry.call_tool() never raises — it catches everything into a JSON
    # {"error": ...} string — so is_error stays False unconditionally, same as v1.
    result = await registry.call_tool(params.name, params.arguments or {})
    return CallToolResult(content=[TextContent(type="text", text=result)], is_error=False)


async def _list_resources(
    ctx: ServerRequestContext, params: PaginatedRequestParams | None
) -> ListResourcesResult:
    resources = [
        Resource(
            uri=f"recipe://{r['name']}",
            name=r["name"],
            description=r.get("description", ""),
            mime_type="text/x-python",
        )
        for r in (await list_recipes()).get("recipes", [])
    ]
    resources += [
        Resource(
            uri=f"skill://{m.name}",
            name=m.name,
            description=m.description,
            mime_type="text/markdown",
        )
        for m in list_skills()
    ]
    return ListResourcesResult(resources=resources)


async def _read_resource(
    ctx: ServerRequestContext, params: ReadResourceRequestParams
) -> ReadResourceResult:
    parsed = urlparse(params.uri)
    name = parsed.netloc or parsed.path.lstrip("/")
    if parsed.scheme == "recipe":
        data = await load_recipe(name)
        if "error" in data:
            raise MCPError(INVALID_PARAMS, data["error"])
        return ReadResourceResult(
            contents=[
                TextResourceContents(uri=params.uri, text=data["code"], mime_type="text/x-python")
            ]
        )
    if parsed.scheme == "skill":
        try:
            body = load_skill(name)
        except SkillError as e:
            raise MCPError(INVALID_PARAMS, str(e)) from e
        return ReadResourceResult(
            contents=[TextResourceContents(uri=params.uri, text=body, mime_type="text/markdown")]
        )
    raise MCPError(INVALID_PARAMS, f"unsupported resource URI scheme: {parsed.scheme!r}")


server = Server(
    "helioai",
    on_list_tools=_list_tools,
    on_call_tool=_call_tool,
    on_list_resources=_list_resources,
    on_read_resource=_read_resource,
)


def _init_options() -> InitializationOptions:
    return server.create_initialization_options(notification_options=NotificationOptions())


async def serve_stdio() -> None:
    """Run the MCP server over stdio, for clients like Claude Desktop.

    Blocks until the client closes the pipe. All registry tools and the recipe/skill
    resources are exposed — over stdio the client owns the process, so no auth applies.
    """
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read, write):
        await server.run(read, write, _init_options())


def _require_bearer_token(app, token: str):
    """Wrap an ASGI app with a constant-time Bearer-token check.

    No-op when `token` is empty — same semantics as `config.dev_unlock`: an
    unconfigured instance requires no token. A raw ASGI wrapper (not
    starlette.middleware.base.BaseHTTPMiddleware) so it never buffers the
    streamable-HTTP SSE body.
    """
    if not token:
        return app
    expected = f"Bearer {token}".encode()

    async def _checked(scope, receive, send):
        if scope["type"] == "http":
            from starlette.responses import PlainTextResponse

            supplied = dict(scope["headers"]).get(b"authorization", b"")
            if not hmac.compare_digest(supplied, expected):
                response = PlainTextResponse(
                    "Unauthorized", status_code=401, headers={"WWW-Authenticate": "Bearer"}
                )
                await response(scope, receive, send)
                return
        await app(scope, receive, send)

    return _checked


def build_http_app():
    """Build the streamable-HTTP ASGI app exposing the MCP server.

    Returns a Starlette app mounting the MCP session manager at `/mcp`, wrapped in a
    Bearer-token check when `HELIOAI_MCP_TOKEN` is set, suitable for any ASGI server
    (`serve_http` wraps it in uvicorn).
    """
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.routing import Mount

    manager = StreamableHTTPSessionManager(app=server, json_response=False, stateless=False)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with manager.run():
            yield

    app = Starlette(routes=[Mount("/mcp", app=manager.handle_request)], lifespan=lifespan)
    return _require_bearer_token(app, settings.mcp.token)


def serve_http(host: str, port: int) -> None:
    """Run the MCP server over streamable HTTP.

    Args:
        host: Bind address.
        port: TCP port.
    """
    import uvicorn

    uvicorn.run(build_http_app(), host=host, port=port)


def _arg(args: list[str], flag: str, default: str) -> str:
    try:
        return args[args.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def main() -> None:
    """Entry point for the `helioai-mcp` command.

    Example:
        helioai-mcp                        # stdio (Claude Desktop, claude CLI)
        helioai-mcp --http --port 8765     # streamable HTTP on 127.0.0.1:8765
    """
    setup_logging("WARNING")
    args = sys.argv[1:]
    if "--http" in args:
        host = _arg(args, "--host", "127.0.0.1")
        port = int(_arg(args, "--port", "8765"))
        if host not in _LOOPBACK_HOSTS and not settings.mcp.token:
            # run_python is arbitrary code execution. Binding off loopback with no
            # token is a deployment error, not a warning someone might read after
            # the fact — refuse to start instead of trusting that.
            get_logger(__name__).error(
                "mcp_http_refused_without_auth",
                host=host,
                port=port,
                detail="set HELIOAI_MCP_TOKEN or bind to loopback",
            )
            raise SystemExit(1)
        serve_http(host, port)
    else:
        asyncio.run(serve_stdio())


if __name__ == "__main__":
    main()
