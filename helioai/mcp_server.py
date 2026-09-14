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
import base64
import contextlib
import hmac
import io
import json
import sys
from urllib.parse import urlparse
from uuid import uuid4
from weakref import WeakKeyDictionary

from mcp import MCPError
from mcp.server import NotificationOptions, Server, ServerRequestContext
from mcp.server.models import InitializationOptions
from mcp.types import (
    INVALID_PARAMS,
    CallToolRequestParams,
    CallToolResult,
    GetPromptRequestParams,
    GetPromptResult,
    ImageContent,
    ListPromptsResult,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ListToolsResult,
    PaginatedRequestParams,
    Prompt,
    PromptArgument,
    PromptMessage,
    ReadResourceRequestParams,
    ReadResourceResult,
    Resource,
    ResourceTemplate,
    TextContent,
    TextResourceContents,
    Tool,
    ToolAnnotations,
)

import helioai
import helioai.tools.setup  # noqa: F401 — registers all tools at import time
from helioai import workspace
from helioai.config import settings
from helioai.core.skills_loader import SkillError, list_skills, load_skill
from helioai.core.tool_exec import inject_run_python_args
from helioai.logging_config import get_logger, setup_logging
from helioai.tools.recipes import list_recipes, load_recipe
from helioai.tools.registry import registry

log = get_logger(__name__)

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


async def _list_tools(
    ctx: ServerRequestContext, params: PaginatedRequestParams | None
) -> ListToolsResult:
    return ListToolsResult(
        tools=[
            Tool(
                name=t.name,
                description=t.description,
                input_schema=t.parameters,
                annotations=ToolAnnotations(read_only_hint=registry.is_read_only(t.name)),
            )
            for t in registry.list_tool_defs()
        ]
    )


MCP_USER = "mcp"

_sessions: WeakKeyDictionary = WeakKeyDictionary()
_PROCESS_SESSION = f"mcp_{uuid4().hex[:12]}"


def _session_id(ctx: ServerRequestContext | None) -> str:
    """Stable workspace id for the connection this request arrived on.

    Keyed on the connection, **not** on `ctx.session`: `runner._make_context` builds a
    fresh ServerSession for every inbound message, whatever the docstring on
    ServerRequestContext says about it being connection-scoped. Keying on the session
    gave every single call its own directory, so `get_timeseries` wrote its npz in one
    and the next `run_python` looked for the manifest in another — `load_data` raised
    "no dataset manifest found" on the call straight after a successful download. Only
    a live client showed it; the tests were passing two different fake sessions and
    calling that two connections.

    Held weakly so a closed connection stops pinning its id. When the connection cannot
    be reached — the attribute is private and may be renamed — the whole process shares
    one id, which is exactly right for stdio, where the client owns the process.
    """
    connection = getattr(getattr(ctx, "session", None), "_connection", None)
    if connection is None:
        return _PROCESS_SESSION
    try:
        return _sessions.setdefault(connection, f"mcp_{uuid4().hex[:12]}")
    except TypeError:
        return _PROCESS_SESSION


def _is_error(result: str) -> bool:
    """Whether a registry result is a failure.

    `registry.call_tool` never raises: an unknown tool, a rejected private argument and
    any exception all come back as a JSON object carrying `error`, and several tools
    report their own failures the same way. Reading that back is the only signal there
    is, and clients drive retries and their own error display off `isError` — returning
    False unconditionally reported every one of those as a success.
    """
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and "error" in payload


FIGURE_MAX_PX = 768


def _figure_content(result: str) -> list[ImageContent]:
    """Attach the figures a tool produced, downscaled, to its result.

    The sandbox returns `figure_paths` — paths on the server's filesystem. That is
    useless to any client that is not the same machine, and over HTTP it is useless to
    all of them, which left the plots of a plotting agent stranded on the server. The
    paths stay in the text body for a local client that wants full resolution.

    Downscaled to FIGURE_MAX_PX because these ride inline in every response: a
    publication-sized figure is several megabytes of base64 to say what a legible
    thumbnail says. A figure that cannot be read is skipped rather than failing the
    call — it already succeeded.
    """
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []

    content: list[ImageContent] = []
    for path in payload.get("figure_paths") or []:
        try:
            from PIL import Image

            with Image.open(path) as img:
                img.thumbnail((FIGURE_MAX_PX, FIGURE_MAX_PX))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
        except Exception:
            log.warning("mcp_figure_unreadable", path=str(path))
            continue
        content.append(
            ImageContent(
                type="image",
                data=base64.b64encode(buf.getvalue()).decode(),
                mime_type="image/png",
            )
        )
    return content


async def _call_tool(ctx: ServerRequestContext, params: CallToolRequestParams) -> CallToolResult:
    user_token = workspace.set_user(MCP_USER)
    session_token = workspace.set_session(_session_id(ctx))
    try:
        result = await registry.call_tool(
            params.name,
            params.arguments or {},
            trusted=inject_run_python_args(params.name),
        )
    finally:
        workspace.reset_session(session_token)
        workspace.reset_user(user_token)
    return CallToolResult(
        content=[TextContent(type="text", text=result), *_figure_content(result)],
        is_error=_is_error(result),
    )


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


async def _list_resource_templates(
    ctx: ServerRequestContext, params: PaginatedRequestParams | None
) -> ListResourceTemplatesResult:
    """Declare the two URI shapes the read handler already accepts."""
    return ListResourceTemplatesResult(
        resource_templates=[
            ResourceTemplate(
                uri_template="recipe://{name}",
                name="recipe",
                description="Source of a derived scientific recipe, by name.",
                mime_type="text/x-python",
            ),
            ResourceTemplate(
                uri_template="skill://{name}",
                name="skill",
                description="Body of an analysis skill, by name.",
                mime_type="text/markdown",
            ),
        ]
    )


async def _list_prompts(
    ctx: ServerRequestContext, params: PaginatedRequestParams | None
) -> ListPromptsResult:
    """Offer the skills as prompts, not only as resources.

    A skill is a procedure meant to be followed, and a resource is something a model has
    to think to go and read. As prompts they become slash commands in the client, which
    is what these six were written to be — the resource listing stays as well so nothing
    that already reads `skill://` breaks.
    """
    return ListPromptsResult(
        prompts=[
            Prompt(
                name=m.name,
                description=f"{m.description} Use when: {m.when_to_use}",
                arguments=[
                    PromptArgument(
                        name="task",
                        description="What you want done — appended to the procedure.",
                        required=False,
                    )
                ],
            )
            for m in list_skills()
        ]
    )


async def _get_prompt(ctx: ServerRequestContext, params: GetPromptRequestParams) -> GetPromptResult:
    try:
        body = load_skill(params.name)
    except SkillError as e:
        raise MCPError(INVALID_PARAMS, str(e)) from e
    task = (params.arguments or {}).get("task", "")
    text = f"{body}\n\n---\n\n{task}" if task else body
    return GetPromptResult(
        messages=[PromptMessage(role="user", content=TextContent(type="text", text=text))]
    )


server = Server(
    "helioai",
    version=helioai.__version__,
    on_list_tools=_list_tools,
    on_call_tool=_call_tool,
    on_list_resources=_list_resources,
    on_read_resource=_read_resource,
    on_list_resource_templates=_list_resource_templates,
    on_list_prompts=_list_prompts,
    on_get_prompt=_get_prompt,
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
