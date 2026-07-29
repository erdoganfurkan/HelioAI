"""Generic MCP client — mounts remote MCP server tools into the ToolRegistry.

Configured via HELIOAI_MCP_SERVERS, a JSON object keyed by server alias:
    {"ads":      {"url": "https://.../mcp", "headers": {"Authorization": "Bearer x"}},
     "alphaxiv": {"command": "npx", "args": ["-y", "mcp-remote", "https://api.alphaxiv.org/mcp/v1"]}}

Remote tools are registered as "<alias>_<toolname>" and proxied with a fresh
connection per call: the CLI runs one asyncio.run() per query, so a persistent
session would not outlive a single request anyway.
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from helioai.config import settings
from helioai.logging_config import get_logger
from helioai.tools.registry import registry

log = get_logger(__name__)

_DISCOVER_TIMEOUT_S = 20
_CALL_TIMEOUT_S = 60
_MAX_RESULT_CHARS = 6000

_discovered = False


def _server_specs() -> dict[str, dict]:
    raw = settings.mcp.servers_json
    if not raw:
        return {}
    try:
        specs = json.loads(raw)
    except ValueError as e:
        log.warning("mcp_servers_json_invalid", error=str(e))
        return {}
    if not isinstance(specs, dict):
        log.warning("mcp_servers_json_invalid", error="expected a JSON object")
        return {}
    return specs


@asynccontextmanager
async def _session(spec: dict):
    if "command" in spec:
        params = StdioServerParameters(
            command=spec["command"], args=spec.get("args", []), env=spec.get("env")
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    elif "url" in spec:
        async with (
            streamablehttp_client(spec["url"], headers=spec.get("headers")) as (
                read,
                write,
                _,
            ),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            yield session
    else:
        raise ValueError("MCP server spec needs 'command' or 'url'")


def _tool_result_to_str(result) -> str:
    parts = [c.text for c in result.content if getattr(c, "text", None)]
    if len(parts) < len(result.content):
        parts.append("[non-text content omitted]")
    text = "\n".join(parts) or "[empty result]"
    if len(text) > _MAX_RESULT_CHARS:
        text = text[:_MAX_RESULT_CHARS] + f"... [truncated {len(text) - _MAX_RESULT_CHARS} chars]"
    return text


def _make_proxy(alias: str, spec: dict, tool_name: str):
    async def _proxy(**kwargs) -> str:
        async def _call():
            async with _session(spec) as session:
                return await session.call_tool(tool_name, kwargs)

        try:
            result = await asyncio.wait_for(_call(), timeout=_CALL_TIMEOUT_S)
        except Exception as e:
            return json.dumps({"error": f"MCP server {alias!r} call failed: {e}"})
        return _tool_result_to_str(result)

    return _proxy


def _safe_name(alias: str, tool_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", f"{alias}_{tool_name}")[:64]


async def discover_and_register() -> list[str]:
    global _discovered
    if _discovered:
        return []
    _discovered = True

    registered: list[str] = []
    for alias, spec in _server_specs().items():
        try:
            # `spec` is bound as a default: the closure would otherwise read the
            # loop variable at await time, so every server after the first would
            # be discovered against the last spec if this were ever deferred.
            async def _list(spec=spec):
                async with _session(spec) as session:
                    return await session.list_tools()

            tools = (await asyncio.wait_for(_list(), timeout=_DISCOVER_TIMEOUT_S)).tools
        except Exception as e:
            log.warning("mcp_server_unreachable", server=alias, error=str(e))
            continue

        for t in tools:
            name = _safe_name(alias, t.name)
            if name in registry:
                log.warning("mcp_tool_collision_skipped", server=alias, tool=name)
                continue
            schema = t.inputSchema or {"type": "object", "properties": {}}
            props = schema.get("properties") or {}
            if any(k.startswith("_") for k in props):
                log.warning("mcp_tool_private_args", server=alias, tool=name)
            registry.register(
                name=name,
                description=f"[{alias} MCP] {t.description or t.name}",
                parameters=schema,
            )(_make_proxy(alias, spec, t.name))
            registered.append(name)
        log.info("mcp_server_mounted", server=alias, n_tools=len(tools))
    return registered
