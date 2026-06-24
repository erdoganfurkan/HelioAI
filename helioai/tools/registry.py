"""Tool registry: wraps Python async functions with JSON Schema metadata.

The agent loop calls `registry.call_tool(name, args)` and never imports
tool modules directly, which keeps the dependency surface small and makes
sub-agent tool whitelisting trivial.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from helioai.core.llm.base import ToolDef


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema object
    func: Callable[..., Coroutine[Any, Any, Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, parameters: dict) -> Callable:
        """Decorator that registers an async function as a tool."""

        def decorator(func: Callable) -> Callable:
            self._tools[name] = Tool(
                name=name,
                description=description,
                parameters=parameters,
                func=func,
            )
            return func

        return decorator

    def list_tool_defs(self, only: set[str] | None = None) -> list[ToolDef]:
        tools = self._tools.values()
        if only is not None:
            tools = [t for t in tools if t.name in only]
        return [
            ToolDef(name=t.name, description=t.description, parameters=t.parameters) for t in tools
        ]

    async def call_tool(
        self, name: str, arguments: dict | None, *, trusted: dict | None = None
    ) -> str:
        """Invoke a tool and return its JSON-serialized result string.

        `arguments` is caller-supplied (LLM/MCP) and may not carry private
        `_*` keys. `trusted` is framework-injected (e.g. the sandbox output
        dir) and bypasses that guard.
        """
        if name not in self._tools:
            return json.dumps({"error": f"unknown tool {name!r}"})
        if arguments and any(k.startswith("_") for k in arguments):
            bad = sorted(k for k in arguments if k.startswith("_"))
            return json.dumps({"error": f"rejected private argument(s): {bad}"})
        try:
            result = await self._tools[name].func(**{**(arguments or {}), **(trusted or {})})
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def __contains__(self, name: str) -> bool:
        return name in self._tools


registry = ToolRegistry()
