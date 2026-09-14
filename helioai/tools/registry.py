"""Tool registry: wraps Python async functions with JSON Schema metadata.

The agent loop calls `registry.call_tool(name, args)` and never imports
tool modules directly, which keeps the dependency surface small and makes
sub-agent tool whitelisting trivial. Every call comes back as a `ToolResult`
(`tools/results.py`): the payload once, the model's text once.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from helioai.core.llm.base import ToolDef
from helioai.tools.results import ToolResult


@dataclass
class Tool:
    """A registered tool: an async function plus the schema shown to the model."""

    name: str
    description: str
    parameters: dict  # JSON Schema object
    func: Callable[..., Coroutine[Any, Any, Any]]
    read_only: bool = True


class ToolRegistry:
    """Maps tool names to async functions and their JSON Schemas.

    The agent loop dispatches through here and never imports tool modules, which
    keeps the dependency surface small and makes per-role whitelisting trivial.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self, name: str, description: str, parameters: dict, *, read_only: bool = True
    ) -> Callable:
        """Decorator that registers an async function as a tool.

        Args:
            read_only: False for a tool that changes something a user would care about
                — running arbitrary code, writing a catalogue. MCP clients gate
                auto-approval on it, so the default is the safe-to-repeat majority and
                the two exceptions say so at their registration site. Kept here rather
                than in a table beside the MCP server: the registry is the one place
                that already describes every tool.
        """

        def decorator(func: Callable) -> Callable:
            self._tools[name] = Tool(
                name=name,
                description=description,
                parameters=parameters,
                func=func,
                read_only=read_only,
            )
            return func

        return decorator

    def list_tool_defs(self, only: set[str] | None = None) -> list[ToolDef]:
        """Return tool definitions for the model.

        Args:
        only: Restrict to these names — how sub-agent whitelists are applied.
        None returns every registered tool.
        """
        tools = self._tools.values()
        if only is not None:
            tools = [t for t in tools if t.name in only]
        return [
            ToolDef(name=t.name, description=t.description, parameters=t.parameters) for t in tools
        ]

    async def call_tool(
        self, name: str, arguments: dict | None, *, trusted: dict | None = None
    ) -> ToolResult:
        """Invoke a tool and return its result, typed.

        Never raises: an unknown tool, a rejected private argument and any exception
        the tool lets out all come back as a failed `ToolResult`, whose `for_llm()` is
        the `{"error": ...}` line this method used to return as a string.

        Args:
            name: Registered tool name.
            arguments: Caller-supplied (LLM/MCP) arguments; may not carry private `_*`
                keys.
            trusted: Framework-injected arguments (the sandbox output directory) that
                bypass that guard.

        Returns:
            The tool's payload and the exact text the model will read.
        """
        if name not in self._tools:
            return ToolResult.failure(name, f"unknown tool {name!r}")
        if arguments and any(k.startswith("_") for k in arguments):
            bad = sorted(k for k in arguments if k.startswith("_"))
            return ToolResult.failure(name, f"rejected private argument(s): {bad}")
        try:
            result = await self._tools[name].func(**{**(arguments or {}), **(trusted or {})})
        except Exception as e:
            return ToolResult.failure(name, str(e) or type(e).__name__)
        return ToolResult.from_raw(name, result)

    def is_read_only(self, name: str) -> bool:
        """Whether the tool leaves the user's world unchanged. Unknown names count as not."""
        tool = self._tools.get(name)
        return bool(tool and tool.read_only)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


registry = ToolRegistry()
