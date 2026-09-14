"""Scripted stand-ins for the model and the tool registry, shared by every loop test.

Two copies of a scripted LLM lived in the suite — `conftest._FakeLLM` and
`test_stream_subagent.ScriptedLLM` — with different `chat` signatures, so a test written
against one could not drive the other loop, and the sub-agent's `tool_choice` was
invisible to the lead's tests. One class, the full `LLMClient.chat` signature, a journal
of every call.

`ScriptedRegistry` replaces the per-file `fake_call_tool` closures: results keyed by tool
name, or by tool name and arguments when a test needs two calls of one tool to differ,
returned as the `ToolResult` the real registry returns. Install it over the real
registry's dispatch — `monkeypatch.setattr(registry, "call_tool", scripted.call_tool)` —
so the tool definitions the model is shown stay the real ones. Nothing here reaches a
network, a model or the disk; that is the point.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from helioai.core.llm.base import LLMClient, Message, ToolCall, ToolDef
from helioai.tools.results import ToolResult


def assistant_text(content: str) -> Message:
    """An assistant turn that answers in prose."""
    return Message(role="assistant", content=content)


def assistant_calls(*names: str, arguments: dict | None = None, content: str = "") -> Message:
    """An assistant turn that calls the named tools, ids `call_0`, `call_1`, …"""
    return Message(
        role="assistant",
        content=content,
        tool_calls=[
            ToolCall(id=f"call_{i}", name=n, arguments=dict(arguments or {}))
            for i, n in enumerate(names)
        ],
    )


class ScriptedLLM(LLMClient):
    """Replays a list of assistant messages, one per `chat` call, and records each call.

    Attributes:
        calls: One dict per call — `messages`, `tools` (the tool *names* offered),
            `system_prompt`, `tool_choice` — so a test can assert what the loop sent
            without a mock's `call_args` gymnastics.
    """

    def __init__(self, responses: Iterable[Message]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system_prompt: str | None = None,
        tool_choice: str = "auto",
    ) -> Message:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": [t.name for t in tools],
                "system_prompt": system_prompt,
                "tool_choice": tool_choice,
            }
        )
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        return self._responses.pop(0)


def _args_key(arguments: dict | None) -> str:
    return json.dumps(arguments or {}, sort_keys=True, default=str)


class ScriptedRegistry:
    """A `registry.call_tool` replacement that answers from a script.

    Args:
        results: `{tool_name: result}`; a result may be a `ToolResult`, a dict or a
            JSON string, all wrapped with `ToolResult.from_raw` as the real registry
            would. A call with no scripted result gets `{"ok": true}`, so a scenario
            only scripts what it asserts on. `script(name, result, arguments=…)` pins
            a result to one set of arguments when two calls of a tool must differ.

    Attributes:
        invoked: The tool names called, in order.
        calls: `(name, arguments)` pairs, in order.
    """

    def __init__(self, results: dict[str, Any] | None = None) -> None:
        self._by_name: dict[str, Any] = dict(results or {})
        self._by_call: dict[tuple[str, str], Any] = {}
        self.invoked: list[str] = []
        self.calls: list[tuple[str, dict]] = []

    def script(self, name: str, result: Any, arguments: dict | None = None) -> None:
        """Add or replace one scripted result; with `arguments`, only for that call."""
        if arguments is None:
            self._by_name[name] = result
        else:
            self._by_call[(name, _args_key(arguments))] = result

    async def call_tool(
        self, name: str, arguments: dict | None, *, trusted: dict | None = None
    ) -> ToolResult:
        self.invoked.append(name)
        self.calls.append((name, dict(arguments or {})))
        scripted = self._by_call.get((name, _args_key(arguments)), self._by_name.get(name))
        if scripted is None:
            scripted = {"ok": True}
        if isinstance(scripted, ToolResult):
            return scripted
        return ToolResult.from_raw(name, scripted)
