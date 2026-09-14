"""What a tool call returns, typed once at the registry boundary.

Every tool hands back a dict (or, for the remote MCP proxies and the internal tools, a
string), and the registry used to serialise it to JSON on the spot. Downstream, five
readers — the history writer, the artifact extractor, the figure review, the MCP server's
`isError`, the event display — each parsed that string back and sniffed its shape. The
`ToolResult` carries the parsed payload alongside the exact text the model receives, so
the shape is read once and the text is produced once.

`for_llm()` is the contract that lets this module exist without changing agent
behaviour: it returns, byte for byte, the string `registry.call_tool` used to return —
a string result untouched, a dict as `json.dumps(..., ensure_ascii=False, default=str)`,
a failure as `{"error": ...}`. `tests/test_tool_results.py` holds it to captured results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """The outcome of one tool call: the parsed payload and the text the model sees.

    Attributes:
        tool: Name of the tool that produced it.
        payload: What the tool returned — a dict for every shipped tool, a list or a
            scalar in principle, a string for a remote MCP proxy whose text is not
            JSON. A string that *is* JSON (the internal tools, the `task` result) is
            parsed here so readers see the object, while `for_llm()` still returns
            the original text.
        raw: The exact string a tool returned, when it returned one. `for_llm()`
            hands it back untouched: re-serialising a parsed copy could change the
            escaping of a non-ASCII character, and the model must see what it saw.
    """

    tool: str
    payload: Any
    raw: str | None = field(default=None, repr=False)

    @classmethod
    def from_raw(cls, tool: str, obj: Any) -> ToolResult:
        """Wrap whatever a tool returned.

        Args:
            tool: The tool's name.
            obj: Its return value — dict, list, string, anything JSON can carry.

        Returns:
            A result whose `payload` is the object (a JSON string is parsed) and
            whose `for_llm()` is what the registry used to return for that value.
        """
        if isinstance(obj, str):
            try:
                parsed = json.loads(obj)
            except ValueError:
                parsed = obj
            return cls(tool, parsed, obj)
        return cls(tool, obj)

    @classmethod
    def failure(cls, tool: str, message: str) -> ToolResult:
        """A result for a tool that never ran, or raised.

        Args:
            tool: The tool that was asked for.
            message: Why it failed. An empty message is replaced by the word `error`
                rather than kept: `str(TimeoutError())` is `""`, every reader tests the
                error field for truth, and a tool that never ran was once displayed as ok.

        Returns:
            A result whose `for_llm()` is the `{"error": ...}` line the registry has
            always returned for these cases, ASCII-escaped as before.
        """
        payload = {"error": message or "error"}
        return cls(tool, payload, json.dumps(payload))

    @property
    def ok(self) -> bool:
        """False when the payload carries a truthy `error`, which is how every shipped
        tool — and the registry itself — reports a failure."""
        return not (isinstance(self.payload, dict) and self.payload.get("error"))

    @property
    def error(self) -> str | None:
        """The error message, or None for a result that is ok."""
        if self.ok:
            return None
        return str(self.payload["error"])

    def with_payload(self, payload: Any) -> ToolResult:
        """The same call with an amended payload — what the figure review returns.

        The original text is dropped on purpose: `for_llm()` must describe the payload
        the model is about to read, amendments included.
        """
        return ToolResult(self.tool, payload)

    def for_llm(self) -> str:
        """The text appended to the history and shown to the model.

        Returns:
            The tool's own string when it returned one; otherwise the payload as JSON
            with non-ASCII kept and unknown types stringified — the serialisation
            `registry.call_tool` has always used, so the model reads the same bytes.
        """
        if self.raw is not None:
            return self.raw
        if isinstance(self.payload, str):
            return self.payload
        return json.dumps(self.payload, ensure_ascii=False, default=str)
