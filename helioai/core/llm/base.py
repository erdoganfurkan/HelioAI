"""Provider-neutral message model and LLMClient interface."""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _error_status(exc: Exception) -> int | None:
    """Return the HTTP status code from an SDK exception, or None."""
    for attr in ("status_code", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    return None


async def call_with_retry(
    fn,
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
):
    """Call async fn with exponential backoff on retryable HTTP errors.

    Non-retryable errors and exhausted attempts are re-raised immediately.
    """
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as exc:
            status = _error_status(exc)
            if status not in RETRYABLE_STATUS:
                raise
            if attempt == attempts - 1:
                raise
            delay = min(base_delay * (2**attempt) + random.uniform(0, 0.5), max_delay)
            log.info(
                "llm_retry",
                extra={"attempt": attempt + 1, "status": status, "delay": round(delay, 2)},
            )
            await asyncio.sleep(delay)


@dataclass
class ToolCall:
    """A tool invocation requested by the model.

    Attributes:
        id: Provider-assigned identifier, echoed back on the matching tool
            result. Gemini has no native ids, so its client synthesises
            `name::hex` and parses the name back out.
        name: Registered tool name.
        arguments: Decoded JSON arguments. Empty when the model emitted
            malformed JSON — a bad tool call must not kill the loop.
    """

    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    """One turn of conversation, in a provider-neutral form.

    Every client converts to and from this shape, so the agent loop, the session
    store and the interfaces never see a provider's wire format.

    Attributes:
        role: Who produced the turn.
        content: Text content. Present alongside `tool_calls` when the model
            narrated what it was about to do.
        tool_calls: Tools the assistant wants invoked, when it requested any.
        tool_call_id: For `tool` messages, the `ToolCall.id` being answered.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


@dataclass
class ToolDef:
    """A tool as advertised to the model.

    Attributes:
        name: Tool name the model will call.
        description: What the tool does and when to reach for it — the model's
            only clue about applicability.
        parameters: JSON Schema object describing the accepted arguments.
    """

    name: str
    description: str
    parameters: dict = field(default_factory=dict)


class LLMClient(ABC):
    """Interface every provider client implements.

    One method, deliberately: the agent loop only ever needs a single completion
    with optional tool calling. Streaming happens at the loop level, not here.
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system_prompt: str | None = None,
        tool_choice: str = "auto",
    ) -> Message:
        """Send one turn and return the assistant's reply.

        Args:
            messages: Conversation history.
            tools: Tools the model may call this turn.
            system_prompt: Instructions placed before the history.
            tool_choice: `auto` to let the model decide, `required` to force a
                tool call — used on a sub-agent's first turn so it cannot answer
                from memory without looking anything up.

        Returns:
            The assistant reply, carrying `tool_calls` when the model requested any.
        """
        raise NotImplementedError
