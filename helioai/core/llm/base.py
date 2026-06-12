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
    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict = field(default_factory=dict)


class LLMClient(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system_prompt: str | None = None,
        tool_choice: str = "auto",
    ) -> Message:
        raise NotImplementedError
