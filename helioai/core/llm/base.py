"""Provider-neutral message model and LLMClient interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


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
    ) -> Message:
        raise NotImplementedError
