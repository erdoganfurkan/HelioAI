"""OllamaClient placeholder — Phase future (local inference).

Ollama's tool-calling format follows OpenAI conventions natively, so the
conversion will mirror GroqClient. This stub exists to anchor the abstraction.
"""

from __future__ import annotations

from .base import LLMClient, Message, ToolDef


class OllamaClient(LLMClient):
    def __init__(self, base_url: str, model: str):
        raise NotImplementedError(
            "OllamaClient is not yet implemented. "
            "Set HELIOAI_LLM_PROVIDER=groq or gemini in .env."
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system_prompt: str | None = None,
    ) -> Message:
        raise NotImplementedError
