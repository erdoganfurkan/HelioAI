from __future__ import annotations

from helioai.config import settings
from helioai.core.llm.base import LLMClient


def build_llm_client(provider: str | None = None) -> LLMClient:
    p = (provider or settings.llm.provider).lower()

    if p == "azure":
        from helioai.core.llm.azure_openai import AzureOpenAIClient
        cfg = settings.llm.azure
        if not cfg.api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY is not set")
        if not cfg.endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set")
        return AzureOpenAIClient(
            api_key=cfg.api_key,
            endpoint=cfg.endpoint,
            api_version=cfg.api_version,
            deployment=cfg.deployment,
            max_output_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature,
        )

    if p == "groq":
        from helioai.core.llm.groq import GroqClient
        cfg = settings.llm.groq
        if not cfg.api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        return GroqClient(
            api_key=cfg.api_key,
            model=cfg.model,
            max_output_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature,
        )

    if p == "gemini":
        from helioai.core.llm.gemini import GeminiClient
        cfg = settings.llm.gemini
        if not cfg.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        return GeminiClient(
            api_key=cfg.api_key,
            model=cfg.model,
            max_output_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature,
        )

    raise RuntimeError(f"Unknown LLM provider: {p!r}. Use azure|groq|gemini")
