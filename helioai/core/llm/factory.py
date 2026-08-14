"""Build the configured LLM client.

Providers that speak the OpenAI wire format are table entries, not classes — see
`OPENAI_COMPAT`. Azure and Gemini need their own SDK client objects and stay
explicit below.
"""

from __future__ import annotations

from helioai.config import settings
from helioai.core.llm.base import LLMClient

# provider -> (settings attribute, base_url template). `key_required` is False for
# local endpoints that authenticate no one.
OPENAI_COMPAT: dict[str, dict] = {
    "groq": {
        "config": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
    },
    "opencode": {
        "config": "opencode",
        "base_url": None,  # taken from settings.llm.opencode.base_url (Zen gateway, generic)
        "key_env": "OPENCODE_API_KEY",
    },
    "ollama": {
        "config": "ollama",
        "base_url": None,  # taken from settings.llm.ollama.base_url
        "key_env": None,
    },
}


def build_llm_client(provider: str | None = None) -> LLMClient:
    """Return a client for the requested provider.

    Args:
        provider: Provider name. Defaults to `HELIOAI_LLM_PROVIDER`.

    Returns:
        A ready-to-use client.

    Raises:
        RuntimeError: If the provider is unknown or its API key is missing.

    Example:
        >>> llm = build_llm_client("groq")
        >>> type(llm).__name__
        'OpenAICompatClient'
    """
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

    if p in OPENAI_COMPAT:
        from helioai.core.llm.openai_compat import OpenAICompatClient

        spec = OPENAI_COMPAT[p]
        cfg = getattr(settings.llm, spec["config"])
        api_key = getattr(cfg, "api_key", "")
        if spec["key_env"] and not api_key:
            raise RuntimeError(f"{spec['key_env']} is not set")
        base_url = spec["base_url"] or f"{getattr(cfg, 'base_url', '').rstrip('/')}/v1"
        return OpenAICompatClient(
            provider=p,
            model=cfg.model,
            api_key=api_key,
            base_url=base_url,
            max_output_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature,
        )

    known = "|".join(["azure", "gemini", *OPENAI_COMPAT])
    raise RuntimeError(f"Unknown LLM provider: {p!r}. Use {known}")
