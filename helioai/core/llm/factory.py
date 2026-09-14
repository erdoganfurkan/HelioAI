"""Build the configured LLM client.

Providers that speak the OpenAI wire format are table entries, not classes — see
`OPENAI_COMPAT`. Azure and Gemini need their own SDK client objects and stay
explicit below.
"""

from __future__ import annotations

from uuid import uuid4

from helioai.config import settings
from helioai.core.llm.base import LLMClient

_UUID_TOKEN = "{uuid}"


def _resolve_headers(headers: dict[str, str] | None) -> dict[str, str]:
    """Expand `{uuid}` in configured headers, once per client.

    A gateway that wants a per-conversation token needs a fresh value, not the literal
    string someone put in their .env — so the placeholder is substituted here, where a
    client is built, rather than at config load, where it would be fixed for the life
    of the process.

    ponytail: one id per client, not per conversation. CLI and Jupyter build a client
    per session so the two coincide; the web app builds one per request and splits a
    conversation into several ids. Threading the real session id through `chat()` as a
    per-request header is the upgrade if that ever measures.
    """
    if not headers:
        return {}
    return {
        name: (uuid4().hex if value == _UUID_TOKEN else value) for name, value in headers.items()
    }


# provider -> (settings attribute, base_url template). `key_required` is False for
# local endpoints that authenticate no one.
OPENAI_COMPAT: dict[str, dict] = {
    "groq": {
        "config": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "key_url": " (https://console.groq.com/keys)",
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


def build_llm_client(provider: str | None = None, model: str | None = None) -> LLMClient:
    """Return a client for the requested provider.

    Args:
        provider: Provider name. Defaults to `HELIOAI_LLM_PROVIDER`.
        model: Model (or, on Azure, deployment) to use instead of the provider's
            configured one — how a delegated role runs on a smaller model than the lead
            (`HELIOAI_ROLE_MODELS`). None keeps the configured model.

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
            raise RuntimeError("AZURE_OPENAI_API_KEY is not set in .env")
        if not cfg.endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set in .env")
        return AzureOpenAIClient(
            api_key=cfg.api_key,
            endpoint=cfg.endpoint,
            api_version=cfg.api_version,
            deployment=model or cfg.deployment,
            max_output_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature,
        )

    if p == "gemini":
        from helioai.core.llm.gemini import GeminiClient

        cfg = settings.llm.gemini
        if not cfg.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set in .env (https://aistudio.google.com/apikey)"
            )
        return GeminiClient(
            api_key=cfg.api_key,
            model=model or cfg.model,
            max_output_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature,
        )

    if p in OPENAI_COMPAT:
        from helioai.core.llm.openai_compat import OpenAICompatClient

        spec = OPENAI_COMPAT[p]
        cfg = getattr(settings.llm, spec["config"])
        api_key = getattr(cfg, "api_key", "")
        if spec["key_env"] and not api_key:
            raise RuntimeError(f"{spec['key_env']} is not set in .env{spec.get('key_url', '')}")
        base_url = spec["base_url"] or f"{getattr(cfg, 'base_url', '').rstrip('/')}/v1"
        return OpenAICompatClient(
            provider=p,
            model=model or cfg.model,
            api_key=api_key,
            base_url=base_url,
            max_output_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature,
            default_headers=_resolve_headers(getattr(cfg, "headers", None)),
        )

    known = "|".join(["azure", "gemini", *OPENAI_COMPAT])
    raise RuntimeError(f"Unknown LLM provider: {p!r}. Use {known}")
