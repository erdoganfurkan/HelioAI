"""One sentence for a failed turn, shared by the CLI, the web UI and Jupyter.

The first thing a new install does is fail: no key, a local server not started, a key
pasted with a space. Each interface used to show that failure its own way — 264 lines of
traceback in the terminal, the SDK's bare "Connection error." in the browser — and none
of them said what to do next. The translation lives here, once, so the three surfaces say
the same thing and the fix is written next to the problem.
"""

from __future__ import annotations

_KEY_ENV = {
    "azure": "AZURE_OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "opencode": "OPENCODE_API_KEY",
}

_MODEL_ENV = {
    "azure": "AZURE_OPENAI_DEPLOYMENT",
    "opencode": "HELIOAI_OPENCODE_MODEL",
    "ollama": "HELIOAI_OLLAMA_MODEL",
}

_DOCTOR = "`helioai doctor` checks the whole setup."


def _provider(provider: str | None) -> str:
    from helioai.config import settings

    return (provider or settings.llm.provider).lower()


def _endpoint(provider: str, exc: BaseException) -> str:
    request = getattr(exc, "request", None)
    url = getattr(request, "url", None)
    if url is not None:
        return str(url).split("/chat/", 1)[0]
    from helioai.config import settings

    cfg = getattr(settings.llm, provider, None)
    return getattr(cfg, "base_url", "") or getattr(cfg, "endpoint", "") or ""


def setup_problem(provider: str | None = None) -> str | None:
    """What stops `provider` from answering before a single request is sent, or None.

    `build_llm_client` already refuses a missing key. A missing *model* it lets through,
    on purpose for OpenCode: the gateway has no sensible default, so its `model` is empty
    until the user picks one — and the request then fails at the gateway with an error
    that does not name the variable to set. Checked here, before the question is spent.
    """
    from helioai.config import settings

    p = _provider(provider)
    cfg = getattr(settings.llm, p, None)
    if p == "opencode" and cfg is not None and not cfg.model:
        return (
            "HELIOAI_OPENCODE_MODEL is not set — OpenCode has no default model. "
            "Set it to a model id from your OpenCode dashboard (e.g. deepseek-v4-pro)."
        )
    return None


def describe_llm_error(exc: BaseException, provider: str | None = None) -> str:
    """Turn an exception raised while answering a question into one actionable line.

    Args:
        exc: What `build_llm_client` or the agent loop raised.
        provider: The provider the turn ran on; the configured one when omitted.

    Returns:
        A sentence saying what failed and how to fix it. Errors this function does not
        recognise keep their type and message, and point at DEBUG logging for the
        traceback, rather than being dressed up as a diagnosis they are not.
    """
    p = _provider(provider)
    message = str(exc).strip()

    if isinstance(exc, RuntimeError) and not _status(exc):
        if " is not set" in message:
            message = message.replace(" is not set in .env", " is not set")
            return (
                f"{message}. Set it in your shell or in a .env file, or choose another "
                f"provider with HELIOAI_LLM_PROVIDER. {_DOCTOR}"
            )
        if message.startswith("Unknown LLM provider"):
            return f"{message}. {_DOCTOR}"
        return message

    name = type(exc).__name__
    if name in ("APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout"):
        where = _endpoint(p, exc)
        where = f" at {where}" if where else ""
        hint = (
            " Is Ollama running? Start it with `ollama serve`."
            if p == "ollama"
            else " Check your network connection and the endpoint URL."
        )
        verb = "timed out reaching" if "Timeout" in name else "could not reach"
        return f"HelioAI {verb} the {p} server{where}.{hint}"

    status = _status(exc)
    if status in (401, 403):
        key = _KEY_ENV.get(p)
        fix = f" — check {key}" if key else ""
        return f"The {p} server refused the credentials (HTTP {status}){fix}. {_DOCTOR}"
    if status == 404:
        model = _MODEL_ENV.get(p)
        fix = f" Check {model}." if model else ""
        return f"The {p} server does not know the requested model (HTTP 404).{fix}"
    if status == 429:
        return (
            f"The {p} server is rate-limiting this key or its quota is used up (HTTP 429). "
            "Wait a moment, or switch provider with HELIOAI_LLM_PROVIDER."
        )
    if status is not None and status >= 500:
        return f"The {p} server failed (HTTP {status}). This is on the provider's side; retry."
    if status is not None:
        return f"The {p} server rejected the request (HTTP {status}): {message}"

    detail = f"{name}: {message}" if message else name
    return f"Unexpected error — {detail}. Set HELIOAI_LOG_LEVEL=DEBUG to see the traceback."


def _status(exc: BaseException) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value < 600:
            return value
    return None
