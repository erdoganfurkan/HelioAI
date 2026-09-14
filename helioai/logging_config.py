"""Structured logging via structlog.

Output format is selected by HELIOAI_LOG_FORMAT:
  - ``console`` (default): human-friendly, colourised.
  - ``json``: one JSON object per line.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog


def _format_from_env() -> str:
    fmt = os.environ.get("HELIOAI_LOG_FORMAT", "console").strip().lower()
    return fmt if fmt in ("console", "json") else "console"


def setup_logging(level: str | int = "INFO") -> None:
    """Configure structlog and the root logger.

    Output format follows `HELIOAI_LOG_FORMAT`: `console` (default) or `json`.
    Safe to call more than once — every entry point calls it, and repeated calls
    replace the handler rather than stacking duplicates.

    `HELIOAI_LOG_LEVEL` overrides `level`. Every entry point hardcodes its own,
    so without this there is no way to quiet a third party that logs at the same
    level — speasy's inventory probes warn loudly on a provider it then disables,
    which is noise in a recorded session or a demo. An unrecognised value is
    ignored rather than obeyed: a typo must not silently turn logging up.

    Args:
        level: Log level name or numeric value. Unknown names fall back to INFO.
    """
    override = os.environ.get("HELIOAI_LOG_LEVEL", "").strip().upper()
    if override and isinstance(getattr(logging, override, None), int):
        level = getattr(logging, override)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    fmt = _format_from_env()

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if fmt == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    _quiet_third_party_advisories()


class _SpeasyProviderTraceback(logging.Filter):
    """Drop the traceback speasy logs after saying it disabled a provider.

    `_safe_init_provider` logs the failure twice at WARNING: the one-line verdict, then
    `Exception: <full traceback>`. Only the second is dropped — the user still reads
    that a provider is off. The traceback is not lost, only demoted: DEBUG lets it
    through, which is where a speasy bug report is written from.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.WARNING:
            return True
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            return True
        return not record.getMessage().startswith("Exception: Traceback")


def _quiet_third_party_advisories() -> None:
    """Keep other libraries' non-actionable notices out of the agent transcript.

    huggingface_hub echoes the server's `X-HF-Warning` header, so every load of
    the cached embedding model printed "set a HF_TOKEN to enable higher rate
    limits" into the middle of a conversation — twice, since structlog's stdlib
    bridge re-emitted it decorated with the sub-agent context, making it look
    like HelioAI was warning about something.

    Nothing is wrong when it fires: the model is cached and the request is only a
    freshness check. Real HTTP failures still raise, and the notice is still
    visible at DEBUG.

    speasy 1.7.1 cannot build the inventory of its `generic_archive` provider
    (`KeyError: 'master_cdf'`) and disables it, which is right, then logs the whole
    traceback at WARNING — twenty red lines in a notebook, on a provider HelioAI does
    not use. The verdict line stays; the traceback goes.
    """
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
    speasy_dispatch = logging.getLogger("speasy.core.requests_scheduling.request_dispatch")
    if not any(isinstance(f, _SpeasyProviderTraceback) for f in speasy_dispatch.filters):
        speasy_dispatch.addFilter(_SpeasyProviderTraceback())


def get_logger(name: str | None = None) -> Any:
    """Return a structlog logger, optionally bound to a module name.

    Args:
        name: Usually `__name__`. Omitted, the logger carries no module field.

    Returns:
        A structlog bound logger. Its output format follows
        `HELIOAI_LOG_FORMAT` (console or json), decided at configuration time
        rather than here.
    """
    return structlog.get_logger(name) if name else structlog.get_logger()
