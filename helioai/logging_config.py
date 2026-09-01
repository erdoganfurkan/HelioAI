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
    """
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)


def get_logger(name: str | None = None) -> Any:
    """Return a structlog logger, optionally bound to a module name."""
    return structlog.get_logger(name) if name else structlog.get_logger()
