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


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name) if name else structlog.get_logger()
