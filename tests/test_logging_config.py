"""Tests for helioai.logging_config.

setup_logging replaces the root logger's handlers globally, so every test here
restores the previous state — otherwise the first test to run would decide how
the rest of the suite logs.
"""

from __future__ import annotations

import json
import logging

import pytest
import structlog

from helioai.logging_config import _format_from_env, get_logger, setup_logging


@pytest.fixture(autouse=True)
def restore_logging_state():
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    saved_structlog = structlog.get_config()
    yield
    root.handlers, root.level = saved_handlers, saved_level
    structlog.configure(**saved_structlog)


# ── format selection ───────────────────────────────────────────────────────────


def test_console_is_the_default(monkeypatch):
    monkeypatch.delenv("HELIOAI_LOG_FORMAT", raising=False)
    assert _format_from_env() == "console"


@pytest.mark.parametrize("value", ["json", "JSON", " json "])
def test_json_is_recognised_case_and_space_insensitively(monkeypatch, value):
    monkeypatch.setenv("HELIOAI_LOG_FORMAT", value)
    assert _format_from_env() == "json"


@pytest.mark.parametrize("value", ["yaml", "", "xml"])
def test_unknown_format_falls_back_to_console(monkeypatch, value):
    """A typo in the env var must not crash the process at startup."""
    monkeypatch.setenv("HELIOAI_LOG_FORMAT", value)
    assert _format_from_env() == "console"


# ── setup_logging ──────────────────────────────────────────────────────────────


def test_setup_installs_exactly_one_root_handler():
    setup_logging("INFO")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert root.level == logging.INFO


def test_repeated_setup_does_not_stack_handlers():
    """Every entry point calls setup_logging; duplicated handlers would double every line."""
    setup_logging("INFO")
    setup_logging("INFO")
    setup_logging("INFO")
    assert len(logging.getLogger().handlers) == 1


@pytest.mark.parametrize(
    ("given", "expected"),
    [("DEBUG", logging.DEBUG), ("warning", logging.WARNING), (logging.ERROR, logging.ERROR)],
)
def test_level_accepts_names_any_case_and_ints(given, expected):
    setup_logging(given)
    assert logging.getLogger().level == expected


def test_unknown_level_name_falls_back_to_info():
    setup_logging("NOT_A_LEVEL")
    assert logging.getLogger().level == logging.INFO


def test_json_format_emits_parseable_lines(monkeypatch, capsys):
    monkeypatch.setenv("HELIOAI_LOG_FORMAT", "json")
    setup_logging("INFO")

    get_logger("test").info("llm_call_end", user_id="vincent", tokens=1234)

    line = capsys.readouterr().err.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["event"] == "llm_call_end"
    assert payload["user_id"] == "vincent"
    assert payload["tokens"] == 1234
    assert payload["level"] == "info"


def test_console_format_is_not_json(monkeypatch, capsys):
    monkeypatch.setenv("HELIOAI_LOG_FORMAT", "console")
    setup_logging("INFO")

    get_logger("test").info("hello_console", extra_field="visible")

    err = capsys.readouterr().err
    assert "hello_console" in err
    with pytest.raises(json.JSONDecodeError):
        json.loads(err.strip().splitlines()[-1])


def test_level_filters_lower_severity(monkeypatch, capsys):
    monkeypatch.setenv("HELIOAI_LOG_FORMAT", "json")
    setup_logging("WARNING")

    log = get_logger("test")
    log.debug("should_not_appear")
    log.warning("should_appear")

    err = capsys.readouterr().err
    assert "should_not_appear" not in err
    assert "should_appear" in err


# ── get_logger ─────────────────────────────────────────────────────────────────


def test_get_logger_with_and_without_a_name():
    assert get_logger("helioai.tools") is not None
    assert get_logger() is not None


# ── level override from the environment ────────────────────────────────────────


def test_env_overrides_the_caller_level(monkeypatch):
    """Entry points hardcode their level; HELIOAI_LOG_LEVEL is the escape hatch.

    Needed because a third party logging at WARNING (speasy's inventory probes)
    cannot otherwise be silenced without editing the CLI.
    """
    monkeypatch.setenv("HELIOAI_LOG_LEVEL", "ERROR")
    setup_logging("WARNING")
    assert logging.getLogger().level == logging.ERROR


def test_caller_level_is_used_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("HELIOAI_LOG_LEVEL", raising=False)
    setup_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING


def test_unknown_env_level_falls_back_to_the_caller_level(monkeypatch):
    """A typo must not silently turn logging up to INFO and flood a demo."""
    monkeypatch.setenv("HELIOAI_LOG_LEVEL", "LOUD")
    setup_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING
