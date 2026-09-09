"""Custom HTTP headers for OpenAI-compatible providers."""

from __future__ import annotations

from helioai.config import _parse_headers
from helioai.core.llm.factory import _resolve_headers


def test_parse_headers_reads_pairs():
    assert _parse_headers("x-team=plasma, x-run=42") == {"x-team": "plasma", "x-run": "42"}


def test_parse_headers_ignores_entries_without_a_name():
    assert _parse_headers("") == {}
    assert _parse_headers("no-equals-sign") == {}
    assert _parse_headers("=orphan-value") == {}


def test_parse_headers_keeps_an_empty_value():
    """A header a gateway wants present but blank is still a header."""
    assert _parse_headers("x-flag=") == {"x-flag": ""}


def test_uuid_placeholder_is_expanded_per_client():
    """A gateway asking for a per-conversation token must not get the literal string.

    OpenCode's Go gateway is the case that surfaced this: it returns 400 MissingSessionID
    without one. Keeping the vendor out of the code keeps a policy decision out of it too.
    """
    a = _resolve_headers({"x-session": "{uuid}"})["x-session"]
    b = _resolve_headers({"x-session": "{uuid}"})["x-session"]
    assert a != b
    assert "{" not in a


def test_literal_values_pass_through():
    assert _resolve_headers({"x-team": "plasma"}) == {"x-team": "plasma"}


def test_no_headers_configured_means_none():
    assert _resolve_headers(None) == {}
    assert _resolve_headers({}) == {}
