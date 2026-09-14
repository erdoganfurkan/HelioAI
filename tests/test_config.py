"""Settings loading: what the environment variables actually change.

`config._load()` is re-run under a patched environment here. The module-level `settings`
singleton is deliberately left alone — every other test reads it — so these tests inspect
the fresh `Settings` object `_load()` returns.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from helioai import config
from helioai.config import LLMConfig

PROVIDERS = [f.name for f in fields(LLMConfig) if f.name != "provider"]

DEFAULT_MAX_OUTPUT = {
    "azure": 8192,
    "gemini": 4096,
    "groq": 4096,
    "opencode": 16384,
    "ollama": 4096,
}


@pytest.mark.parametrize("provider", PROVIDERS)
def test_max_output_tokens_override_reaches_every_provider(monkeypatch, provider):
    """HELIOAI_MAX_OUTPUT_TOKENS is "one knob for every provider" — including the one
    the README recommends. The override used to iterate a hand-written tuple that
    predated opencode, so the exact setting the empty-response error points to did
    nothing for it."""
    monkeypatch.setenv("HELIOAI_MAX_OUTPUT_TOKENS", "32768")
    s = config._load()
    assert getattr(s.llm, provider).max_output_tokens == 32768


@pytest.mark.parametrize("provider", PROVIDERS)
def test_without_override_each_provider_keeps_its_own_default(monkeypatch, provider):
    """The per-provider defaults are decisions (reasoning budgets differ) — see the
    dataclass docstrings in config.py — and must survive an unset variable."""
    monkeypatch.delenv("HELIOAI_MAX_OUTPUT_TOKENS", raising=False)
    s = config._load()
    assert getattr(s.llm, provider).max_output_tokens == DEFAULT_MAX_OUTPUT[provider]


def test_non_numeric_override_is_ignored(monkeypatch):
    monkeypatch.setenv("HELIOAI_MAX_OUTPUT_TOKENS", "lots")
    s = config._load()
    assert s.llm.azure.max_output_tokens == DEFAULT_MAX_OUTPUT["azure"]
