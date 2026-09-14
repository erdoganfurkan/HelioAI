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


def test_data_dir_override_drives_every_derived_path(monkeypatch, tmp_path):
    """`HELIOAI_DATA_DIR` promised (docs/installation.md) to relocate user data. It only
    moved `data_dir`; the index, the catalogues and the profile kept deriving from the
    module-level default computed before the variable was read — so a Docker volume
    ended up holding two trees."""
    monkeypatch.setenv("HELIOAI_DATA_DIR", str(tmp_path / "elsewhere"))
    for var in ("HELIOAI_PROFILE", "HELIOAI_CATALOGS_DIR"):
        monkeypatch.delenv(var, raising=False)
    s = config._load()
    root = tmp_path / "elsewhere"
    assert s.data_dir == root
    assert s.rag.chroma_dir == root / "chroma"
    assert s.catalogs.catalogs_dir == root / "catalogs"
    assert s.profile.profile_path == root / "profile.md"


def test_explicit_path_overrides_still_win_over_the_derivation(monkeypatch, tmp_path):
    monkeypatch.setenv("HELIOAI_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("HELIOAI_PROFILE", str(tmp_path / "me.md"))
    monkeypatch.setenv("HELIOAI_CATALOGS_DIR", str(tmp_path / "cats"))
    s = config._load()
    assert s.profile.profile_path == tmp_path / "me.md"
    assert s.catalogs.catalogs_dir == tmp_path / "cats"
    assert s.rag.chroma_dir == tmp_path / "d" / "chroma"


def test_without_override_nothing_moves(monkeypatch):
    monkeypatch.delenv("HELIOAI_DATA_DIR", raising=False)
    s = config._load()
    assert s.rag.chroma_dir == s.data_dir / "chroma"
    assert s.catalogs.catalogs_dir == s.data_dir / "catalogs"
    assert s.profile.profile_path == s.data_dir / "profile.md"


def test_public_unauthenticated_opt_out_defaults_off(monkeypatch):
    monkeypatch.delenv("HELIOAI_ALLOW_UNAUTHENTICATED_PUBLIC", raising=False)
    assert config._load().web_auth.allow_unauthenticated_public is False
    monkeypatch.setenv("HELIOAI_ALLOW_UNAUTHENTICATED_PUBLIC", "1")
    assert config._load().web_auth.allow_unauthenticated_public is True
