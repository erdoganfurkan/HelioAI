"""Tests for the scope guardrail and dev-token unlock mechanism."""

from __future__ import annotations


# ── build_lead_system_prompt ──────────────────────────────────────────────────


def test_restricted_prompt_contains_guardrail():
    from helioai.core.agent_loop import SCOPE_GUARDRAIL, build_lead_system_prompt

    prompt = build_lead_system_prompt(restricted=True)
    assert SCOPE_GUARDRAIL in prompt


def test_unrestricted_prompt_has_no_guardrail():
    from helioai.core.agent_loop import SCOPE_GUARDRAIL, build_lead_system_prompt

    prompt = build_lead_system_prompt(restricted=False)
    assert SCOPE_GUARDRAIL not in prompt


def test_unrestricted_prompt_still_has_system_prompt():
    from helioai.core.agent_loop import SYSTEM_PROMPT, build_lead_system_prompt

    prompt = build_lead_system_prompt(restricted=False)
    assert SYSTEM_PROMPT in prompt


def test_restricted_prompt_still_has_system_prompt():
    from helioai.core.agent_loop import SYSTEM_PROMPT, build_lead_system_prompt

    prompt = build_lead_system_prompt(restricted=True)
    assert SYSTEM_PROMPT in prompt


# ── dev_unlock ────────────────────────────────────────────────────────────────


def test_dev_unlock_correct_token(monkeypatch):
    monkeypatch.setattr("helioai.config.settings.dev.token", "s3cr3t")
    from helioai.config import dev_unlock

    assert dev_unlock("s3cr3t") is True


def test_dev_unlock_wrong_token(monkeypatch):
    monkeypatch.setattr("helioai.config.settings.dev.token", "s3cr3t")
    from helioai.config import dev_unlock

    assert dev_unlock("wrong") is False


def test_dev_unlock_none_token(monkeypatch):
    monkeypatch.setattr("helioai.config.settings.dev.token", "s3cr3t")
    from helioai.config import dev_unlock

    assert dev_unlock(None) is False


def test_dev_unlock_empty_server_token(monkeypatch):
    """When no server token is configured, nothing can unlock."""
    monkeypatch.setattr("helioai.config.settings.dev.token", "")
    from helioai.config import dev_unlock

    assert dev_unlock("anything") is False
    assert dev_unlock("") is False
    assert dev_unlock(None) is False
