"""A failed turn reads as one actionable sentence, the same in every interface."""

from __future__ import annotations

import httpx
import openai
import pytest

from helioai.config import settings
from helioai.interfaces.errors import describe_llm_error, setup_problem


def _request(url: str = "http://localhost:11434/v1/chat/completions") -> httpx.Request:
    return httpx.Request("POST", url)


def _status_error(cls, status: int):
    response = httpx.Response(status, request=_request("https://api.example/v1/chat/completions"))
    return cls("boom", response=response, body=None)


def test_missing_key_names_the_variable_and_the_way_out():
    msg = describe_llm_error(RuntimeError("AZURE_OPENAI_API_KEY is not set in .env"), "azure")

    assert "AZURE_OPENAI_API_KEY is not set" in msg
    assert "in .env" not in msg, "said .env to users who have none"
    assert "HELIOAI_LLM_PROVIDER" in msg
    assert "helioai doctor" in msg


def test_unreachable_ollama_says_where_and_how_to_start_it():
    exc = openai.APIConnectionError(request=_request())

    msg = describe_llm_error(exc, "ollama")

    assert "http://localhost:11434/v1" in msg
    assert "ollama serve" in msg
    assert "Connection error" not in msg


def test_unreachable_hosted_provider_does_not_mention_ollama():
    exc = openai.APIConnectionError(request=_request("https://opencode.ai/zen/go/v1/chat/x"))

    msg = describe_llm_error(exc, "opencode")

    assert "could not reach the opencode server at https://opencode.ai/zen/go/v1" in msg
    assert "ollama" not in msg.lower()


def test_a_timeout_is_not_reported_as_a_refusal():
    exc = openai.APITimeoutError(request=_request())

    assert "timed out" in describe_llm_error(exc, "ollama")


@pytest.mark.parametrize(
    ("cls", "status", "expected"),
    [
        (openai.AuthenticationError, 401, "check OPENCODE_API_KEY"),
        (openai.PermissionDeniedError, 403, "refused the credentials"),
        (openai.NotFoundError, 404, "HELIOAI_OPENCODE_MODEL"),
        (openai.RateLimitError, 429, "HTTP 429"),
        (openai.InternalServerError, 503, "provider's side"),
        (openai.BadRequestError, 400, "HTTP 400): boom"),
    ],
)
def test_http_failures_are_named_by_what_to_do(cls, status, expected):
    assert expected in describe_llm_error(_status_error(cls, status), "opencode")


def test_a_runtime_error_written_for_humans_is_left_alone():
    text = "The model returned an empty answer; raise HELIOAI_MAX_OUTPUT_TOKENS"

    assert describe_llm_error(RuntimeError(text), "groq") == text


def test_an_unknown_error_keeps_its_type_and_points_at_the_traceback():
    msg = describe_llm_error(KeyError("reply"), "groq")

    assert "KeyError" in msg and "HELIOAI_LOG_LEVEL=DEBUG" in msg


def test_opencode_without_a_model_is_caught_before_the_request(monkeypatch):
    monkeypatch.setattr(settings.llm.opencode, "model", "")

    assert "HELIOAI_OPENCODE_MODEL" in setup_problem("opencode")

    monkeypatch.setattr(settings.llm.opencode, "model", "deepseek-v4-pro")
    assert setup_problem("opencode") is None
    assert setup_problem("ollama") is None


def test_the_configured_provider_is_used_when_none_is_given(monkeypatch):
    monkeypatch.setattr(settings.llm, "provider", "ollama")

    assert "ollama serve" in describe_llm_error(openai.APIConnectionError(request=_request()))
