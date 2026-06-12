"""Tests for helioai.core.llm.base call_with_retry."""

from __future__ import annotations

import pytest

from helioai.core.llm.base import RETRYABLE_STATUS, _error_status, call_with_retry


class _FakeError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _FakeCodeError(Exception):
    """Simulates google.genai error that uses .code instead of .status_code."""

    def __init__(self, code: int):
        super().__init__(f"code={code}")
        self.code = code


# ── _error_status ──────────────────────────────────────────────────────────────


def test_error_status_status_code():
    assert _error_status(_FakeError(429)) == 429


def test_error_status_code_attr():
    assert _error_status(_FakeCodeError(429)) == 429


def test_error_status_none_for_plain_exception():
    assert _error_status(ValueError("oops")) is None


# ── call_with_retry ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retries_twice_then_succeeds():
    calls = []

    async def fn():
        calls.append(1)
        if len(calls) < 3:
            raise _FakeError(429)
        return "ok"

    result = await call_with_retry(fn, attempts=3, base_delay=0)
    assert result == "ok"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_gemini_code_429_retried():
    calls = []

    async def fn():
        calls.append(1)
        if len(calls) < 2:
            raise _FakeCodeError(429)
        return "gemini_ok"

    result = await call_with_retry(fn, attempts=3, base_delay=0)
    assert result == "gemini_ok"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_non_retryable_raises_immediately():
    calls = []

    async def fn():
        calls.append(1)
        raise _FakeError(401)

    with pytest.raises(_FakeError) as exc_info:
        await call_with_retry(fn, attempts=3, base_delay=0)
    assert exc_info.value.status_code == 401
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_exhausted_attempts_reraises():
    calls = []

    async def fn():
        calls.append(1)
        raise _FakeError(429)

    with pytest.raises(_FakeError):
        await call_with_retry(fn, attempts=3, base_delay=0)
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_non_http_exception_raises_immediately():
    calls = []

    async def fn():
        calls.append(1)
        raise ValueError("unexpected")

    with pytest.raises(ValueError):
        await call_with_retry(fn, attempts=3, base_delay=0)
    assert len(calls) == 1


def test_retryable_status_set_contains_expected():
    assert 429 in RETRYABLE_STATUS
    assert 500 in RETRYABLE_STATUS
    assert 401 not in RETRYABLE_STATUS
    assert 404 not in RETRYABLE_STATUS
