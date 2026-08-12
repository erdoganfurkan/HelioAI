"""Characterisation tests for the LLM clients.

These pin the wire-format behaviour of every client through the public `chat()`
API — never through private helpers — so they survive the consolidation of the
per-provider classes into a single OpenAI-compatible client.

Each test asserts on one of two seams:
  * outbound — the kwargs the SDK actually received (`fake.calls[-1]`)
  * inbound  — the neutral `Message` returned to the agent loop
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from helioai.core.llm.base import Message, ToolCall, ToolDef

# ── OpenAI-shaped fakes ────────────────────────────────────────────────────────


class _FakeCompletions:
    def __init__(self):
        self.calls: list[dict] = []
        self.response = _openai_response(content="ok")

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _FakeOpenAIClient:
    def __init__(self):
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)

    @property
    def calls(self) -> list[dict]:
        return self.completions.calls


def _openai_response(content: str | None = None, tool_calls: list[tuple] | None = None):
    """Build an OpenAI-shaped response.

    tool_calls entries are (id, name, arguments_json_string).
    """
    tcs = [
        SimpleNamespace(id=tc_id, function=SimpleNamespace(name=name, arguments=args))
        for tc_id, name, args in (tool_calls or [])
    ]
    message = SimpleNamespace(content=content, tool_calls=tcs or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _groq_client():
    from helioai.core.llm.openai_compat import OpenAICompatClient

    client = OpenAICompatClient(
        provider="groq",
        api_key="test-key",
        model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
    )
    fake = _FakeOpenAIClient()
    client._client = fake
    return client, fake


def _azure_client(temperature: float | None = None):
    from helioai.core.llm.azure_openai import AzureOpenAIClient

    client = AzureOpenAIClient(
        api_key="test-key",
        endpoint="https://example.openai.azure.com",
        api_version="2024-10-21",
        deployment="gpt-4o-deploy",
        temperature=temperature,
    )
    fake = _FakeOpenAIClient()
    client._client = fake
    return client, fake


# Both OpenAI-wire clients must behave identically on everything below.
OPENAI_CLIENTS = [
    pytest.param(_groq_client, id="groq"),
    pytest.param(_azure_client, id="azure"),
]


# ── outbound: message conversion ───────────────────────────────────────────────


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_user_message_passed_through(build):
    client, fake = build()
    await client.chat([Message(role="user", content="hello")], tools=[])
    assert fake.calls[-1]["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_system_messages_in_history_are_dropped(build):
    """A system Message inside the history is skipped; only system_prompt creates one."""
    client, fake = build()
    await client.chat(
        [Message(role="system", content="ignored"), Message(role="user", content="hi")],
        tools=[],
    )
    assert [m["content"] for m in fake.calls[-1]["messages"]] == ["hi"]


@pytest.mark.asyncio
async def test_groq_system_prompt_uses_system_role():
    client, fake = _groq_client()
    await client.chat([Message(role="user", content="hi")], tools=[], system_prompt="be brief")
    assert fake.calls[-1]["messages"][0] == {"role": "system", "content": "be brief"}


@pytest.mark.asyncio
async def test_azure_system_prompt_uses_developer_role():
    """Azure/o-series expect `developer`, not `system`."""
    client, fake = _azure_client()
    await client.chat([Message(role="user", content="hi")], tools=[], system_prompt="be brief")
    assert fake.calls[-1]["messages"][0] == {"role": "developer", "content": "be brief"}


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_assistant_tool_calls_serialised_with_json_arguments(build):
    client, fake = build()
    await client.chat(
        [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(id="call_1", name="get_timeseries", arguments={"id": "b_gse"})
                ],
            )
        ],
        tools=[],
    )
    sent = fake.calls[-1]["messages"][0]
    assert sent["role"] == "assistant"
    assert sent["content"] is None, "empty content must be None, not '', alongside tool_calls"
    assert sent["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_timeseries", "arguments": json.dumps({"id": "b_gse"})},
        }
    ]


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_assistant_text_alongside_tool_calls_is_preserved(build):
    """Regression: all three clients used to drop `content` when tool_calls were present."""
    client, fake = build()
    await client.chat(
        [
            Message(
                role="assistant",
                content="let me look that up",
                tool_calls=[ToolCall(id="call_1", name="search_parameters", arguments={})],
            )
        ],
        tools=[],
    )
    assert fake.calls[-1]["messages"][0]["content"] == "let me look that up"


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_tool_result_message_carries_tool_call_id(build):
    client, fake = build()
    await client.chat(
        [Message(role="tool", content='{"ok": true}', tool_call_id="call_1")],
        tools=[],
    )
    assert fake.calls[-1]["messages"][0] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"ok": true}',
    }


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_tool_result_without_id_sends_empty_string(build):
    client, fake = build()
    await client.chat([Message(role="tool", content="x", tool_call_id=None)], tools=[])
    assert fake.calls[-1]["messages"][0]["tool_call_id"] == ""


# ── outbound: tool schema ──────────────────────────────────────────────────────


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_tools_converted_to_openai_function_schema(build):
    client, fake = build()
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    await client.chat(
        [Message(role="user", content="hi")],
        tools=[ToolDef(name="search", description="Find things", parameters=schema)],
    )
    assert fake.calls[-1]["tools"] == [
        {
            "type": "function",
            "function": {"name": "search", "description": "Find things", "parameters": schema},
        }
    ]


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_tool_without_parameters_gets_empty_object_schema(build):
    client, fake = build()
    await client.chat(
        [Message(role="user", content="hi")],
        tools=[ToolDef(name="ping", description="Ping", parameters={})],
    )
    params = fake.calls[-1]["tools"][0]["function"]["parameters"]
    assert params == {"type": "object", "properties": {}}


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_no_tools_omits_tools_and_tool_choice(build):
    client, fake = build()
    await client.chat([Message(role="user", content="hi")], tools=[])
    assert "tools" not in fake.calls[-1]
    assert "tool_choice" not in fake.calls[-1]


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_tool_choice_forwarded_when_tools_present(build):
    client, fake = build()
    await client.chat(
        [Message(role="user", content="hi")],
        tools=[ToolDef(name="ping", description="Ping", parameters={})],
        tool_choice="required",
    )
    assert fake.calls[-1]["tool_choice"] == "required"


# ── outbound: model and sampling parameters ────────────────────────────────────


@pytest.mark.asyncio
async def test_groq_sends_model_name_and_temperature():
    client, fake = _groq_client()
    await client.chat([Message(role="user", content="hi")], tools=[])
    assert fake.calls[-1]["model"] == "llama-3.3-70b-versatile"
    assert fake.calls[-1]["temperature"] == 0.2


@pytest.mark.asyncio
async def test_azure_sends_deployment_as_model():
    """Azure addresses the deployment name, not the model name."""
    client, fake = _azure_client()
    await client.chat([Message(role="user", content="hi")], tools=[])
    assert fake.calls[-1]["model"] == "gpt-4o-deploy"


@pytest.mark.asyncio
async def test_azure_omits_temperature_when_none():
    """GPT-5 and the o-series reject an explicit temperature."""
    client, fake = _azure_client(temperature=None)
    await client.chat([Message(role="user", content="hi")], tools=[])
    assert "temperature" not in fake.calls[-1]


@pytest.mark.asyncio
async def test_azure_sends_temperature_when_set():
    client, fake = _azure_client(temperature=0.7)
    await client.chat([Message(role="user", content="hi")], tools=[])
    assert fake.calls[-1]["temperature"] == 0.7


# ── inbound: response parsing ──────────────────────────────────────────────────


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_plain_text_response(build):
    client, fake = build()
    fake.completions.response = _openai_response(content="the answer is 42")
    result = await client.chat([Message(role="user", content="hi")], tools=[])
    assert result.role == "assistant"
    assert result.content == "the answer is 42"
    assert result.tool_calls is None


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_inline_reasoning_block_is_stripped(build):
    """MiniMax (via OpenCode Go) inlines its chain of thought into `content` itself,
    wrapped in <think> — no separate reasoning field on this wire format. Left in,
    it is what the user sees and what gets replayed as history every following turn."""
    client, fake = build()
    fake.completions.response = _openai_response(
        content="<think>let me work this out</think>\n\nOK"
    )
    result = await client.chat([Message(role="user", content="hi")], tools=[])
    assert result.content == "OK"


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_unterminated_reasoning_block_strips_to_empty(build):
    """A reasoning-heavy generation that exhausted max_output_tokens before closing
    </think> must not leak raw reasoning as if it were the answer — empty is the
    correct result here, it is what triggers the loop's "output budget" error."""
    client, fake = build()
    fake.completions.response = _openai_response(content="<think>still thinking, ran out of")
    result = await client.chat([Message(role="user", content="hi")], tools=[])
    assert result.content == ""


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_response_without_reasoning_is_untouched(build):
    client, fake = build()
    fake.completions.response = _openai_response(content="no reasoning tags here")
    result = await client.chat([Message(role="user", content="hi")], tools=[])
    assert result.content == "no reasoning tags here"


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_null_content_becomes_empty_string(build):
    client, fake = build()
    fake.completions.response = _openai_response(content=None)
    result = await client.chat([Message(role="user", content="hi")], tools=[])
    assert result.content == ""


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_tool_calls_parsed_from_response(build):
    client, fake = build()
    fake.completions.response = _openai_response(
        content=None,
        tool_calls=[("call_9", "get_timeseries", '{"id": "b_gse", "start": "2005-01-16"}')],
    )
    result = await client.chat([Message(role="user", content="hi")], tools=[])
    assert result.tool_calls == [
        ToolCall(
            id="call_9", name="get_timeseries", arguments={"id": "b_gse", "start": "2005-01-16"}
        )
    ]


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_response_text_kept_alongside_tool_calls(build):
    """The other half of the dropped-content regression, on the inbound path."""
    client, fake = build()
    fake.completions.response = _openai_response(
        content="I'll fetch that", tool_calls=[("call_1", "get_timeseries", "{}")]
    )
    result = await client.chat([Message(role="user", content="hi")], tools=[])
    assert result.content == "I'll fetch that"
    assert len(result.tool_calls) == 1


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_malformed_tool_arguments_degrade_to_empty_dict(build, caplog):
    """A model emitting broken JSON must not crash the agent loop."""
    client, fake = build()
    fake.completions.response = _openai_response(
        content=None, tool_calls=[("call_1", "get_timeseries", "{not json")]
    )
    result = await client.chat([Message(role="user", content="hi")], tools=[])
    assert result.tool_calls[0].arguments == {}
    assert result.tool_calls[0].name == "get_timeseries"
    assert "get_timeseries" in caplog.text


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_empty_tool_arguments_become_empty_dict(build):
    client, fake = build()
    fake.completions.response = _openai_response(
        content=None, tool_calls=[("call_1", "list_missions", "")]
    )
    result = await client.chat([Message(role="user", content="hi")], tools=[])
    assert result.tool_calls[0].arguments == {}


# ── factory: provider resolution ───────────────────────────────────────────────


def test_factory_builds_groq_from_table(monkeypatch):
    from helioai.config import settings
    from helioai.core.llm.factory import build_llm_client

    monkeypatch.setattr(settings.llm.groq, "api_key", "gsk_test")
    client = build_llm_client("groq")
    assert client._model == settings.llm.groq.model
    assert str(client._client.base_url).startswith("https://api.groq.com/openai/v1")


def test_factory_builds_ollama_without_api_key(monkeypatch):
    """Ollama is a local endpoint: it must build with no key at all.

    Previously `OllamaClient` raised NotImplementedError and the factory did not
    even accept the name, while the README advertised it as a working provider.
    """
    from helioai.config import settings
    from helioai.core.llm.factory import build_llm_client

    monkeypatch.setattr(settings.llm.ollama, "api_key", "")
    client = build_llm_client("ollama")
    assert client._model == settings.llm.ollama.model
    assert str(client._client.base_url).rstrip("/").endswith("/v1")


def test_factory_missing_groq_key_raises(monkeypatch):
    from helioai.config import settings
    from helioai.core.llm.factory import build_llm_client

    monkeypatch.setattr(settings.llm.groq, "api_key", "")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        build_llm_client("groq")


def test_factory_builds_opencode_from_table(monkeypatch):
    from helioai.config import settings
    from helioai.core.llm.factory import build_llm_client

    monkeypatch.setattr(settings.llm.opencode, "api_key", "oc_test")
    monkeypatch.setattr(settings.llm.opencode, "model", "kimi-k3")
    client = build_llm_client("opencode")
    assert client._model == "kimi-k3"
    assert str(client._client.base_url).startswith("https://opencode.ai/zen/go/v1")


def test_factory_opencode_base_url_is_overridable(monkeypatch):
    """A different OpenCode access path (BYOK proxy, self-hosted gateway...) must
    not require a code change — same override mechanism as Ollama's."""
    from helioai.config import settings
    from helioai.core.llm.factory import build_llm_client

    monkeypatch.setattr(settings.llm.opencode, "api_key", "oc_test")
    monkeypatch.setattr(settings.llm.opencode, "base_url", "https://example.org/custom")
    client = build_llm_client("opencode")
    assert str(client._client.base_url).startswith("https://example.org/custom/v1")


def test_factory_missing_opencode_key_raises(monkeypatch):
    from helioai.config import settings
    from helioai.core.llm.factory import build_llm_client

    monkeypatch.setattr(settings.llm.opencode, "api_key", "")
    with pytest.raises(RuntimeError, match="OPENCODE_API_KEY"):
        build_llm_client("opencode")


def test_factory_unknown_provider_lists_every_supported_name():
    from helioai.core.llm.factory import build_llm_client

    with pytest.raises(RuntimeError) as exc:
        build_llm_client("not-a-provider")
    message = str(exc.value)
    for name in ("azure", "gemini", "groq", "opencode", "ollama"):
        assert name in message


def test_factory_builds_azure_with_developer_role(monkeypatch):
    from helioai.config import settings
    from helioai.core.llm.factory import build_llm_client

    monkeypatch.setattr(settings.llm.azure, "api_key", "az_test")
    monkeypatch.setattr(settings.llm.azure, "endpoint", "https://example.openai.azure.com")
    client = build_llm_client("azure")
    assert client._system_role == "developer"
    assert client._model == settings.llm.azure.deployment


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_multiple_tool_calls_preserved_in_order(build):
    client, fake = build()
    fake.completions.response = _openai_response(
        content=None,
        tool_calls=[
            ("call_1", "search_parameters", '{"q": "Bz"}'),
            ("call_2", "list_missions", "{}"),
        ],
    )
    result = await client.chat([Message(role="user", content="hi")], tools=[])
    assert [tc.name for tc in result.tool_calls] == ["search_parameters", "list_missions"]


# ── connection pool teardown ───────────────────────────────────────────────────


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_aclose_closes_the_sdk_client(build):
    """Callers build a client per request; the pool must be released explicitly.

    An async pool binds to the loop that used it. Left to the garbage collector,
    the client schedules its own teardown after `asyncio.run` has closed that
    loop, and asyncio surfaces an unretrieved
    `RuntimeError: Event loop is closed` while the sockets stay open.
    """
    client, fake = build()
    closed = []

    async def _aclose():
        closed.append(True)

    fake.close = _aclose
    await client.aclose()
    assert closed == [True]


@pytest.mark.asyncio
async def test_aclose_tolerates_a_sync_close():
    """`google.genai.Client.close` is not a coroutine, unlike openai's."""
    from helioai.core.llm.base import close_sdk_client

    calls = []

    class SyncOnly:
        def close(self):
            calls.append("sync")

    await close_sdk_client(SyncOnly())
    assert calls == ["sync"]


@pytest.mark.asyncio
async def test_aclose_never_raises_during_teardown():
    """A pool that will not close must not crash a finished analysis."""
    from helioai.core.llm.base import close_sdk_client

    class Broken:
        async def close(self):
            raise RuntimeError("event loop is closed")

    await close_sdk_client(Broken())


@pytest.mark.asyncio
async def test_aclose_is_a_noop_without_a_close_method():
    from helioai.core.llm.base import close_sdk_client

    await close_sdk_client(object())


# ── retry / rate limits ────────────────────────────────────────────────────────


def _http_error(status: int, retry_after: str | None = None):
    """An SDK-shaped exception: status code plus the response headers."""
    headers = {} if retry_after is None else {"retry-after": retry_after}
    exc = RuntimeError(f"HTTP {status}")
    exc.status_code = status
    exc.response = SimpleNamespace(headers=headers)
    return exc


@pytest.fixture
def slept(monkeypatch):
    """Record backoff delays instead of waiting them out."""
    import helioai.core.llm.base as base

    delays: list[float] = []

    async def _fake_sleep(d):
        delays.append(d)

    monkeypatch.setattr(base.asyncio, "sleep", _fake_sleep)
    return delays


@pytest.mark.asyncio
async def test_rate_limit_waits_the_delay_the_server_asked_for(slept):
    """A 429 states its window; guessing a shorter one ends the session for nothing."""
    from helioai.core.llm.base import call_with_retry

    attempts = {"n": 0}

    async def fn():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _http_error(429, retry_after="14")
        return "ok"

    assert await call_with_retry(fn) == "ok"
    assert slept == [14.0], "the 14s hint must win over the 1s exponential backoff"


@pytest.mark.asyncio
async def test_a_window_longer_than_max_delay_is_not_waited_out(slept):
    """This test used to assert the opposite, and encoded the bug.

    Capping a one-hour Retry-After to max_delay and retrying anyway does not make the
    quota come back — it just delays the same failure by four minutes of silence.
    """
    from helioai.core.llm.base import call_with_retry

    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise _http_error(429, retry_after="3600")

    with pytest.raises(RuntimeError):
        await call_with_retry(fn, max_delay=60.0)
    assert calls["n"] == 1
    assert slept == []


@pytest.mark.asyncio
async def test_backoff_is_used_when_the_server_gives_no_hint(slept):
    from helioai.core.llm.base import call_with_retry

    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(503)
        return "ok"

    await call_with_retry(fn)
    assert len(slept) == 1 and 1.0 <= slept[0] <= 1.5


@pytest.mark.asyncio
async def test_non_retryable_status_raises_at_once(slept):
    from helioai.core.llm.base import call_with_retry

    async def fn():
        raise _http_error(400)

    with pytest.raises(RuntimeError):
        await call_with_retry(fn)
    assert slept == [], "a 400 is not going to fix itself"


@pytest.mark.asyncio
async def test_an_exhausted_daily_quota_fails_at_once(slept):
    """Azure answers an exhausted daily quota with Retry-After: 35464 (~10 h).

    Capping that to max_delay and retrying anyway bought four minutes of silence and
    then failed regardless — which is what "the agent just spins" looked like.
    """
    from helioai.core.llm.base import call_with_retry

    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise _http_error(429, retry_after="35464")

    with pytest.raises(RuntimeError):
        await call_with_retry(fn, max_delay=60.0)
    assert calls["n"] == 1, "no point retrying a window we will never wait out"
    assert slept == []


@pytest.mark.asyncio
async def test_a_short_window_is_still_waited_out(slept):
    """The per-minute limiter case must keep working."""
    from helioai.core.llm.base import call_with_retry

    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(429, retry_after="14")
        return "ok"

    assert await call_with_retry(fn, max_delay=60.0) == "ok"
    assert slept == [14.0]
