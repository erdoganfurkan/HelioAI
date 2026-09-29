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
        return _respond(kwargs, self.response)


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


def _respond(kwargs: dict, response):
    """What the SDK hands back for `kwargs`: chunks when the request streams.

    `chat()` streams underneath, so a whole completion is replayed as the chunks a
    streaming endpoint would send — reasoning, text, tool calls, finish, usage.
    """
    if not kwargs.get("stream") or isinstance(response, _Stream):
        return response
    choice = response.choices[0]
    msg = choice.message
    tcs = msg.tool_calls or []
    chunks = []
    if getattr(msg, "reasoning_content", None):
        chunks.append(_chunk(reasoning=msg.reasoning_content))
    if msg.content:
        chunks.append(_chunk(content=msg.content))
    if tcs:
        chunks.append(
            _chunk(
                tool_calls=[
                    (i, tc.id, tc.function.name, tc.function.arguments) for i, tc in enumerate(tcs)
                ]
            )
        )
    finish = getattr(choice, "finish_reason", None) or ("tool_calls" if tcs else "stop")
    chunks.append(_chunk(finish_reason=finish))
    if getattr(response, "usage", None):
        chunks.append(_chunk(usage=response.usage))
    return _Stream(chunks)


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


def test_factory_model_override_wins_over_the_configured_model(monkeypatch):
    """`HELIOAI_ROLE_MODELS` runs a role on another model of the same provider; the
    factory must honour an explicit model over the provider's configured one, and leave
    the configured one alone when none is given."""
    from helioai.config import settings
    from helioai.core.llm.factory import build_llm_client

    monkeypatch.setattr(settings.llm.groq, "api_key", "gsk_test")
    monkeypatch.setattr(settings.llm.groq, "model", "llama-3.3-70b-versatile")
    assert build_llm_client("groq", model="llama-3.1-8b-instant")._model == "llama-3.1-8b-instant"
    assert build_llm_client("groq")._model == "llama-3.3-70b-versatile"
    assert settings.llm.groq.model == "llama-3.3-70b-versatile"


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


@pytest.mark.asyncio
async def test_a_rejected_tool_choice_falls_back_to_auto():
    """DeepSeek v4 in thinking mode 400s on `required`; losing the turn is worse.

    Whether a model accepts a forced tool call depends on the model and its reasoning
    mode, not on the provider, so the client asks rather than consulting a table.
    """
    from openai import BadRequestError

    client, fake = _groq_client()

    class _Resp:
        status_code = 400
        headers: dict = {}
        request = None

        def json(self):
            return {"error": {"message": "Thinking mode does not support this tool_choice"}}

    async def flaky(**kwargs):
        fake.completions.calls.append(kwargs)
        if kwargs.get("tool_choice") == "required":
            raise BadRequestError(
                "Thinking mode does not support this tool_choice", response=_Resp(), body=None
            )
        return _respond(kwargs, _openai_response(content="ok"))

    fake.completions.create = flaky
    tools = [ToolDef(name="t", description="d", parameters={"type": "object", "properties": {}})]
    await client.chat([Message(role="user", content="hi")], tools, tool_choice="required")

    assert [c.get("tool_choice") for c in fake.calls] == ["required", "auto"]


@pytest.mark.asyncio
async def test_an_empty_turn_is_retried_once():
    """Neither text nor a tool call is transient on a reasoning model, not an answer.

    The whole output allowance can go into hidden reasoning and leave nothing to emit;
    the identical request replayed came back with two tool calls. The agent loop treats
    an empty turn as fatal, so without this the question is abandoned.
    """
    client, fake = _groq_client()
    replies = [_openai_response(content=""), _openai_response(content="second time lucky")]

    async def flaky(**kwargs):
        fake.completions.calls.append(kwargs)
        return _respond(kwargs, replies[len(fake.completions.calls) - 1])

    fake.completions.create = flaky
    msg = await client.chat([Message(role="user", content="hi")], [])

    assert len(fake.calls) == 2, "the empty turn should have been retried"
    assert msg.content == "second time lucky"


@pytest.mark.asyncio
async def test_a_turn_with_tool_calls_but_no_text_is_not_retried():
    """Empty content next to a tool call is the normal shape of a working turn."""
    client, fake = _groq_client()
    fake.completions.response = _openai_response(
        content="", tool_calls=[("c1", "search_parameters", '{"queries": ["x"]}')]
    )
    msg = await client.chat([Message(role="user", content="hi")], [])

    assert len(fake.calls) == 1, "a tool call is an answer; retrying would double the work"
    assert [t.name for t in msg.tool_calls] == ["search_parameters"]


# ── inbound: token accounting ──────────────────────────────────────────────────


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_token_counts_reach_the_agent_loop(build):
    """The response already carries the cost; throwing it away made it unmeasurable.

    HelioBench had to wrap the SDK client to recover numbers that were sitting on
    `response.usage` all along.
    """
    client, fake = build()
    fake.completions.response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
        usage=SimpleNamespace(
            prompt_tokens=11250,
            completion_tokens=84,
            prompt_tokens_details=SimpleNamespace(cached_tokens=8192),
        ),
    )

    reply = await client.chat([Message(role="user", content="hi")], tools=[])

    assert (reply.prompt_tokens, reply.completion_tokens, reply.cached_tokens) == (11250, 84, 8192)


@pytest.mark.parametrize("build", OPENAI_CLIENTS)
@pytest.mark.asyncio
async def test_a_provider_reporting_no_usage_costs_zero_not_a_crash(build):
    """Not every OpenAI-compatible endpoint fills `usage`, and none fill it always."""
    client, fake = build()
    fake.completions.response = _openai_response(content="ok")

    reply = await client.chat([Message(role="user", content="hi")], tools=[])

    assert (reply.prompt_tokens, reply.completion_tokens, reply.cached_tokens) == (0, 0, 0)


# ── streaming ──────────────────────────────────────────────────────────────────


def _chunk(content=None, tool_calls=None, finish_reason=None, usage=None, reasoning=None):
    """One OpenAI-shaped stream chunk. tool_calls entries: (index, id, name, arguments)."""
    frags = [
        SimpleNamespace(index=i, id=tc_id, function=SimpleNamespace(name=name, arguments=args))
        for i, tc_id, name, args in (tool_calls or [])
    ]
    delta = SimpleNamespace(content=content, tool_calls=frags or None)
    if reasoning is not None:
        delta.reasoning_content = reasoning
    choices = (
        [SimpleNamespace(delta=delta, finish_reason=finish_reason)]
        if (content is not None or frags or finish_reason or reasoning is not None)
        else []
    )
    return SimpleNamespace(choices=choices, usage=usage)


class _Stream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _streaming_client(chunks):
    client, fake = _groq_client()
    fake.completions.response = _Stream(chunks)
    return client, fake


async def _collect(client, tools=()):
    items = [
        item
        async for item in client.stream_chat(
            [Message(role="user", content="q")], list(tools), system_prompt="be brief"
        )
    ]
    return items[:-1], items[-1]


async def test_streaming_yields_text_deltas_then_the_same_message_chat_would_return():
    usage = SimpleNamespace(
        prompt_tokens=120,
        completion_tokens=9,
        prompt_tokens_details=SimpleNamespace(cached_tokens=40),
    )
    client, fake = _streaming_client(
        [
            _chunk(content="θ_Bn "),
            _chunk(content="= 57.5°"),
            _chunk(finish_reason="stop"),
            _chunk(usage=usage),
        ]
    )
    deltas, final = await _collect(client)
    assert deltas == ["θ_Bn ", "= 57.5°"]
    assert final.role == "assistant" and final.content == "θ_Bn = 57.5°"
    assert final.tool_calls is None
    assert (final.prompt_tokens, final.completion_tokens, final.cached_tokens) == (120, 9, 40)
    sent = fake.calls[-1]
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}
    assert sent["messages"][0] == {"role": "system", "content": "be brief"}


async def test_streaming_reassembles_tool_call_fragments_by_index():
    client, _ = _streaming_client(
        [
            _chunk(tool_calls=[(0, "call_a", "get_timeseries", '{"param_id": "amda/')]),
            _chunk(tool_calls=[(1, "call_b", "load_recipe", '{"name": "theta_bn"}')]),
            _chunk(tool_calls=[(0, None, None, 'imf", "start": "2015-03-17T03:30:00"}')]),
            _chunk(finish_reason="tool_calls"),
        ]
    )
    deltas, final = await _collect(client)
    assert deltas == []
    assert [(tc.id, tc.name) for tc in final.tool_calls] == [
        ("call_a", "get_timeseries"),
        ("call_b", "load_recipe"),
    ]
    assert final.tool_calls[0].arguments == {"param_id": "amda/imf", "start": "2015-03-17T03:30:00"}
    assert final.tool_calls[1].arguments == {"name": "theta_bn"}


async def test_streaming_holds_back_an_inline_reasoning_block():
    """A reasoning model thinks inline; the reader must not watch the thinking scroll by,
    and the final content is stripped exactly as chat() strips it."""
    client, _ = _streaming_client(
        [
            _chunk(content="<think>the shock is"),
            _chunk(content=" quasi-perp</think>"),
            _chunk(content="θ_Bn ≈ 60°."),
            _chunk(finish_reason="stop"),
        ]
    )
    deltas, final = await _collect(client)
    assert "".join(deltas) == "θ_Bn ≈ 60°."
    assert "<think>" not in "".join(deltas)
    assert final.content == "θ_Bn ≈ 60°."


async def test_a_streamed_turn_with_nothing_in_it_is_retried_once_still_streamed():
    """The retry streams too: a non-streamed reply loses `reasoning_content` on the
    OpenCode gateway, and DeepSeek then rejects the next request."""
    client, fake = _groq_client()
    streams = [
        _Stream([_chunk(content=""), _chunk(finish_reason="stop")]),
        _Stream(
            [_chunk(reasoning="r"), _chunk(content="second try"), _chunk(finish_reason="stop")]
        ),
    ]

    async def create(**kwargs):
        fake.calls.append(kwargs)
        return streams[len(fake.calls) - 1]

    fake.completions.create = create
    deltas, final = await _collect(client)
    assert final.content == "second try" and final.reasoning == "r"
    assert [c.get("stream") for c in fake.calls] == [True, True]


async def test_chat_streams_underneath_so_the_reasoning_is_not_lost():
    """0 of 24 non-streamed DeepSeek replies carried `reasoning_content` through the
    OpenCode gateway on 2026-09-29, 24 of 24 streamed ones did; the sub-agents call
    `chat()`, and died on the 400 that followed."""
    client, fake = _streaming_client(
        [
            _chunk(reasoning="Wind first."),
            _chunk(tool_calls=[(0, "c1", "get_timeseries", "{}")]),
            _chunk(finish_reason="tool_calls"),
        ]
    )
    reply = await client.chat([Message(role="user", content="q")], [])
    assert fake.calls[-1]["stream"] is True
    assert reply.reasoning == "Wind first." and reply.tool_calls[0].id == "c1"


async def test_an_endpoint_that_refuses_streaming_is_asked_again_without_it():
    from openai import BadRequestError

    client, fake = _groq_client()

    class _Resp:
        status_code = 400
        headers: dict = {}
        request = None

        def json(self):
            return {"error": {"message": "stream is not supported"}}

    async def create(**kwargs):
        fake.calls.append(kwargs)
        if kwargs.get("stream"):
            raise BadRequestError("stream is not supported", response=_Resp(), body=None)
        return _openai_response(content="whole")

    fake.completions.create = create
    reply = await client.chat([Message(role="user", content="q")], [])
    assert reply.content == "whole"
    assert [c.get("stream") for c in fake.calls] == [True, None]


async def test_a_client_without_streaming_support_yields_the_reply_whole():
    """The base default: `chat()` in one piece, so every provider works behind a caller
    that streams."""
    from helioai.core.llm.base import LLMClient

    class Plain(LLMClient):
        async def chat(self, messages, tools, system_prompt=None, tool_choice="auto"):
            return Message(role="assistant", content="whole")

    items = [item async for item in Plain().stream_chat([], [], system_prompt="s")]
    assert len(items) == 1 and items[0].content == "whole"


# ── reasoning_content (DeepSeek thinking mode) ─────────────────────────────────


async def test_reasoning_content_is_kept_and_sent_back_with_its_tool_calls():
    """DeepSeek in thinking mode 400s a request carrying `tools` whose history lacks the
    `reasoning_content` of an earlier assistant turn. HelioAI dropped it on every reply,
    so the lead died mid-question on 2 of ~51 calls on 28–29/09."""
    client, fake = _groq_client()
    response = _openai_response(tool_calls=[("c1", "search_parameters", '{"queries": ["imf"]}')])
    response.choices[0].message.reasoning_content = "The user wants IMF data."
    fake.completions.response = response
    tools = [ToolDef(name="search_parameters", description="d")]

    reply = await client.chat([Message(role="user", content="q")], tools)
    assert reply.reasoning == "The user wants IMF data."

    fake.completions.response = _openai_response(content="done")
    history = [
        Message(role="user", content="q"),
        reply,
        Message(role="tool", tool_call_id="c1", content="{}"),
    ]
    await client.chat(history, tools)
    assistant = fake.calls[-1]["messages"][1]
    assert assistant["reasoning_content"] == "The user wants IMF data."
    assert assistant["tool_calls"][0]["id"] == "c1"


async def test_reasoning_content_is_sent_back_on_a_plain_answer_too():
    """The rule covers every earlier assistant turn, not only the ones that called a tool."""
    client, fake = _groq_client()
    history = [
        Message(role="user", content="q"),
        Message(role="assistant", content="a", reasoning="why a"),
        Message(role="user", content="q2"),
    ]
    await client.chat(history, [ToolDef(name="t", description="d")])
    assert fake.calls[-1]["messages"][1] == {
        "role": "assistant",
        "content": "a",
        "reasoning_content": "why a",
    }


async def test_a_reply_without_reasoning_sends_no_reasoning_field():
    """Providers that never emit the field must never receive it."""
    client, fake = _groq_client()
    fake.completions.response = _openai_response(content="plain")
    reply = await client.chat([Message(role="user", content="q")], [])
    assert reply.reasoning is None

    await client.chat([Message(role="user", content="q"), reply], [])
    assert fake.calls[-1]["messages"][1] == {"role": "assistant", "content": "plain"}


async def test_streamed_reasoning_is_kept_whole_and_never_shown():
    client, _ = _streaming_client(
        [
            _chunk(reasoning="Wind MFI "),
            _chunk(reasoning="first."),
            _chunk(content="Loading."),
            _chunk(tool_calls=[(0, "c1", "get_timeseries", "{}")]),
            _chunk(finish_reason="tool_calls"),
        ]
    )
    deltas, final = await _collect(client)
    assert deltas == ["Loading."]
    assert final.reasoning == "Wind MFI first."
    assert final.content == "Loading."


async def test_a_relayed_upstream_error_is_not_read_as_a_refusal_to_stream():
    """The OpenCode gateway prefixes every relayed error with "Upstream request failed";
    the letters s-t-r-e-a-m in it sent a DeepSeek 400 back unstreamed, losing the
    reasoning the retry needed."""
    from openai import BadRequestError

    client, fake = _groq_client()

    class _Resp:
        status_code = 400
        headers: dict = {}
        request = None

        def json(self):
            return {}

    message = "Upstream request failed: [invalid_request_error] reasoning_content ..."

    async def create(**kwargs):
        fake.calls.append(kwargs)
        raise BadRequestError(message, response=_Resp(), body=None)

    fake.completions.create = create
    with pytest.raises(BadRequestError):
        await client.chat([Message(role="user", content="q")], [])
    assert [c.get("stream") for c in fake.calls] == [True]
