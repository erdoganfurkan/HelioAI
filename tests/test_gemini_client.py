"""Characterisation tests for GeminiClient.

Gemini is the one provider that does not speak the OpenAI wire format, so it kept
its native client when the others were consolidated into `openai_compat`. That
makes it the only conversion path with no shared implementation behind it — and
it was sitting at 0% coverage.

Two Gemini-specific behaviours matter most and are pinned below:
  * tool results are addressed by *name*, because Gemini has no tool-call ids —
    the client synthesises `name::hex` ids and parses the name back out;
  * a system prompt is looked up from the history when not passed explicitly.

As in test_llm_clients, everything is asserted through the public `chat()` API so
the tests survive a future migration to Gemini's OpenAI-compatible endpoint.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from helioai.core.llm.base import Message, ToolCall, ToolDef


class _FakeModels:
    def __init__(self):
        self.calls: list[dict] = []
        self.response = _gemini_response(text="ok")

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _FakeGenAIClient:
    def __init__(self):
        self.models = _FakeModels()
        self.aio = SimpleNamespace(models=self.models)


def _gemini_response(text: str | None = None, function_calls: list[tuple] | None = None):
    """Build a Gemini-shaped response. function_calls entries are (name, args_dict)."""
    parts = []
    for name, args in function_calls or []:
        parts.append(
            SimpleNamespace(function_call=SimpleNamespace(name=name, args=args), text=None)
        )
    if text is not None:
        parts.append(SimpleNamespace(function_call=None, text=text))
    content = SimpleNamespace(parts=parts)
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


@pytest.fixture
def client():
    from helioai.core.llm.gemini import GeminiClient

    c = GeminiClient(api_key="test-key", model="gemini-2.5-flash")
    fake = _FakeGenAIClient()
    c._client = fake
    return c, fake.models


# ── outbound ───────────────────────────────────────────────────────────────────


async def test_user_message_becomes_user_content(client):
    c, models = client
    await c.chat([Message(role="user", content="hello")], tools=[])
    contents = models.calls[-1]["contents"]
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "hello"


async def test_assistant_message_uses_the_model_role(client):
    """Gemini calls the assistant turn `model`, not `assistant`."""
    c, models = client
    await c.chat([Message(role="assistant", content="hi there")], tools=[])
    assert models.calls[-1]["contents"][0].role == "model"


async def test_system_message_is_not_sent_as_content(client):
    c, models = client
    await c.chat(
        [Message(role="system", content="be brief"), Message(role="user", content="hi")],
        tools=[],
    )
    assert [c_.role for c_ in models.calls[-1]["contents"]] == ["user"]


async def test_system_prompt_is_recovered_from_history_when_not_passed(client):
    """The agent loop sometimes leaves the system prompt in the message list."""
    c, models = client
    await c.chat(
        [
            Message(role="system", content="you are a plasma physicist"),
            Message(role="user", content="hi"),
        ],
        tools=[],
    )
    assert models.calls[-1]["config"].system_instruction == "you are a plasma physicist"


def test_explicit_system_prompt_wins():
    import asyncio

    from helioai.core.llm.gemini import GeminiClient

    c = GeminiClient(api_key="k", model="m")
    fake = _FakeGenAIClient()
    c._client = fake
    asyncio.run(
        c.chat(
            [Message(role="system", content="from history"), Message(role="user", content="hi")],
            tools=[],
            system_prompt="explicit",
        )
    )
    assert fake.models.calls[-1]["config"].system_instruction == "explicit"


async def test_assistant_tool_calls_become_function_call_parts(client):
    c, models = client
    await c.chat(
        [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id="get_timeseries::ab12", name="get_timeseries", arguments={"id": "b_gse"}
                    )
                ],
            )
        ],
        tools=[],
    )
    part = models.calls[-1]["contents"][0].parts[0]
    assert part.function_call.name == "get_timeseries"
    assert dict(part.function_call.args) == {"id": "b_gse"}


async def test_tool_result_is_addressed_by_name_parsed_from_the_id(client):
    """Gemini matches responses by function name, so the synthesised id encodes it."""
    c, models = client
    await c.chat(
        [Message(role="tool", content='{"n_points": 42}', tool_call_id="get_timeseries::ab12")],
        tools=[],
    )
    part = models.calls[-1]["contents"][0].parts[0]
    assert part.function_response.name == "get_timeseries"
    assert dict(part.function_response.response) == {"n_points": 42}


async def test_non_json_tool_result_is_wrapped(client):
    c, models = client
    await c.chat([Message(role="tool", content="plain text", tool_call_id="x::1")], tools=[])
    resp = models.calls[-1]["contents"][0].parts[0].function_response.response
    assert dict(resp) == {"result": "plain text"}


async def test_json_scalar_tool_result_is_wrapped(client):
    """Gemini requires a dict response; a bare JSON scalar must not be sent raw."""
    c, models = client
    await c.chat([Message(role="tool", content="42", tool_call_id="x::1")], tools=[])
    resp = models.calls[-1]["contents"][0].parts[0].function_response.response
    assert dict(resp) == {"result": 42}


async def test_tools_become_function_declarations(client):
    c, models = client
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    await c.chat(
        [Message(role="user", content="hi")],
        tools=[ToolDef(name="search", description="Find things", parameters=schema)],
    )
    tools = models.calls[-1]["config"].tools
    decls = tools[0].function_declarations
    assert [d.name for d in decls] == ["search"]
    assert decls[0].description == "Find things"


async def test_no_tools_sends_no_tool_config(client):
    c, models = client
    await c.chat([Message(role="user", content="hi")], tools=[])
    cfg = models.calls[-1]["config"]
    assert cfg.tools is None
    assert cfg.tool_config is None


async def test_tool_choice_required_forces_any_mode(client):
    c, models = client
    await c.chat(
        [Message(role="user", content="hi")],
        tools=[ToolDef(name="ping", description="Ping", parameters={})],
        tool_choice="required",
    )
    cfg = models.calls[-1]["config"]
    assert cfg.tool_config is not None
    assert cfg.tool_config.function_calling_config.mode.upper().endswith("ANY")


async def test_model_and_sampling_parameters_are_sent(client):
    c, models = client
    await c.chat([Message(role="user", content="hi")], tools=[])
    assert models.calls[-1]["model"] == "gemini-2.5-flash"
    assert models.calls[-1]["config"].temperature == 0.2
    assert models.calls[-1]["config"].max_output_tokens == 4096


# ── inbound ────────────────────────────────────────────────────────────────────


async def test_plain_text_response(client):
    c, models = client
    models.response = _gemini_response(text="the answer is 42")
    result = await c.chat([Message(role="user", content="hi")], tools=[])
    assert result.role == "assistant"
    assert result.content == "the answer is 42"
    assert result.tool_calls is None


async def test_function_call_becomes_a_tool_call_with_a_synthesised_id(client):
    c, models = client
    models.response = _gemini_response(function_calls=[("get_timeseries", {"id": "b_gse"})])
    result = await c.chat([Message(role="user", content="hi")], tools=[])
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.name == "get_timeseries"
    assert call.arguments == {"id": "b_gse"}
    assert call.id.startswith("get_timeseries::"), "the id must carry the name back"


async def test_text_is_kept_alongside_function_calls(client):
    """Same regression as the OpenAI clients: content used to be dropped."""
    c, models = client
    models.response = _gemini_response(
        text="let me fetch that", function_calls=[("get_timeseries", {})]
    )
    result = await c.chat([Message(role="user", content="hi")], tools=[])
    assert result.content == "let me fetch that"
    assert len(result.tool_calls) == 1


async def test_multiple_function_calls_get_distinct_ids(client):
    c, models = client
    models.response = _gemini_response(
        function_calls=[("search_parameters", {"q": "Bz"}), ("list_missions", {})]
    )
    result = await c.chat([Message(role="user", content="hi")], tools=[])
    assert [t.name for t in result.tool_calls] == ["search_parameters", "list_missions"]
    assert result.tool_calls[0].id != result.tool_calls[1].id


async def test_empty_candidates_yields_empty_message(client):
    """A safety-blocked or empty completion must not raise IndexError."""
    c, models = client
    models.response = SimpleNamespace(candidates=[])
    result = await c.chat([Message(role="user", content="hi")], tools=[])
    assert result.content == ""
    assert result.tool_calls is None


async def test_round_trip_id_survives_a_tool_reply(client):
    """The id minted on the way in must parse back to the same name on the way out."""
    c, models = client
    models.response = _gemini_response(function_calls=[("power_spectrum", {"n": 1})])
    call = (await c.chat([Message(role="user", content="hi")], tools=[])).tool_calls[0]

    await c.chat([Message(role="tool", content="{}", tool_call_id=call.id)], tools=[])
    part = models.calls[-1]["contents"][0].parts[0]
    assert part.function_response.name == "power_spectrum"


async def test_token_counts_reach_the_agent_loop(client):
    """Gemini names every count differently, which is why it has its own client."""
    c, models = client
    models.response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(function_call=None, text="ok")])
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=4096,
            candidates_token_count=57,
            cached_content_token_count=1024,
        ),
    )

    reply = await c.chat([Message(role="user", content="hi")], tools=[])

    assert (reply.prompt_tokens, reply.completion_tokens, reply.cached_tokens) == (4096, 57, 1024)


async def test_missing_usage_metadata_costs_zero(client):
    c, models = client
    reply = await c.chat([Message(role="user", content="hi")], tools=[])
    assert (reply.prompt_tokens, reply.completion_tokens, reply.cached_tokens) == (0, 0, 0)


async def test_tool_result_prefers_persisted_msg_name_after_provider_switch(client):
    """Switching from OpenAI/Groq to Gemini mid-session leaves `call_xxx` ids without `::`;
    `Message.name` is already persisted and must win over splitting `tool_call_id`."""
    c, models = client
    await c.chat(
        [
            Message(
                role="tool",
                content='{"ok": true}',
                tool_call_id="call_abc123",
                name="get_timeseries",
            )
        ],
        tools=[],
    )
    part = models.calls[-1]["contents"][0].parts[0]
    assert part.function_response.name == "get_timeseries"


async def test_parallel_tool_results_are_grouped_in_one_user_content(client):
    c, models = client
    await c.chat(
        [
            Message(
                role="tool", content='{"a": 1}', tool_call_id="t1::1", name="search_parameters"
            ),
            Message(role="tool", content='{"b": 2}', tool_call_id="t2::2", name="list_missions"),
        ],
        tools=[],
    )
    contents = models.calls[-1]["contents"]
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert [p.function_response.name for p in contents[0].parts] == [
        "search_parameters",
        "list_missions",
    ]


async def test_thought_parts_are_excluded_from_visible_reply(client):
    c, models = client
    models.response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(function_call=None, text="secret chain", thought=True),
                        SimpleNamespace(function_call=None, text="visible answer", thought=False),
                    ]
                )
            )
        ]
    )
    reply = await c.chat([Message(role="user", content="hi")], tools=[])
    assert reply.content == "visible answer"


async def test_stream_chat_yields_visible_deltas_and_final_message(client):
    c, models = client

    async def _gen_stream(**kwargs):
        models.calls.append(kwargs)

        async def _iter():
            yield SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(function_call=None, text="thinking", thought=True),
                                SimpleNamespace(function_call=None, text="Hello ", thought=False),
                            ]
                        )
                    )
                ],
                usage_metadata=None,
            )
            yield SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[SimpleNamespace(function_call=None, text="world", thought=False)]
                        )
                    )
                ],
                usage_metadata=SimpleNamespace(
                    prompt_token_count=100,
                    candidates_token_count=20,
                    cached_content_token_count=10,
                ),
            )

        return _iter()

    models.generate_content_stream = _gen_stream
    items = [item async for item in c.stream_chat([Message(role="user", content="hi")], tools=[])]
    assert items[:-1] == ["Hello ", "world"]
    final = items[-1]
    assert isinstance(final, Message)
    assert final.content == "Hello world"
    assert (final.prompt_tokens, final.completion_tokens, final.cached_tokens) == (100, 20, 10)


async def test_thought_signature_round_trips_through_chat(client):
    """Gemini 2.5+/3+ thinking models attach `thought_signature` to function_call parts
    and return HTTP 400 on the next tool turn if it is not echoed back."""
    c, models = client
    sig = b"\xde\xad\xbe\xef-thought-sig"
    models.response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            function_call=SimpleNamespace(name="get_timeseries", args={"id": "b"}),
                            text=None,
                            thought_signature=sig,
                        )
                    ]
                )
            )
        ]
    )
    reply = await c.chat([Message(role="user", content="hi")], tools=[])
    assert reply.tool_calls is not None
    assert reply.tool_calls[0].thought_signature == sig

    await c.chat(
        [
            Message(role="user", content="hi"),
            reply,
            Message(
                role="tool",
                content='{"ok": true}',
                tool_call_id=reply.tool_calls[0].id,
                name="get_timeseries",
            ),
        ],
        tools=[],
    )
    model_turn = models.calls[-1]["contents"][1]
    assert model_turn.role == "model"
    assert model_turn.parts[0].thought_signature == sig


async def test_thoughts_token_count_is_included_in_completion_tokens(client):
    """Thinking tokens are billed as output tokens and must not be dropped from usage."""
    c, models = client
    models.response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(function_call=None, text="ok")])
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=500,
            candidates_token_count=40,
            thoughts_token_count=310,
            cached_content_token_count=100,
        ),
    )
    reply = await c.chat([Message(role="user", content="hi")], tools=[])
    assert (reply.prompt_tokens, reply.completion_tokens, reply.cached_tokens) == (500, 350, 100)


async def test_stream_chat_preserves_thought_signature_and_retries_first_chunk(client, monkeypatch):
    import helioai.core.llm.base as base_mod

    c, models = client
    sig = b"stream-sig-123"
    attempts = 0

    class _RetryableError(Exception):
        status_code = 503

    async def _no_sleep(_):
        return None

    monkeypatch.setattr(base_mod.asyncio, "sleep", _no_sleep)

    async def _gen_stream(**kwargs):
        nonlocal attempts
        attempts += 1
        models.calls.append(kwargs)

        async def _iter():
            if attempts == 1:
                raise _RetryableError("transient 503 on first chunk")
            yield SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(
                                    function_call=SimpleNamespace(name="search", args={"q": "Bz"}),
                                    text=None,
                                    thought_signature=sig,
                                )
                            ]
                        )
                    )
                ],
                usage_metadata=SimpleNamespace(
                    prompt_token_count=80,
                    candidates_token_count=12,
                    thoughts_token_count=88,
                    cached_content_token_count=0,
                ),
            )

        return _iter()

    models.generate_content_stream = _gen_stream
    items = [item async for item in c.stream_chat([Message(role="user", content="hi")], tools=[])]
    assert attempts == 2
    assert len(items) == 1
    final = items[0]
    assert isinstance(final, Message)
    assert final.tool_calls is not None
    assert final.tool_calls[0].thought_signature == sig
    assert final.completion_tokens == 100
