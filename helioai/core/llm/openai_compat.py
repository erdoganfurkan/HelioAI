"""Single client for every provider that speaks the OpenAI chat-completions wire format.

Groq, Ollama, Azure OpenAI and OpenAI itself all accept the same request shape, so
they share one implementation here instead of one near-identical class each. A
provider is a `base_url` plus a couple of dialect flags, not a subclass.

Azure is the one exception that still needs its own SDK client object (deployment
routing and `api-version`), so it subclasses this to swap the constructor only.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI, BadRequestError

from .base import LLMClient, Message, ToolCall, ToolDef, call_with_retry, close_sdk_client

log = logging.getLogger(__name__)

# Some OpenAI-compatible reasoning models (MiniMax, DeepSeek-R1-style serving...) have
# no separate "reasoning" field on this wire format — they inline their chain of thought
# into `content` itself, wrapped in <think>. Left in place, that is what the agent loop
# treats as the reply: printed to the user, and re-sent as conversation history on every
# following turn. `\Z` also matches an unclosed tag — a reasoning-heavy generation that
# exhausted `max_output_tokens` before ever closing it, which must still strip to nothing
# rather than leak raw reasoning, and an empty reply is already the case the loop's
# "output token budget" error message exists for.
_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)


def _strip_reasoning(content: str) -> str:
    """Remove inline <think>...</think> reasoning from a reply's content."""
    if "<think>" not in content:
        return content
    return _THINK_BLOCK.sub("", content).strip()


def to_openai_messages(messages: list[Message]) -> list[dict]:
    """Convert neutral messages to the OpenAI wire format.

    System messages already in the history are dropped: the system prompt is
    passed separately by `chat()` so it always lands first.

    Args:
        messages: Conversation history in HelioAI's provider-neutral form.

    Returns:
        Message dicts ready to send as the `messages` request field.
    """
    out: list[dict] = []
    for msg in messages:
        if msg.role == "system":
            continue
        if msg.role == "user":
            out.append({"role": "user", "content": msg.content})
        elif msg.role == "assistant":
            if msg.tool_calls:
                out.append(
                    {
                        "role": "assistant",
                        "content": msg.content or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments or {}),
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
            else:
                out.append({"role": "assistant", "content": msg.content})
        elif msg.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id or "",
                    "content": msg.content,
                }
            )
    return out


def to_openai_tools(tools: list[ToolDef]) -> list[dict]:
    """Convert tool definitions to OpenAI function-calling schemas.

    Args:
        tools: Tools the agent may call this turn.

    Returns:
        Function schemas ready to send as the `tools` request field.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def _usage(response: Any) -> dict[str, int]:
    """Token counts as the provider reported them, zeroed when it reported none.

    Every OpenAI-compatible endpoint fills `usage`, but not all of them fill
    `prompt_tokens_details`, so the cached count is read defensively rather than
    assumed. These were thrown away until the benchmark had to wrap the SDK client
    to recover what the response already carried.
    """
    u = getattr(response, "usage", None)
    details = getattr(u, "prompt_tokens_details", None) if u else None
    return {
        "prompt_tokens": (getattr(u, "prompt_tokens", 0) or 0) if u else 0,
        "completion_tokens": (getattr(u, "completion_tokens", 0) or 0) if u else 0,
        "cached_tokens": (getattr(details, "cached_tokens", 0) or 0) if details else 0,
    }


def from_openai_response(response: Any, provider: str = "openai") -> Message:
    """Convert an OpenAI chat-completions response to a neutral message.

    Text content is preserved even when tool calls are present, and a tool call
    whose arguments are not valid JSON degrades to `{}` with a warning rather
    than raising — a malformed model output must not kill the agent loop. An
    inline `<think>...</think>` reasoning block, when a provider emits one, is
    stripped from the content before it reaches the agent loop or the user.

    Args:
        response: The SDK response object.
        provider: Provider name, used only to label log messages.

    Returns:
        The assistant's reply, with `tool_calls` set when the model requested any.
    """
    choice = response.choices[0]
    msg = choice.message
    usage = _usage(response)
    raw_content = msg.content or ""
    content = _strip_reasoning(raw_content)
    tool_calls_raw = getattr(msg, "tool_calls", None) or []
    finish_reason = getattr(choice, "finish_reason", None)

    if not tool_calls_raw:
        if not content.strip():
            # A turn that produced nothing has exactly three causes and they need
            # different fixes: the budget ran out mid-generation (raise it), the model
            # spent the whole turn inside <think> and closed with nothing (shorten the
            # question), or it genuinely returned an empty completion (retry/provider).
            # The agent loop used to state the first as fact for all three. It is
            # visible here and nowhere else, so it is recorded here.
            log.warning(
                "%s empty completion: finish_reason=%s, %d raw chars, %d after stripping reasoning",
                provider,
                finish_reason,
                len(raw_content),
                len(content),
            )
        return Message(role="assistant", content=content, **usage)

    tool_calls = [
        _parse_tool_call(tc.id, tc.function.name, tc.function.arguments, finish_reason, provider)
        for tc in tool_calls_raw
    ]
    return Message(role="assistant", content=content, tool_calls=tool_calls, **usage)


def _parse_tool_call(
    call_id: str, name: str, arguments: str | None, finish_reason: str | None, provider: str
) -> ToolCall:
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as e:
        # `finish_reason="length"` next to unparseable arguments is not "the model
        # emits bad JSON" — it is OUR output budget slicing a valid call mid-string.
        # The two need different fixes (raise max_output_tokens vs distrust the
        # model), and a log line that cannot tell them apart cost an hour of
        # diagnosis on a run where six 12k-char run_python calls all "lost" their code.
        log.warning(
            "%s tool_call %s args unparseable (finish_reason=%s, %d chars, %s): %r",
            provider,
            name,
            finish_reason,
            len(arguments or ""),
            "output budget truncated the call" if finish_reason == "length" else e,
            (arguments or "")[:200],
        )
        args = {}
    return ToolCall(id=call_id, name=name, arguments=args)


def _visible_so_far(raw: str) -> str:
    """The part of a streamed reply a reader may see: everything but an inline reasoning
    block, including one that has opened and not closed yet."""
    if "<think>" in raw and "</think>" not in raw:
        return raw[: raw.index("<think>")]
    return _strip_reasoning(raw)


class OpenAICompatClient(LLMClient):
    """Chat client for any OpenAI-compatible endpoint.

    Args:
        model: Model name sent in the request (the deployment name on Azure).
        api_key: Provider API key. Local endpoints such as Ollama ignore it, but
            the SDK requires a non-empty value.
        base_url: Endpoint root. `None` targets OpenAI itself.
        system_role: Role used for the system prompt — `developer` on Azure and
            the o-series, `system` everywhere else.
        max_output_tokens: Cap on generated tokens.
        temperature: Sampling temperature. `None` omits the field entirely, which
            reasoning models require.
        provider: Name used to label log messages.
        client: Pre-built SDK client. Injected by tests and by subclasses.

    Example:
        >>> client = OpenAICompatClient(
        ...     model="llama-3.3-70b-versatile",
        ...     api_key="gsk_...",
        ...     base_url="https://api.groq.com/openai/v1",
        ... )
        >>> reply = await client.chat([Message(role="user", content="hi")], tools=[])
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str = "",
        base_url: str | None = None,
        system_role: str = "system",
        max_output_tokens: int = 4096,
        temperature: float | None = 0.2,
        provider: str = "openai",
        client: Any = None,
        default_headers: dict[str, str] | None = None,
    ):
        self._client = client or AsyncOpenAI(
            api_key=api_key or "unused",
            base_url=base_url,
            max_retries=0,
            default_headers=default_headers or None,
        )
        self._model = model
        self._system_role = system_role
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._provider = provider

    async def aclose(self) -> None:
        """Close the httpx pool held by the SDK client."""
        await close_sdk_client(self._client)

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system_prompt: str | None = None,
        tool_choice: str = "auto",
    ) -> Message:
        """Send one chat turn and return the assistant's reply.

        Args:
            messages: Conversation history.
            tools: Tools the model may call.
            system_prompt: Instructions prepended as the first message.
            tool_choice: `auto` to let the model decide, `required` to force a call.

        Returns:
            The assistant reply, carrying `tool_calls` when the model requested any.
        """
        kwargs = self._request(messages, tools, system_prompt, tool_choice)
        response = await self._create(kwargs, tool_choice)
        reply = from_openai_response(response, self._provider)
        if not (reply.content or "").strip() and not reply.tool_calls:
            # A turn with neither text nor a tool call is not an answer, and on a
            # reasoning model it is not rare either: the whole output allowance can go
            # into hidden reasoning and leave nothing to emit. It is also transient —
            # the identical request, replayed, came back with two tool calls in half
            # the wall time. The loop above treats this as fatal and abandons the
            # question, so two acts of a six-act notebook were lost to a condition that
            # one more attempt clears. Retried once, not in a loop: if the second is
            # empty too, the caller's error is the honest outcome.
            log.warning("%s empty turn, retrying once: %s", self._provider, self._model)
            response = await call_with_retry(lambda: self._client.chat.completions.create(**kwargs))
            reply = from_openai_response(response, self._provider)
        return reply

    def _request(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system_prompt: str | None,
        tool_choice: str,
    ) -> dict:
        openai_messages: list[dict] = []
        if system_prompt:
            openai_messages.append({"role": self._system_role, "content": system_prompt})
        openai_messages.extend(to_openai_messages(messages))

        kwargs: dict = {
            "model": self._model,
            "messages": openai_messages,
            "max_tokens": self._max_output_tokens,
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
            kwargs["tool_choice"] = tool_choice
        return kwargs

    async def _create(self, kwargs: dict, tool_choice: str) -> Any:
        try:
            return await call_with_retry(lambda: self._client.chat.completions.create(**kwargs))
        except BadRequestError as e:
            # Forcing a tool call is a preference, never worth losing the turn over.
            # DeepSeek v4 in thinking mode rejects `required` outright ("Thinking mode
            # does not support this tool_choice"), which killed a sub-agent on its very
            # first turn. Whether a model accepts it depends on the model and its
            # reasoning mode, not on the provider, so it is asked rather than tabulated.
            if tool_choice == "auto" or "tool_choice" not in str(e):
                raise
            log.warning("tool_choice_rejected_falling_back_to_auto: %s", self._model)
            kwargs["tool_choice"] = "auto"
            return await call_with_retry(lambda: self._client.chat.completions.create(**kwargs))

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system_prompt: str | None = None,
        tool_choice: str = "auto",
    ) -> AsyncIterator[str | Message]:
        """Send one chat turn, yielding the reply's text as it arrives, then the reply.

        Text deltas are yielded as the provider sends them, with an inline
        `<think>` block held back until it closes. Tool-call fragments are
        reassembled by index and parsed exactly as `chat()` parses a finished call;
        the usage comes from the final chunk (`stream_options.include_usage`). An
        endpoint that rejects the streaming request falls back to `chat()`, and a
        streamed turn that ends with neither text nor a tool call is retried once
        through `chat()`, as `chat()` retries its own.

        Args:
            messages: Conversation history.
            tools: Tools the model may call.
            system_prompt: Instructions prepended as the first message.
            tool_choice: `auto` to let the model decide, `required` to force a call.

        Yields:
            Text deltas, then the final `Message`.
        """
        kwargs = self._request(messages, tools, system_prompt, tool_choice)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        try:
            stream = await self._create(kwargs, tool_choice)
        except BadRequestError as e:
            if "stream" not in str(e).lower():
                raise
            log.warning("%s rejected streaming, falling back to chat(): %s", self._provider, e)
            yield await self.chat(messages, tools, system_prompt, tool_choice)
            return

        raw = ""
        sent = 0
        calls: dict[int, dict] = {}
        usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
        finish_reason = None
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = _usage(chunk)
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            if getattr(delta, "content", None):
                raw += delta.content
                visible = _visible_so_far(raw)
                if len(visible) > sent:
                    yield visible[sent:]
                    sent = len(visible)
            for frag in getattr(delta, "tool_calls", None) or []:
                slot = calls.setdefault(
                    getattr(frag, "index", 0) or 0, {"id": None, "name": None, "arguments": ""}
                )
                if getattr(frag, "id", None):
                    slot["id"] = frag.id
                fn = getattr(frag, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["arguments"] += fn.arguments

        content = _strip_reasoning(raw)
        tool_calls = [
            _parse_tool_call(
                c["id"] or f"call_{i}",
                c["name"] or "",
                c["arguments"],
                finish_reason,
                self._provider,
            )
            for i, c in sorted(calls.items())
        ]
        if not content.strip() and not tool_calls:
            log.warning("%s empty streamed turn, retrying once: %s", self._provider, self._model)
            yield await self.chat(messages, tools, system_prompt, tool_choice)
            return
        yield Message(role="assistant", content=content, tool_calls=tool_calls or None, **usage)
