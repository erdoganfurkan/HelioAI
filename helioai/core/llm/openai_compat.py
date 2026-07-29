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

from openai import AsyncOpenAI

from .base import LLMClient, Message, ToolCall, ToolDef, call_with_retry

log = logging.getLogger(__name__)


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


def from_openai_response(response, provider: str = "openai") -> Message:
    """Convert an OpenAI chat-completions response to a neutral message.

    Text content is preserved even when tool calls are present, and a tool call
    whose arguments are not valid JSON degrades to `{}` with a warning rather
    than raising — a malformed model output must not kill the agent loop.

    Args:
        response: The SDK response object.
        provider: Provider name, used only to label log messages.

    Returns:
        The assistant's reply, with `tool_calls` set when the model requested any.
    """
    msg = response.choices[0].message
    tool_calls_raw = getattr(msg, "tool_calls", None) or []
    if not tool_calls_raw:
        return Message(role="assistant", content=msg.content or "")

    tool_calls: list[ToolCall] = []
    for tc in tool_calls_raw:
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            log.warning(
                "%s tool_call %s bad JSON args: %r",
                provider,
                tc.function.name,
                tc.function.arguments,
            )
            args = {}
        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
    return Message(role="assistant", content=msg.content or "", tool_calls=tool_calls)


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
        client=None,
    ):
        self._client = client or AsyncOpenAI(
            api_key=api_key or "unused", base_url=base_url, max_retries=0
        )
        self._model = model
        self._system_role = system_role
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._provider = provider

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

        response = await call_with_retry(lambda: self._client.chat.completions.create(**kwargs))
        return from_openai_response(response, self._provider)
