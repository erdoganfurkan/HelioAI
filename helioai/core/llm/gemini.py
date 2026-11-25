"""GeminiClient: Google genai SDK → neutral Message model."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator

from google import genai
from google.genai import types as gt

from .base import LLMClient, Message, ToolCall, ToolDef, call_with_retry, close_sdk_client

log = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    """Chat client for Google Gemini, using the native `google-genai` SDK.

    Kept separate from `OpenAICompatClient` because the wire format genuinely
    differs: turns are `Content` objects with typed parts, the assistant role is
    called `model`, and tool results are matched by function *name* rather than
    by id. Since Gemini issues no call ids, this client synthesises `name::hex`
    and parses the name back out on the way in — a format that is persisted in
    existing sessions, so it must stay readable.

    Args:
        api_key: Gemini API key.
        model: Model name, e.g. `gemini-2.5-flash`.
        max_output_tokens: Cap on generated tokens.
        temperature: Sampling temperature.
    """

    def __init__(
        self, api_key: str, model: str, max_output_tokens: int = 4096, temperature: float = 0.2
    ):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature

    async def aclose(self) -> None:
        """Close the transport held by the genai client (its close() is sync)."""
        await close_sdk_client(self._client)

    def _build_request(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system_prompt: str | None,
        tool_choice: str,
    ) -> tuple[list[gt.Content], gt.GenerateContentConfig]:
        contents = self._to_gemini_contents(messages)
        gemini_tools = self._to_gemini_tools(tools) if tools else None

        if system_prompt is None:
            system_prompt = next((m.content for m in messages if m.role == "system"), None)

        tool_config = None
        if gemini_tools and tool_choice == "required":
            tool_config = gt.ToolConfig(
                function_calling_config=gt.FunctionCallingConfig(mode="ANY")
            )

        config = gt.GenerateContentConfig(
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
            tools=gemini_tools,
            tool_config=tool_config,
            system_instruction=system_prompt,
        )
        return contents, config

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system_prompt: str | None = None,
        tool_choice: str = "auto",
    ) -> Message:
        """Send one turn to Gemini and return the assistant's reply.

        When `system_prompt` is not given, it is recovered from any system message
        left in the history. `tool_choice="required"` maps to Gemini's ANY mode.
        """
        contents, config = self._build_request(messages, tools, system_prompt, tool_choice)
        response = await call_with_retry(
            lambda: self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        )
        return self._from_gemini_response(response)

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system_prompt: str | None = None,
        tool_choice: str = "auto",
    ) -> AsyncIterator[str | Message]:
        """Stream one Gemini turn, yielding visible text deltas then the final Message.

        Thought parts (`part.thought == True` on Gemini 2.5+/3+ reasoning models) are
        kept out of both the streamed deltas and the returned content, while any
        `thought_signature` on `function_call` parts is preserved on `ToolCall` for
        echoing back on the next turn. Retries cover opening the stream and reading
        its first chunk (before any delta is yielded); mid-stream failures propagate
        directly to avoid duplicating already-streamed text.
        """
        contents, config = self._build_request(messages, tools, system_prompt, tool_choice)

        async def _open_and_peek():
            stream = await self._client.aio.models.generate_content_stream(
                model=self._model, contents=contents, config=config
            )
            it = stream.__aiter__()
            try:
                first = await it.__anext__()
            except StopAsyncIteration:
                return None, it
            return first, it

        first_chunk, rest = await call_with_retry(_open_and_peek)
        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}

        async def _all_chunks():
            if first_chunk is not None:
                yield first_chunk
            async for chunk in rest:
                yield chunk

        async for chunk in _all_chunks():
            if getattr(chunk, "usage_metadata", None) is not None:
                usage = self._extract_usage(chunk)
            candidate = chunk.candidates[0] if getattr(chunk, "candidates", None) else None
            parts = candidate.content.parts if candidate and candidate.content else []
            for part in parts or []:
                fc = getattr(part, "function_call", None)
                if fc and fc.name:
                    args = dict(fc.args) if fc.args else {}
                    call_id = f"{fc.name}::{uuid.uuid4().hex[:8]}"
                    sig = self._part_signature(part)
                    tool_calls.append(
                        ToolCall(id=call_id, name=fc.name, arguments=args, thought_signature=sig)
                    )
                    continue
                if getattr(part, "thought", False):
                    continue
                text = getattr(part, "text", None)
                if text:
                    text_chunks.append(text)
                    yield text

        yield Message(
            role="assistant",
            content="".join(text_chunks),
            tool_calls=tool_calls or None,
            **usage,
        )

    @staticmethod
    def _tool_name(msg: Message) -> str:
        """Resolve the function name for a tool reply, surviving mid-session provider switches.

        Persisted `msg.name` is authoritative (OpenAI/Groq call ids have no `::` prefix);
        older rows without `name` fall back to the synthesised `name::hex` id.
        """
        if msg.name:
            return msg.name
        name, _, _ = (msg.tool_call_id or "::").partition("::")
        return name

    @staticmethod
    def _to_gemini_contents(messages: list[Message]) -> list[gt.Content]:
        contents: list[gt.Content] = []
        pending_tool_parts: list[gt.Part] = []

        def _flush_tool_parts() -> None:
            if pending_tool_parts:
                contents.append(gt.Content(role="user", parts=list(pending_tool_parts)))
                pending_tool_parts.clear()

        for msg in messages:
            if msg.role == "system":
                continue
            if msg.role == "tool":
                name = GeminiClient._tool_name(msg)
                try:
                    response_payload = json.loads(msg.content)
                    if not isinstance(response_payload, dict):
                        response_payload = {"result": response_payload}
                except json.JSONDecodeError:
                    response_payload = {"result": msg.content}
                pending_tool_parts.append(
                    gt.Part(
                        function_response=gt.FunctionResponse(
                            name=name,
                            response=response_payload,
                        )
                    )
                )
                continue

            _flush_tool_parts()
            if msg.role == "user":
                contents.append(gt.Content(role="user", parts=[gt.Part(text=msg.content)]))
            elif msg.role == "assistant":
                if msg.tool_calls:
                    parts: list[gt.Part] = []
                    if msg.content:
                        parts.append(gt.Part(text=msg.content))
                    for tc in msg.tool_calls:
                        fc_kwargs: dict = {
                            "function_call": gt.FunctionCall(name=tc.name, args=tc.arguments)
                        }
                        if tc.thought_signature is not None:
                            fc_kwargs["thought_signature"] = tc.thought_signature
                        parts.append(gt.Part(**fc_kwargs))
                    contents.append(gt.Content(role="model", parts=parts))
                else:
                    contents.append(gt.Content(role="model", parts=[gt.Part(text=msg.content)]))

        _flush_tool_parts()
        return contents

    @staticmethod
    def _to_gemini_tools(tools: list[ToolDef]) -> list[gt.Tool]:
        declarations = [
            gt.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters=t.parameters or None,
            )
            for t in tools
        ]
        return [gt.Tool(function_declarations=declarations)]

    @staticmethod
    def _part_signature(part) -> bytes | None:
        sig = getattr(part, "thought_signature", None)
        if isinstance(sig, (bytes, bytearray)) and sig:
            return bytes(sig)
        return None

    @staticmethod
    def _extract_usage(response) -> dict[str, int]:
        """Map Gemini's `usage_metadata` onto provider-neutral token counts.

        Thinking tokens (`thoughts_token_count`) are billed as output tokens on
        Gemini 2.5+/3+ models, so they are folded into `completion_tokens` just as
        OpenAI's reasoning tokens are.
        """
        um = getattr(response, "usage_metadata", None)
        if um is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
        candidates = getattr(um, "candidates_token_count", 0) or 0
        thoughts = getattr(um, "thoughts_token_count", 0) or 0
        return {
            "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
            "completion_tokens": candidates + thoughts,
            "cached_tokens": getattr(um, "cached_content_token_count", 0) or 0,
        }

    @staticmethod
    def _from_gemini_response(response) -> Message:
        tool_calls: list[ToolCall] = []
        text_chunks: list[str] = []
        candidate = response.candidates[0] if response.candidates else None
        parts = candidate.content.parts if candidate and candidate.content else []
        for part in parts or []:
            fc = getattr(part, "function_call", None)
            if fc and fc.name:
                args = dict(fc.args) if fc.args else {}
                call_id = f"{fc.name}::{uuid.uuid4().hex[:8]}"
                sig = GeminiClient._part_signature(part)
                tool_calls.append(
                    ToolCall(id=call_id, name=fc.name, arguments=args, thought_signature=sig)
                )
                continue
            if getattr(part, "thought", False):
                continue
            text = getattr(part, "text", None)
            if text:
                text_chunks.append(text)
        # google-genai names every count differently from the OpenAI wire format,
        # which is the whole reason this client is not the compat one.
        usage = GeminiClient._extract_usage(response)
        if tool_calls:
            return Message(
                role="assistant",
                content="".join(text_chunks),
                tool_calls=tool_calls,
                **usage,
            )
        return Message(role="assistant", content="".join(text_chunks), **usage)
