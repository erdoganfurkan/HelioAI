"""GeminiClient: Google genai SDK → neutral Message model."""

from __future__ import annotations

import json
import logging
import uuid

from google import genai
from google.genai import types as gt

from .base import LLMClient, Message, ToolCall, ToolDef, call_with_retry

log = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    def __init__(
        self, api_key: str, model: str, max_output_tokens: int = 4096, temperature: float = 0.2
    ):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system_prompt: str | None = None,
        tool_choice: str = "auto",
    ) -> Message:
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

        response = await call_with_retry(
            lambda: self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        )
        return self._from_gemini_response(response)

    @staticmethod
    def _to_gemini_contents(messages: list[Message]) -> list[gt.Content]:
        contents: list[gt.Content] = []
        for msg in messages:
            if msg.role == "system":
                continue
            if msg.role == "user":
                contents.append(gt.Content(role="user", parts=[gt.Part(text=msg.content)]))
            elif msg.role == "assistant":
                if msg.tool_calls:
                    parts = [
                        gt.Part(function_call=gt.FunctionCall(name=tc.name, args=tc.arguments))
                        for tc in msg.tool_calls
                    ]
                    contents.append(gt.Content(role="model", parts=parts))
                else:
                    contents.append(gt.Content(role="model", parts=[gt.Part(text=msg.content)]))
            elif msg.role == "tool":
                name, _, _ = (msg.tool_call_id or "::").partition("::")
                try:
                    response_payload = json.loads(msg.content)
                    if not isinstance(response_payload, dict):
                        response_payload = {"result": response_payload}
                except json.JSONDecodeError:
                    response_payload = {"result": msg.content}
                contents.append(
                    gt.Content(
                        role="user",
                        parts=[
                            gt.Part(
                                function_response=gt.FunctionResponse(
                                    name=name,
                                    response=response_payload,
                                )
                            )
                        ],
                    )
                )
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
                tool_calls.append(ToolCall(id=call_id, name=fc.name, arguments=args))
                continue
            text = getattr(part, "text", None)
            if text:
                text_chunks.append(text)
        if tool_calls:
            return Message(role="assistant", tool_calls=tool_calls)
        return Message(role="assistant", content="".join(text_chunks))
