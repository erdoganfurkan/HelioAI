"""AzureOpenAIClient: Azure-hosted OpenAI → neutral Message model.

Wire format identique à Groq/OpenAI. Différences Azure :
  - endpoint = {azure_endpoint}/openai/deployments/{deployment}/...
  - `model` dans la requête = nom du deployment, pas le nom du modèle
  - temperature=None → kwarg omis (requis pour GPT-5 et o-series)
"""

from __future__ import annotations

import json
import logging

from openai import AsyncAzureOpenAI

from .base import LLMClient, Message, ToolCall, ToolDef

log = logging.getLogger(__name__)


class AzureOpenAIClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        api_version: str,
        deployment: str,
        max_output_tokens: int = 4096,
        temperature: float | None = None,
    ):
        self._client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self._deployment = deployment
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system_prompt: str | None = None,
        tool_choice: str = "auto",
    ) -> Message:
        openai_messages: list[dict] = []
        if system_prompt:
            openai_messages.append({"role": "developer", "content": system_prompt})
        openai_messages.extend(self._to_openai_messages(messages))
        openai_tools = self._to_openai_tools(tools) if tools else None

        kwargs: dict = {
            "model": self._deployment,
            "messages": openai_messages,
            "max_tokens": self._max_output_tokens,
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = tool_choice

        response = await self._client.chat.completions.create(**kwargs)
        return self._from_openai_response(response)

    @staticmethod
    def _to_openai_messages(messages: list[Message]) -> list[dict]:
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

    @staticmethod
    def _to_openai_tools(tools: list[ToolDef]) -> list[dict]:
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

    @staticmethod
    def _from_openai_response(response) -> Message:
        choice = response.choices[0]
        msg = choice.message
        tool_calls_raw = getattr(msg, "tool_calls", None) or []
        if tool_calls_raw:
            tool_calls: list[ToolCall] = []
            for tc in tool_calls_raw:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    log.warning(
                        "azure tool_call %s bad JSON args: %r",
                        tc.function.name,
                        tc.function.arguments,
                    )
                    args = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
            return Message(role="assistant", tool_calls=tool_calls)
        return Message(role="assistant", content=msg.content or "")
