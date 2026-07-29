"""Azure-hosted OpenAI.

Same wire format as every other OpenAI-compatible provider, so the conversion
logic lives in `openai_compat`. Azure only differs in how the client is built —
the endpoint embeds the deployment name and an `api-version` — plus two dialect
details the base class already exposes as parameters: the system prompt is sent
with the `developer` role, and `temperature` is omitted when unset because GPT-5
and the o-series reject it.
"""

from __future__ import annotations

from openai import AsyncAzureOpenAI

from .openai_compat import OpenAICompatClient


class AzureOpenAIClient(OpenAICompatClient):
    """Chat client for an Azure OpenAI deployment.

    Args:
        api_key: Azure OpenAI API key.
        endpoint: Resource root, e.g. `https://myresource.openai.azure.com`.
        api_version: Azure API version, e.g. `2024-12-01-preview`.
        deployment: Deployment name, sent as the request's `model`.
        max_output_tokens: Cap on generated tokens.
        temperature: Sampling temperature; `None` omits the field.
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        api_version: str,
        deployment: str,
        max_output_tokens: int = 4096,
        temperature: float | None = None,
    ):
        super().__init__(
            client=AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version,
                max_retries=0,
            ),
            model=deployment,
            system_role="developer",
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            provider="azure",
        )
