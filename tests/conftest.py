"""Shared pytest fixtures for HelioAI tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from helioai.core.llm.base import LLMClient, Message, ToolDef


@pytest.fixture
def tmp_chroma_dir(tmp_path: Path) -> Path:
    return tmp_path / "chroma"


@pytest.fixture
def sample_skill_dir(tmp_path: Path) -> Path:
    d = tmp_path / "skills"
    d.mkdir()
    return d


class _FakeLLM(LLMClient):
    """Replays a scripted list of Messages, one per chat() call."""

    def __init__(self, responses: list[Message]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def chat(self, messages, tools, system_prompt=None):
        self.calls.append({
            "messages": list(messages),
            "tools": list(tools),
            "system_prompt": system_prompt,
        })
        return self._responses.pop(0)


@pytest.fixture
def fake_llm_factory():
    return _FakeLLM


class _FakeEmbedModel:
    """Returns deterministic unit-norm embeddings for any text."""

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, **kwargs):
        rng = np.random.default_rng(42)
        vecs = rng.random((len(texts), 128)).astype("float32")
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-9)
        return vecs


@pytest.fixture
def fake_embed_model():
    return _FakeEmbedModel()
