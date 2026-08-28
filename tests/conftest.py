"""Shared pytest fixtures for HelioAI tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from helioai.core.llm.base import LLMClient, Message
from helioai.tools.sandbox import _sandbox_env

_SANDBOX_WARMUP = """\
import os; os.environ['MPLBACKEND'] = 'Agg'
import numpy, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot
import scipy, scipy.signal, scipy.stats, scipy.fft
import astropy, astropy.units
try:
    import plasmapy, plasmapy.formulary
except Exception:
    pass
"""


@pytest.fixture(scope="session", autouse=True)
def _warm_sandbox_imports():
    """Pre-compile .pyc for all sandbox heavy imports.

    Runs once per test session as a subprocess so that subsequent sandbox
    subprocess calls find pre-compiled bytecode instead of compiling from scratch.
    Without this, cold-start CI runners time out on trivial sandbox tests.

    Best-effort only: a slow/hanging warmup must not fail the whole test session
    over a perf optimization — fall through and let tests cold-start instead.

    speasy is deliberately NOT warmed here. Its import builds the provider
    inventory, which goes to the network whenever the sandbox HOME is cold —
    and while the SciQLop inventory cache is down (HTTP 502) that costs the full
    timeout on EVERY pytest invocation, whatever the selection: 180 s of setup
    for a suite whose tests are all under a second. The rest of the stack warms
    in 2 s. The sandbox imports speasy lazily anyway, so nothing else pays for it.

    Runs under _sandbox_env() so the caches land in the exact HOME (/tmp) that
    sandbox subprocesses will read — warming the runner's real HOME leaves the
    sandbox cold.
    """
    try:
        subprocess.run(
            [sys.executable, "-c", _SANDBOX_WARMUP],
            timeout=60,
            capture_output=True,
            env=_sandbox_env(),
        )
    except subprocess.TimeoutExpired:
        pass


@pytest.fixture(autouse=True)
def _reset_workspace_ctx():
    """Keep the workspace contextvars (user/label/session) from leaking across tests."""
    import helioai.workspace as ws

    yield
    ws._current_user.set(None)
    ws._current_label.set(None)
    ws._current_session.set(None)


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
        self.calls.append(
            {
                "messages": list(messages),
                "tools": list(tools),
                "system_prompt": system_prompt,
            }
        )
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
