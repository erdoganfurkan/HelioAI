"""Tests for the vision figure-review side-call."""

from __future__ import annotations

import base64
import io
import json

import pytest

from helioai.config import settings
from helioai.core import vision


@pytest.fixture
def vision_on(monkeypatch):
    monkeypatch.setattr(settings.vision, "enabled", True)
    monkeypatch.setattr(settings.vision, "provider", "azure")
    monkeypatch.setattr(settings.llm.azure, "api_key", "k")
    monkeypatch.setattr(settings.llm.azure, "endpoint", "https://x")
    monkeypatch.setattr(vision, "_warned_no_creds", False)


def _fig(tmp_path, size=(100, 80)):
    from PIL import Image

    p = tmp_path / "fig_0_0.png"
    Image.new("RGB", size, (10, 20, 30)).save(p)
    return str(p)


def _result(fig_path: str) -> str:
    return json.dumps({"figure_paths": [fig_path], "exports": {}, "stdout": ""})


async def test_disabled_is_passthrough(monkeypatch, tmp_path):
    monkeypatch.setattr(settings.vision, "enabled", False)
    called = []

    async def boom(*a):
        called.append(1)
        return "x"

    monkeypatch.setattr(vision, "_call_vision", boom)
    result = _result(_fig(tmp_path))
    out, verdict = await vision.maybe_review("run_python", result)
    assert out == result and verdict is None and not called


async def test_verdict_attached(vision_on, monkeypatch, tmp_path):
    async def fake(images, prompt):
        assert len(images) == 1
        return "OK — labels and data visible"

    monkeypatch.setattr(vision, "_call_vision", fake)
    out, verdict = await vision.maybe_review("run_python", _result(_fig(tmp_path)))
    assert verdict == "OK — labels and data visible"
    assert json.loads(out)["figure_review"] == verdict


async def test_other_tools_ignored(vision_on, monkeypatch, tmp_path):
    async def fake(images, prompt):
        raise AssertionError("must not be called")

    monkeypatch.setattr(vision, "_call_vision", fake)
    result = _result(_fig(tmp_path))
    out, verdict = await vision.maybe_review("get_catalog", result)
    assert out == result and verdict is None


async def test_error_and_figureless_results_ignored(vision_on, monkeypatch):
    async def fake(images, prompt):
        raise AssertionError("must not be called")

    monkeypatch.setattr(vision, "_call_vision", fake)
    for result in (
        json.dumps({"error": "boom", "figure_paths": ["x.png"]}),
        json.dumps({"exports": {}, "figure_paths": []}),
        "not-json",
    ):
        out, verdict = await vision.maybe_review("run_python", result)
        assert out == result and verdict is None


async def test_exception_never_blocks(vision_on, monkeypatch, tmp_path):
    async def fake(images, prompt):
        raise RuntimeError("proxy down")

    monkeypatch.setattr(vision, "_call_vision", fake)
    result = _result(_fig(tmp_path))
    out, verdict = await vision.maybe_review("run_python", result)
    assert out == result and verdict is None


async def test_missing_creds_disables(monkeypatch, tmp_path):
    monkeypatch.setattr(settings.vision, "enabled", True)
    monkeypatch.setattr(settings.vision, "provider", "azure")
    monkeypatch.setattr(settings.llm.azure, "api_key", "")
    monkeypatch.setattr(vision, "_warned_no_creds", False)
    result = _result(_fig(tmp_path))
    out, verdict = await vision.maybe_review("run_python", result)
    assert out == result and verdict is None


def test_png_downscaled(tmp_path):
    from PIL import Image

    big = _fig(tmp_path, size=(2000, 1200))
    data = base64.b64decode(vision._png_b64(big))
    im = Image.open(io.BytesIO(data))
    assert max(im.size) <= 768


async def test_loop_emits_figure_review_event(vision_on, monkeypatch, tmp_path):
    from helioai.core import agent_loop
    from helioai.core.llm.base import Message, ToolCall
    from helioai.core.session import SessionStore

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    store = SessionStore(tmp_path / "sessions.db")
    monkeypatch.setattr(agent_loop, "store", store)

    fig = _fig(tmp_path)

    class _StubRegistry:
        def list_tool_defs(self, only=None):
            return []

        def __contains__(self, name):
            return name == "run_python"

        async def call_tool(self, name, args, trusted=None):
            return _result(fig)

    monkeypatch.setattr(agent_loop, "registry", _StubRegistry())

    async def fake(images, prompt):
        return "ISSUE: y-axis has no label"

    monkeypatch.setattr(vision, "_call_vision", fake)

    class _FakeLLM:
        async def chat(self, messages, tools, **k):
            if any(m.role == "tool" for m in messages):
                return Message(role="assistant", content="done")
            return Message(
                role="assistant",
                tool_calls=[ToolCall(id="t1", name="run_python", arguments={"code": "plot"})],
            )

    events = [
        ev async for ev in agent_loop.stream_chat(_FakeLLM(), "web", "s1", "hi", restricted=False)
    ]
    reviews = [e for e in events if e["event"] == "figure_review"]
    assert len(reviews) == 1
    assert "y-axis" in reviews[0]["data"]["text"]

    history = store.get_or_create("web", "s1")
    tool_msgs = [m for m in history if m.role == "tool"]
    assert any("figure_review" in (m.content or "") for m in tool_msgs)
