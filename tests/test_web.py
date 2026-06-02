"""Tests for the FastAPI web interface (app.py) and LLM factory."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_stream():
    """Async generator that replays a scripted sequence of SSE events."""

    async def _gen(_llm, _user, _sid, _msg):
        yield {
            "event": "tool_call",
            "data": {"turn": 1, "name": "search_parameters", "arguments": {"query": "solar wind"}},
        }
        yield {
            "event": "tool_result",
            "data": {"turn": 1, "name": "search_parameters", "summary": "3 results"},
        }
        yield {
            "event": "artifact",
            "data": {"kind": "image", "figure_paths": ["/tmp/helioai_test/fig_0.png"]},
        }
        yield {"event": "reply", "data": {"text": "Here is the answer."}}
        yield {"event": "done", "data": {"n_iterations": 2}}

    return _gen


@pytest.fixture
def web_client(monkeypatch, fake_stream, tmp_path):
    """TestClient with stream_chat and build_llm_client monkeypatched."""
    from helioai.core.session import SessionStore

    test_store = SessionStore(tmp_path / "sessions.db")

    monkeypatch.setattr("helioai.interfaces.web.app.stream_chat", fake_stream)
    monkeypatch.setattr("helioai.interfaces.web.app.build_llm_client", lambda provider=None: None)
    monkeypatch.setattr("helioai.interfaces.web.app.store", test_store)

    from helioai.interfaces.web.app import app

    return TestClient(app, raise_server_exceptions=False)


# ── Health ────────────────────────────────────────────────────────────────────


def test_health(web_client):
    r = web_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ── Index ─────────────────────────────────────────────────────────────────────


def test_index_returns_html(web_client):
    r = web_client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert b"HelioAI" in r.content


# ── SSE streaming ─────────────────────────────────────────────────────────────


def _parse_sse(raw: bytes) -> list[dict]:
    events = []
    for line in raw.decode().splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def test_chat_stream_events(web_client):
    with web_client.stream(
        "POST",
        "/chat/stream",
        json={
            "message": "solar wind density",
            "session_id": "test-session-001",
        },
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        raw = resp.read()

    events = _parse_sse(raw)
    event_types = [e["event"] for e in events]

    assert "tool_call" in event_types
    assert "tool_result" in event_types
    assert "artifact" in event_types
    assert "reply" in event_types
    assert "done" in event_types


def test_chat_stream_reply_content(web_client):
    with web_client.stream(
        "POST",
        "/chat/stream",
        json={
            "message": "test query",
            "session_id": "test-session-002",
        },
    ) as resp:
        raw = resp.read()

    events = _parse_sse(raw)
    reply_events = [e for e in events if e["event"] == "reply"]
    assert len(reply_events) == 1
    assert reply_events[0]["data"]["text"] == "Here is the answer."


def test_chat_stream_artifact(web_client):
    with web_client.stream(
        "POST",
        "/chat/stream",
        json={
            "message": "plot something",
            "session_id": "test-session-003",
        },
    ) as resp:
        raw = resp.read()

    events = _parse_sse(raw)
    artifacts = [e for e in events if e["event"] == "artifact"]
    assert len(artifacts) == 1
    assert artifacts[0]["data"]["kind"] == "image"


# ── Sessions API ──────────────────────────────────────────────────────────────


def test_sessions_empty(web_client):
    r = web_client.get("/api/sessions")
    assert r.status_code == 200
    assert r.json() == []


def test_sessions_after_save(monkeypatch, tmp_path):
    """Sessions list reflects saved conversations."""
    from helioai.core.session import SessionStore
    from helioai.core.llm.base import Message

    test_store = SessionStore(tmp_path / "sessions.db")
    history = [Message(role="user", content="solar wind")]
    test_store.save("web", "sess-abc", history)

    async def _noop_gen(_llm, _user, _sid, _msg):
        yield {"event": "reply", "data": {"text": "ok"}}
        yield {"event": "done", "data": {"n_iterations": 1}}

    monkeypatch.setattr("helioai.interfaces.web.app.stream_chat", _noop_gen)
    monkeypatch.setattr("helioai.interfaces.web.app.build_llm_client", lambda provider=None: None)
    monkeypatch.setattr("helioai.interfaces.web.app.store", test_store)

    from helioai.interfaces.web.app import app

    client = TestClient(app)

    r = client.get("/api/sessions")
    assert r.status_code == 200
    sessions = r.json()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "sess-abc"
    assert sessions[0]["first_message"] == "solar wind"


def test_session_messages(monkeypatch, tmp_path):
    import json
    from helioai.core.session import SessionStore
    from helioai.core.llm.base import Message

    test_store = SessionStore(tmp_path / "sessions.db")
    tool_result = json.dumps(
        {"stdout": "ok", "figure_paths": ["/tmp/ws/fig_0_0.png"], "n_figures": 1, "exports": {}}
    )
    history = [
        Message(role="user", content="Plot something."),
        Message(role="assistant", content="", tool_calls=[]),
        Message(role="tool", tool_call_id="t1", content=tool_result),
        Message(role="assistant", content="Here is your plot."),
    ]
    test_store.save("web", "sess-xyz", history)

    monkeypatch.setattr("helioai.interfaces.web.app.stream_chat", lambda *a, **kw: (_ for _ in []))
    monkeypatch.setattr("helioai.interfaces.web.app.build_llm_client", lambda provider=None: None)
    monkeypatch.setattr("helioai.interfaces.web.app.store", test_store)

    from helioai.interfaces.web.app import app

    client = TestClient(app)

    r = client.get("/api/sessions/sess-xyz/messages")
    assert r.status_code == 200
    data = r.json()
    msgs = data["messages"]
    # intermediate assistant (empty content) is skipped, so 2 entries: user + final assistant
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "Here is your plot."
    # figures are attached to the assistant message that follows the tool result
    assert msgs[1].get("figures") == ["/tmp/ws/fig_0_0.png"]


# ── Figure endpoint ───────────────────────────────────────────────────────────


def test_figure_invalid_path(web_client):
    r = web_client.get("/figure?path=/etc/passwd")
    assert r.status_code == 404


def test_figure_path_outside_workspace(web_client):
    r = web_client.get("/figure?path=/home/user/secret/fig.png")
    assert r.status_code == 404


def test_figure_path_traversal(web_client, tmp_path, monkeypatch):
    from helioai.config import settings

    monkeypatch.setattr(settings.workspace, "workspace_dir", tmp_path)
    # path traversal attempt
    r = web_client.get(f"/figure?path={tmp_path}/../etc/passwd")
    assert r.status_code == 404


def test_figure_valid(web_client, tmp_path, monkeypatch):
    from helioai.config import settings

    monkeypatch.setattr(settings.workspace, "workspace_dir", tmp_path)

    fig_dir = tmp_path / "sess123" / "run001"
    fig_dir.mkdir(parents=True)
    fig = fig_dir / "fig_0.png"
    fig.write_bytes(b"\x89PNG\r\n")

    r = web_client.get(f"/figure?path={fig}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"


# ── Code endpoint ────────────────────────────────────────────────────────────


def test_code_outside_workspace(web_client):
    r = web_client.get("/code?path=/etc/passwd")
    assert r.status_code == 404


def test_code_path_traversal(web_client, tmp_path, monkeypatch):
    from helioai.config import settings

    monkeypatch.setattr(settings.workspace, "workspace_dir", tmp_path)
    r = web_client.get(f"/code?path={tmp_path}/../etc/passwd")
    assert r.status_code == 404


def test_code_not_py(web_client, tmp_path, monkeypatch):
    from helioai.config import settings

    monkeypatch.setattr(settings.workspace, "workspace_dir", tmp_path)
    txt_file = tmp_path / "sess" / "data.txt"
    txt_file.parent.mkdir(parents=True)
    txt_file.write_text("not python")
    r = web_client.get(f"/code?path={txt_file}")
    assert r.status_code == 404


def test_code_valid(web_client, tmp_path, monkeypatch):
    from helioai.config import settings

    monkeypatch.setattr(settings.workspace, "workspace_dir", tmp_path)
    code_dir = tmp_path / "sess123"
    code_dir.mkdir(parents=True)
    code_file = code_dir / "code_0.py"
    source = "import numpy as np\nprint(np.pi)\n"
    code_file.write_text(source)

    r = web_client.get(f"/code?path={code_file}")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    assert r.text == source


def test_session_messages_attach_code(monkeypatch, tmp_path):
    import json
    from helioai.core.session import SessionStore
    from helioai.core.llm.base import Message

    test_store = SessionStore(tmp_path / "sessions.db")
    code_path = str(tmp_path / "sess" / "code_0.py")
    tool_result = json.dumps(
        {
            "stdout": "ok",
            "figure_paths": [],
            "n_figures": 0,
            "exports": {},
            "code_path": code_path,
            "n_lines": 5,
        }
    )
    history = [
        Message(role="user", content="Run some code."),
        Message(role="assistant", content="", tool_calls=[]),
        Message(role="tool", tool_call_id="t1", content=tool_result),
        Message(role="assistant", content="Done."),
    ]
    test_store.save("web", "sess-code", history)

    monkeypatch.setattr("helioai.interfaces.web.app.stream_chat", lambda *a, **kw: (_ for _ in []))
    monkeypatch.setattr("helioai.interfaces.web.app.build_llm_client", lambda provider=None: None)
    monkeypatch.setattr("helioai.interfaces.web.app.store", test_store)

    from helioai.interfaces.web.app import app
    from starlette.testclient import TestClient

    client = TestClient(app)

    r = client.get("/api/sessions/sess-code/messages")
    assert r.status_code == 200
    msgs = r.json()["messages"]
    assert len(msgs) == 2
    assert msgs[1]["role"] == "assistant"
    code_artifacts = msgs[1].get("code", [])
    assert len(code_artifacts) == 1
    art = code_artifacts[0]
    assert art["kind"] == "code"
    assert art["name"] == "code_0.py"
    assert art["n_lines"] == 5
    assert art["code_path"] == code_path


# ── LLM Factory ──────────────────────────────────────────────────────────────


def test_factory_unknown_provider():
    from helioai.core.llm.factory import build_llm_client

    with pytest.raises(RuntimeError, match="Unknown"):
        build_llm_client("unknown_provider_xyz")


def test_factory_default_returns_client():
    """Factory with no override returns a valid LLMClient instance."""
    from helioai.core.llm.base import LLMClient
    from helioai.core.llm.factory import build_llm_client

    client = build_llm_client()
    assert isinstance(client, LLMClient)


# ── Export endpoint ─────────────────────────────────────────────────────────────


def test_export_endpoint(monkeypatch, tmp_path):
    import nbformat
    import helioai.export as export_module
    from helioai.core.llm.base import Message
    from helioai.core.session import SessionStore

    test_store = SessionStore(tmp_path / "sessions.db")
    sid = "web-export-1"
    test_store.save("web", sid, [Message(role="user", content="Plot ACE IMF")])

    monkeypatch.setattr("helioai.interfaces.web.app.store", test_store)
    monkeypatch.setattr(export_module, "store", test_store)
    monkeypatch.setattr(export_module.settings.workspace, "workspace_dir", tmp_path / "ws")

    from helioai.interfaces.web.app import app

    client = TestClient(app, raise_server_exceptions=False)

    r = client.get(f"/api/export?session_id={sid}")
    assert r.status_code == 200
    assert "attachment" in r.headers.get("content-disposition", "")
    nb = nbformat.reads(r.content.decode(), as_version=4)
    nbformat.validate(nb)


def test_export_endpoint_unknown_session(monkeypatch, tmp_path):
    from helioai.core.session import SessionStore

    test_store = SessionStore(tmp_path / "sessions.db")
    monkeypatch.setattr("helioai.interfaces.web.app.store", test_store)

    from helioai.interfaces.web.app import app

    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/api/export?session_id=does-not-exist")
    assert r.status_code == 404
