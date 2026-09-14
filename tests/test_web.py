"""Tests for the FastAPI web interface (app.py) and LLM factory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

STATIC_DIR = Path(__file__).parent.parent / "helioai" / "interfaces" / "web" / "static"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_stream():
    """Async generator that replays a scripted sequence of SSE events."""

    async def _gen(_llm, _user, _sid, _msg, *, restricted=True):
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


def test_chat_stream_refuses_a_session_that_is_already_streaming(web_client, monkeypatch):
    """A second tab on a busy session gets a 409 at once, not a stream that seems hung
    while it waits for the first turn's lock."""
    import helioai.interfaces.web.app as web_app

    monkeypatch.setattr(web_app.store, "is_busy", lambda user_id, session_id: True)
    resp = web_client.post("/chat/stream", json={"message": "hello", "session_id": "busy-session"})
    assert resp.status_code == 409
    assert "already streaming" in resp.json()["detail"]


def test_chat_stream_proceeds_when_the_session_is_idle(web_client, monkeypatch):
    import helioai.interfaces.web.app as web_app

    monkeypatch.setattr(web_app.store, "is_busy", lambda user_id, session_id: False)
    with web_client.stream(
        "POST", "/chat/stream", json={"message": "hello", "session_id": "idle-session"}
    ) as resp:
        assert resp.status_code == 200


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
    from helioai.core.llm.base import Message
    from helioai.core.session import SessionStore

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

    from helioai.core.llm.base import Message
    from helioai.core.session import SessionStore

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

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    # path traversal attempt
    r = web_client.get(f"/figure?path={tmp_path}/../etc/passwd")
    assert r.status_code == 404


def test_figure_valid(web_client, tmp_path, monkeypatch):
    from helioai.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)

    fig_dir = tmp_path / "users" / "web" / "workspace" / "sess123" / "run001"
    fig_dir.mkdir(parents=True)
    fig = fig_dir / "fig_0.png"
    fig.write_bytes(b"\x89PNG\r\n")

    r = web_client.get(f"/figure?path={fig}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"


def test_figure_pdf_served(web_client, tmp_path, monkeypatch):
    from helioai.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)

    fig_dir = tmp_path / "users" / "web" / "workspace" / "sess123" / "run001"
    fig_dir.mkdir(parents=True)
    pdf = fig_dir / "fig_0.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    r = web_client.get(f"/figure?path={pdf}")
    assert r.status_code == 200
    assert "application/pdf" in r.headers["content-type"]


def test_figure_unsupported_type_rejected(web_client, tmp_path, monkeypatch):
    from helioai.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)

    fig_dir = tmp_path / "users" / "web" / "workspace" / "sess123"
    fig_dir.mkdir(parents=True)
    txt = fig_dir / "data.txt"
    txt.write_text("not a figure")

    r = web_client.get(f"/figure?path={txt}")
    assert r.status_code == 404


# ── Code endpoint ────────────────────────────────────────────────────────────


def test_code_outside_workspace(web_client):
    r = web_client.get("/code?path=/etc/passwd")
    assert r.status_code == 404


def test_code_path_traversal(web_client, tmp_path, monkeypatch):
    from helioai.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    r = web_client.get(f"/code?path={tmp_path}/../etc/passwd")
    assert r.status_code == 404


def test_code_not_py(web_client, tmp_path, monkeypatch):
    from helioai.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    txt_file = tmp_path / "users" / "web" / "workspace" / "sess" / "data.txt"
    txt_file.parent.mkdir(parents=True)
    txt_file.write_text("not python")
    r = web_client.get(f"/code?path={txt_file}")
    assert r.status_code == 404


def test_code_valid(web_client, tmp_path, monkeypatch):
    from helioai.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    code_dir = tmp_path / "users" / "web" / "workspace" / "sess123"
    code_dir.mkdir(parents=True)
    code_file = code_dir / "code_0.py"
    source = "b = clean(np.array([1.0]))\nparam_card(b, 'amda/imf')\nexport('b', b)\n"
    code_file.write_text(source)

    r = web_client.get(f"/code?path={code_file}")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    assert "param_card" not in r.text  # agent-only call stripped
    assert "def clean(" in r.text and "def export(" in r.text  # shims supplied


def test_session_messages_attach_code(monkeypatch, tmp_path):
    import json

    from helioai.core.llm.base import Message
    from helioai.core.session import SessionStore

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

    from starlette.testclient import TestClient

    from helioai.interfaces.web.app import app

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


def test_factory_default_returns_client(monkeypatch):
    """Factory with no override returns a client for the configured provider.

    Pinned to ollama, the one provider that needs no key: with the default (azure)
    this test only passed when a `.env` or CI-injected key happened to be present,
    so the documented `python -m pytest` failed on a clean clone.
    """
    from helioai.core.llm.base import LLMClient
    from helioai.core.llm.factory import build_llm_client, settings

    monkeypatch.setattr(settings.llm, "provider", "ollama")
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
    monkeypatch.setattr(export_module.settings, "data_dir", tmp_path)

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


# ── Scope guardrail / dev token ───────────────────────────────────────────────


def _make_capturing_stream():
    """Returns a fake stream_chat that records the `restricted` kwarg."""
    captured = {}

    async def _gen(_llm, _user, _sid, _msg, *, restricted=True):
        captured["restricted"] = restricted
        yield {"event": "reply", "data": {"text": "ok"}}
        yield {"event": "done", "data": {"n_iterations": 1}}

    return _gen, captured


def test_chat_stream_restricted_by_default(monkeypatch, tmp_path):
    """Without a dev-token header the request must be restricted."""
    from helioai.core.session import SessionStore

    gen, captured = _make_capturing_stream()
    test_store = SessionStore(tmp_path / "sessions.db")
    monkeypatch.setattr("helioai.interfaces.web.app.stream_chat", gen)
    monkeypatch.setattr("helioai.interfaces.web.app.build_llm_client", lambda provider=None: None)
    monkeypatch.setattr("helioai.interfaces.web.app.store", test_store)

    from helioai.interfaces.web.app import app

    client = TestClient(app, raise_server_exceptions=False)
    with client.stream(
        "POST",
        "/chat/stream",
        json={"message": "hello", "session_id": "s1"},
    ) as resp:
        resp.read()

    assert captured.get("restricted") is True


def test_chat_stream_dev_token_unlocks(monkeypatch, tmp_path):
    """A matching dev token must set restricted=False."""
    from helioai.core.session import SessionStore

    gen, captured = _make_capturing_stream()
    test_store = SessionStore(tmp_path / "sessions.db")
    monkeypatch.setattr("helioai.interfaces.web.app.stream_chat", gen)
    monkeypatch.setattr("helioai.interfaces.web.app.build_llm_client", lambda provider=None: None)
    monkeypatch.setattr("helioai.interfaces.web.app.store", test_store)
    monkeypatch.setattr("helioai.config.settings.dev.token", "secret123")

    from helioai.interfaces.web.app import app

    client = TestClient(app, raise_server_exceptions=False)
    with client.stream(
        "POST",
        "/chat/stream",
        json={"message": "hello", "session_id": "s2"},
        headers={"X-Helio-Dev-Token": "secret123"},
    ) as resp:
        resp.read()

    assert captured.get("restricted") is False


def test_chat_stream_wrong_token_stays_restricted(monkeypatch, tmp_path):
    """A wrong dev token must keep restricted=True."""
    from helioai.core.session import SessionStore

    gen, captured = _make_capturing_stream()
    test_store = SessionStore(tmp_path / "sessions.db")
    monkeypatch.setattr("helioai.interfaces.web.app.stream_chat", gen)
    monkeypatch.setattr("helioai.interfaces.web.app.build_llm_client", lambda provider=None: None)
    monkeypatch.setattr("helioai.interfaces.web.app.store", test_store)
    monkeypatch.setattr("helioai.config.settings.dev.token", "secret123")

    from helioai.interfaces.web.app import app

    client = TestClient(app, raise_server_exceptions=False)
    with client.stream(
        "POST",
        "/chat/stream",
        json={"message": "hello", "session_id": "s3"},
        headers={"X-Helio-Dev-Token": "wrongtoken"},
    ) as resp:
        resp.read()

    assert captured.get("restricted") is True


# ── D.2 auth ──────────────────────────────────────────────────────────────────


@pytest.fixture
def auth_client(web_client, monkeypatch):
    """web_client with HELIOAI_USERS configured (two nominative tokens)."""
    from helioai.config import settings

    monkeypatch.setattr(settings.web_auth, "users", {"tok-v": "vincent", "tok-a": "alice"})
    return web_client


def test_api_requires_token_when_users_configured(auth_client):
    assert auth_client.get("/api/sessions").status_code == 401
    r = auth_client.post("/chat/stream", json={"message": "hi", "session_id": "s1"})
    assert r.status_code == 401


def test_invalid_token_rejected(auth_client):
    r = auth_client.get("/api/sessions", headers={"X-Helio-Token": "nope"})
    assert r.status_code == 401


def test_valid_token_isolates_sessions(auth_client):
    import helioai.interfaces.web.app as web_app
    from helioai.core.llm.base import Message

    web_app.store.save("vincent", "s1", [Message(role="user", content="hi")])
    # alice sees none of vincent's sessions
    r = auth_client.get("/api/sessions", headers={"X-Helio-Token": "tok-a"})
    assert r.status_code == 200
    assert r.json() == []
    # vincent sees his own
    r = auth_client.get("/api/sessions", headers={"X-Helio-Token": "tok-v"})
    assert any(s["session_id"] == "s1" for s in r.json())


def test_no_users_configured_stays_open(web_client):
    # Default: no HELIOAI_USERS → single-user, no token needed (local dev)
    assert web_client.get("/api/sessions").status_code == 200


def test_figure_path_ownership(auth_client, tmp_path, monkeypatch):
    import helioai.interfaces.web.app as web_app
    from helioai.core.llm.base import Message

    monkeypatch.setattr(web_app.settings, "data_dir", tmp_path)
    wdir = tmp_path / "users" / "vincent" / "workspace" / "v-sess"
    wdir.mkdir(parents=True)
    fig = wdir / "fig_0_0.png"
    fig.write_bytes(b"\x89PNG")
    web_app.store.save("vincent", "s1", [Message(role="user", content="hi")])
    web_app.store.set_workspace_dir("vincent", "s1", "v-sess")

    # alice cannot read vincent's figure
    r = auth_client.get("/figure", params={"path": str(fig)}, headers={"X-Helio-Token": "tok-a"})
    assert r.status_code == 404
    # vincent can
    r = auth_client.get("/figure", params={"path": str(fig)}, headers={"X-Helio-Token": "tok-v"})
    assert r.status_code == 200


# ── XSS hardening: vendored deps + sanitized markdown ──────────────────────────


def test_index_html_loads_no_external_cdn_scripts():
    """No third-party CDN in the trust chain — everything served from /static/vendor/."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "cdn.jsdelivr.net" not in html
    assert "cdnjs.cloudflare.com" not in html
    assert "/static/vendor/marked.min.js" in html
    assert "/static/vendor/purify.min.js" in html


def test_app_js_sanitizes_markdown_before_innerHTML():
    """LLM/tool output rendered as markdown must go through DOMPurify before innerHTML."""
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert js.count("DOMPurify.sanitize(marked.parse(") >= 2  # live reply + history replay
    assert "= marked.parse(" not in js  # no unsanitized innerHTML assignment left


def test_every_streamed_event_has_a_web_handler():
    """`plan` was emitted, forwarded by SSE, and silently dropped by the browser.

    The CLI and the notebook rendered it, so it looked implemented. This pins the two
    sides together: any event the loop can emit must be handled in app.js.
    """
    import re
    from pathlib import Path

    import helioai.core.agent_loop as loop
    import helioai.core.sub_agents as subs

    src = Path(loop.__file__).read_text(encoding="utf-8") + Path(subs.__file__).read_text(
        encoding="utf-8"
    )
    emitted = set(re.findall(r'"event":\s*"(\w+)"', src))

    js = (Path(loop.__file__).parents[1] / "interfaces/web/static/app.js").read_text(
        encoding="utf-8"
    )
    handled = set(re.findall(r"event === '(\w+)'", js))

    missing = emitted - handled
    assert not missing, f"events emitted but not rendered in the web UI: {sorted(missing)}"


def test_delete_session_cannot_escape_the_user_workspace(web_client, tmp_path, monkeypatch):
    """DELETE /api/sessions/{id} rmtree'd `<home>/workspace/<label>` with no containment.

    The label is derived from a client-supplied session_id, so `../../..` walked the
    delete out of the user's home. Sanitising the id closes the front door; this pins
    the back one, where the label is already-persisted data from an older build.
    """
    from helioai.config import settings
    from helioai.core.llm.base import Message
    from helioai.interfaces.web.app import store

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    sentinel = tmp_path / "users" / "web" / "catalogs"
    sentinel.mkdir(parents=True)
    (sentinel / "keepme.json").write_text("{}", encoding="utf-8")
    (tmp_path / "users" / "web" / "workspace").mkdir()

    store.save("web", "sess-1", [Message(role="user", content="hi")])
    store.set_workspace_dir("web", "sess-1", "../catalogs")
    assert store.get_workspace_dir("web", "sess-1") == "../catalogs", "test would be vacuous"

    r = web_client.delete("/api/sessions/sess-1")

    assert r.status_code == 200
    assert (sentinel / "keepme.json").exists(), "rmtree escaped the workspace root"


def test_chat_stream_rejects_a_traversal_session_id(web_client):
    """session_id reached the filesystem unvalidated — uuid4 is all we ever mint."""
    r = web_client.post(
        "/chat/stream",
        json={"message": "hello", "session_id": "../../../etc"},
    )
    assert r.status_code == 422


# ── the provider selector must reflect the server ──────────────────────────────


def test_config_endpoint_reports_the_configured_provider(web_client, monkeypatch):
    """The browser had no way to know what the server is set to.

    `<option value="azure">` is first in the markup and nothing selected it, so the UI
    silently sent `azure` on every message whatever `HELIOAI_LLM_PROVIDER` said — a user
    on opencode was talking to Azure, or to a 'missing key' error.
    """
    from helioai.config import settings

    monkeypatch.setattr(settings.llm, "provider", "opencode")
    r = web_client.get("/api/config")
    assert r.status_code == 200
    assert r.json()["provider"] == "opencode"


def test_the_selector_is_not_hardcoded_to_the_first_option():
    """Guards the markup half: app.js must ask the server rather than trust option order."""
    # encoding pinned: app.js holds arrows and emoji, and Windows defaults to cp1252.
    js = (Path(__file__).resolve().parents[1] / "helioai/interfaces/web/static/app.js").read_text(
        encoding="utf-8"
    )
    assert "/api/config" in js


# ── E8: what the browser shows must be what the session did ──────────────────


def _messages_client(monkeypatch, tmp_path, history, sid):
    from helioai.core.session import SessionStore

    test_store = SessionStore(tmp_path / "sessions.db")
    test_store.save("web", sid, history)
    monkeypatch.setattr("helioai.interfaces.web.app.stream_chat", lambda *a, **kw: (_ for _ in []))
    monkeypatch.setattr("helioai.interfaces.web.app.build_llm_client", lambda provider=None: None)
    monkeypatch.setattr("helioai.interfaces.web.app.store", test_store)
    from helioai.interfaces.web.app import app

    return TestClient(app)


def test_session_messages_keeps_the_artifacts_of_an_interrupted_turn(monkeypatch, tmp_path):
    """Audit probe: `user → tool(figure, code)` with no final assistant replayed as the
    user message alone — the figure and the script existed on disk and were invisible.
    """
    from helioai.core.llm.base import Message

    tool_result = json.dumps(
        {"stdout": "", "figure_paths": ["/tmp/ws/fig_0_0.png"], "code_path": "/tmp/ws/code_0.py"}
    )
    history = [
        Message(role="user", content="Plot it."),
        Message(role="assistant", content="", tool_calls=[]),
        Message(role="tool", tool_call_id="t1", content=tool_result),
    ]
    client = _messages_client(monkeypatch, tmp_path, history, "sess-cut")
    msgs = client.get("/api/sessions/sess-cut/messages").json()["messages"]

    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["figures"] == ["/tmp/ws/fig_0_0.png"]
    assert msgs[1]["code"][0]["name"] == "code_0.py"


def test_session_messages_do_not_attach_a_cut_turn_to_the_next_answer(monkeypatch, tmp_path):
    """Audit probe: the figure of an interrupted turn was pinned onto the next assistant
    message — an unrelated answer replayed with someone else's plot.
    """
    from helioai.core.llm.base import Message

    tool_result = json.dumps({"stdout": "", "figure_paths": ["/tmp/ws/fig_0_0.png"]})
    history = [
        Message(role="user", content="Plot it."),
        Message(role="assistant", content="", tool_calls=[]),
        Message(role="tool", tool_call_id="t1", content=tool_result),
        Message(role="user", content="Unrelated question."),
        Message(role="assistant", content="Unrelated answer."),
    ]
    client = _messages_client(monkeypatch, tmp_path, history, "sess-mix")
    msgs = client.get("/api/sessions/sess-mix/messages").json()["messages"]

    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[1]["figures"] == ["/tmp/ws/fig_0_0.png"]
    assert msgs[3]["content"] == "Unrelated answer."
    assert "figures" not in msgs[3]


def test_chat_stream_uses_the_requested_provider_or_the_server_default(monkeypatch, tmp_path):
    """The selector shows a provider; the request must carry that one, and none when the
    browser has not been told the server's setting yet.
    """
    from helioai.core.session import SessionStore

    seen: list = []

    async def _gen(_llm, _user, _sid, _msg, *, restricted=True):
        yield {"event": "done", "data": {"n_iterations": 0}}

    monkeypatch.setattr("helioai.interfaces.web.app.stream_chat", _gen)
    monkeypatch.setattr(
        "helioai.interfaces.web.app.build_llm_client", lambda provider=None: seen.append(provider)
    )
    monkeypatch.setattr("helioai.interfaces.web.app.store", SessionStore(tmp_path / "sessions.db"))
    from helioai.interfaces.web.app import app

    client = TestClient(app)
    client.post("/chat/stream", json={"message": "hi", "session_id": "s1", "provider": "gemini"})
    client.post("/chat/stream", json={"message": "hi", "session_id": "s2"})
    assert seen == ["gemini", None]


def test_the_selector_lists_every_provider_the_factory_accepts():
    """Ollama was configured server-side and absent from the markup, so the browser kept
    `azure` selected and sent it — a local-model user got an Azure key error.
    """
    import re

    from helioai.core.llm.factory import OPENAI_COMPAT

    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    options = set(re.findall(r'<option value="(\w+)"', html))
    assert {"azure", "gemini", *OPENAI_COMPAT} <= options


def test_provider_is_only_sent_once_the_server_setting_is_known():
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "dataset.synced" in js, "an unsynced selector must not override the server"


def test_streams_stay_bound_to_their_session_in_the_real_app_js():
    """Audit probe: a reply still streaming for session A rendered into session B after
    "New session". Reproduced by driving the shipped app.js under Node with a DOM stub —
    the Python tests never load app.js at all.
    """
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    proc = subprocess.run(
        [node, str(Path(__file__).parent / "web" / "test_session_streams.js")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "OK web session streams" in proc.stdout


# ── an injected correction replays as a system note, not as the user's question ──


def test_session_messages_show_an_automated_correction_as_a_system_note(monkeypatch, tmp_path):
    from helioai.core.llm.base import Message
    from helioai.core.session import SessionStore

    test_store = SessionStore(tmp_path / "sessions.db")
    test_store.save(
        "web",
        "sess-corr",
        [
            Message(role="user", content="MMS1 FGM in GSM?"),
            Message(role="assistant", content="Use cda/BOGUS/id."),
            Message(
                role="user",
                content="⚠️ AUTOMATED CORRECTION — not in the catalogue",
                origin="correction",
            ),
            Message(role="assistant", content="Use cda/MMS1_FGM_SRVY_L2/b_gsm."),
        ],
    )
    monkeypatch.setattr("helioai.interfaces.web.app.store", test_store)
    from helioai.interfaces.web.app import app

    msgs = TestClient(app).get("/api/sessions/sess-corr/messages").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "system", "assistant"]
    assert msgs[2]["origin"] == "correction"
    assert all("AUTOMATED CORRECTION" not in m["content"] for m in msgs if m["role"] == "user")


# ── hardening that used to be untestable ───────────────────────────────────────


def test_valid_token_resolves_its_user_and_a_prefix_does_not(auth_client):
    assert auth_client.get("/api/sessions", headers={"X-Helio-Token": "tok-v"}).status_code == 200
    assert auth_client.get("/api/sessions", headers={"X-Helio-Token": "tok-"}).status_code == 401
    assert auth_client.get("/api/sessions", headers={"X-Helio-Token": "tok-vv"}).status_code == 401


def test_tokens_are_compared_in_constant_time(auth_client, monkeypatch):
    """`token in users` is a dict lookup whose timing depends on the match; the check
    goes through hmac.compare_digest like the dev token and the MCP bearer."""
    import hmac as hmac_module

    import helioai.interfaces.web.app as web_app

    seen: list[tuple[str, str]] = []
    real = hmac_module.compare_digest

    def spy(a, b):
        seen.append((a, b))
        return real(a, b)

    monkeypatch.setattr(web_app.hmac, "compare_digest", spy)
    auth_client.get("/api/sessions", headers={"X-Helio-Token": "tok-a"})
    assert ("tok-a", "tok-a") in seen


def test_every_response_carries_a_content_security_policy(web_client):
    r = web_client.get("/health")
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"


def test_loopback_bind_pins_the_host_header(monkeypatch):
    """The DNS-rebinding guard was added inside serve_web, on an app the TestClient
    never saw; harden_for_host builds exactly what uvicorn serves."""
    from fastapi import FastAPI

    from helioai.interfaces.web.app import harden_for_host

    probe = FastAPI()

    @probe.get("/ping")
    async def ping():
        return {"ok": True}

    client = TestClient(harden_for_host(probe, "127.0.0.1"), raise_server_exceptions=False)
    assert client.get("/ping", headers={"Host": "localhost"}).status_code == 200
    assert client.get("/ping", headers={"Host": "evil.example"}).status_code == 400


def test_public_bind_does_not_pin_the_host(monkeypatch):
    from fastapi import FastAPI

    from helioai.interfaces.web.app import harden_for_host

    probe = FastAPI()

    @probe.get("/ping")
    async def ping():
        return {"ok": True}

    client = TestClient(harden_for_host(probe, "0.0.0.0"), raise_server_exceptions=False)
    assert client.get("/ping", headers={"Host": "helio.lab.example"}).status_code == 200


# ── a public bind without anyone authenticated is a deployment error ───────────


def test_public_bind_without_users_refuses_to_start(monkeypatch):
    from helioai.config import settings
    from helioai.interfaces.web.app import refuse_unauthenticated_public_bind

    monkeypatch.setattr(settings.web_auth, "users", {})
    monkeypatch.setattr(settings.web_auth, "allow_unauthenticated_public", False)
    with pytest.raises(SystemExit):
        refuse_unauthenticated_public_bind("0.0.0.0")


def test_public_bind_with_users_or_the_explicit_opt_out_starts(monkeypatch):
    from helioai.config import settings
    from helioai.interfaces.web.app import refuse_unauthenticated_public_bind

    monkeypatch.setattr(settings.web_auth, "users", {"t": "u"})
    refuse_unauthenticated_public_bind("0.0.0.0")

    monkeypatch.setattr(settings.web_auth, "users", {})
    monkeypatch.setattr(settings.web_auth, "allow_unauthenticated_public", True)
    refuse_unauthenticated_public_bind("0.0.0.0")


def test_loopback_bind_never_needs_users(monkeypatch):
    from helioai.config import settings
    from helioai.interfaces.web.app import refuse_unauthenticated_public_bind

    monkeypatch.setattr(settings.web_auth, "users", {})
    monkeypatch.setattr(settings.web_auth, "allow_unauthenticated_public", False)
    for host in ("127.0.0.1", "localhost", "::1"):
        refuse_unauthenticated_public_bind(host)


def test_api_me_reports_the_callers_usage(web_client):
    import helioai.interfaces.web.app as web_app

    web_app.store.record_usage(
        "web", "s", turn=1, agent="lead", provider="groq", prompt_tokens=100, completion_tokens=20
    )
    body = web_client.get("/api/me").json()
    assert body["user_id"] == "web"
    assert body["usage"]["total"]["prompt_tokens"] == 100
    assert body["usage"]["day"]["n_calls"] == 1
