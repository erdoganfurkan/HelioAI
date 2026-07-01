"""Tests for the user profile feature (4F).

Covers:
- _load_user_profile() helper in agent_loop
- GET/PUT /api/profile endpoints in web app
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient


# ── _load_user_profile ────────────────────────────────────────────────────────


def _write_profile(monkeypatch, tmp_path, user, text):
    from helioai.config import settings
    from helioai.workspace import user_home

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    p = user_home(user) / "profile.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_load_user_profile_missing(tmp_path, monkeypatch):
    from helioai.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)

    from helioai.core.agent_loop import _load_user_profile

    assert _load_user_profile("alice") == ""


def test_load_user_profile_present(tmp_path, monkeypatch):
    _write_profile(monkeypatch, tmp_path, "alice", "preferred missions: ACE, WIND\nunits: SI\n")

    from helioai.core.agent_loop import _load_user_profile

    content = _load_user_profile("alice")
    assert "ACE" in content
    assert "WIND" in content


def test_load_user_profile_per_user(tmp_path, monkeypatch):
    """Each user reads their own profile, not another user's."""
    _write_profile(monkeypatch, tmp_path, "alice", "alice prefs")
    _write_profile(monkeypatch, tmp_path, "bob", "bob prefs")

    from helioai.core.agent_loop import _load_user_profile

    assert _load_user_profile("alice") == "alice prefs"
    assert _load_user_profile("bob") == "bob prefs"


def test_load_user_profile_stripped(tmp_path, monkeypatch):
    _write_profile(monkeypatch, tmp_path, "alice", "  hello  \n\n")

    from helioai.core.agent_loop import _load_user_profile

    assert _load_user_profile("alice") == "hello"


# ── Web endpoints ─────────────────────────────────────────────────────────────


@pytest.fixture
def fake_stream():
    async def _gen(_llm, _user, _sid, _msg):
        yield {"event": "reply", "data": {"text": "ok"}}
        yield {"event": "done", "data": {"n_iterations": 1}}

    return _gen


@pytest.fixture
def web_client(monkeypatch, fake_stream, tmp_path):
    from helioai.config import settings
    from helioai.core.session import SessionStore

    test_store = SessionStore(tmp_path / "sessions.db")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr("helioai.interfaces.web.app.stream_chat", fake_stream)
    monkeypatch.setattr("helioai.interfaces.web.app.build_llm_client", lambda provider=None: None)
    monkeypatch.setattr("helioai.interfaces.web.app.store", test_store)

    from helioai.interfaces.web.app import app

    return TestClient(app, raise_server_exceptions=False)


def test_get_profile_empty(web_client):
    r = web_client.get("/api/profile")
    assert r.status_code == 200
    assert r.json() == {"content": ""}


def test_put_profile_then_get(web_client):
    r = web_client.put("/api/profile", json={"content": "prefer ACE\nunit: km/s"})
    assert r.status_code == 200
    assert r.json() == {"ok": True}

    r2 = web_client.get("/api/profile")
    assert r2.status_code == 200
    assert "ACE" in r2.json()["content"]


def test_put_profile_overwrites(web_client):
    web_client.put("/api/profile", json={"content": "first"})
    web_client.put("/api/profile", json={"content": "second"})
    r = web_client.get("/api/profile")
    assert r.json()["content"] == "second"
