"""Tests for D.5 — per-user storage namespacing (data/users/<user>/)."""

from __future__ import annotations

import json

import pytest

from helioai.config import settings


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    return tmp_path


def test_workspace_dir_is_user_namespaced(_data_dir):
    import helioai.workspace as ws

    ws.set_user("vincent")
    ws.set_label("plot-imf_abc123")
    d = ws.get_session_dir()
    assert d == _data_dir / "users" / "vincent" / "workspace" / "plot-imf_abc123"
    assert d.is_dir()


def test_user_home_and_default(_data_dir):
    import helioai.workspace as ws

    assert ws.current_user() == "web"  # no user set → default
    assert ws.user_home("alice") == _data_dir / "users" / "alice"


@pytest.mark.asyncio
async def test_catalog_same_name_no_collision(_data_dir):
    from helioai.tools.catalog_tools import save_catalog
    from helioai.workspace import set_user

    events = [{"start": "2005-01-01T00:00:00", "stop": "2005-01-01T06:00:00"}]

    set_user("vincent")
    r1 = await save_catalog("shocks", events, description="vincent's")
    set_user("alice")
    r2 = await save_catalog("shocks", events, description="alice's")

    assert r1["catalog_id"] == r2["catalog_id"] == "local/shocks"
    assert r2["overwritten"] is False  # alice did NOT overwrite vincent

    vfile = _data_dir / "users" / "vincent" / "catalogs" / "shocks.json"
    afile = _data_dir / "users" / "alice" / "catalogs" / "shocks.json"
    assert vfile.exists() and afile.exists()
    assert json.loads(vfile.read_text())["description"] == "vincent's"
    assert json.loads(afile.read_text())["description"] == "alice's"


def test_cleanup_old_runs_spares_user_homes(_data_dir):
    import os
    import time

    import helioai.workspace as ws

    ws.set_user("vincent")
    ws.set_label("old_sess")
    old = ws.get_session_dir()
    past = time.time() - 10 * 86400
    os.utime(old, (past, past))

    removed = ws.cleanup_old_runs(ttl_seconds=7 * 86400)
    assert removed == 1
    assert not old.exists()
    assert (_data_dir / "users" / "vincent").is_dir()  # home preserved


def test_migrate_storage_idempotent(_data_dir, monkeypatch):
    from helioai import config
    from helioai.interfaces.cli import _run_migrate_storage

    # The split-directory step moves `chroma/` from the *default* data directory to the
    # configured one. Both live under tmp_path here (the autouse fixture redirects the
    # default too); the assertion below is what makes that redirection load-bearing.
    legacy_root = config._default_data_dir()
    assert str(legacy_root).startswith(str(_data_dir.parent)), legacy_root
    (legacy_root / "chroma").mkdir(parents=True)
    (legacy_root / "chroma" / "marker").write_text("index")

    # legacy flat catalog + profile
    legacy_cat = _data_dir / "catalogs"
    legacy_cat.mkdir(parents=True)
    (legacy_cat / "icmes.json").write_text('{"name": "icmes", "events": []}')
    monkeypatch.setattr(settings.catalogs, "catalogs_dir", legacy_cat)

    legacy_prof = _data_dir / "profiles"
    legacy_prof.mkdir(parents=True)
    (legacy_prof / "vincent.md").write_text("vincent profile")
    monkeypatch.setattr(settings.profile, "profile_path", _data_dir / "profile.md")

    _run_migrate_storage()
    assert (_data_dir / "users" / "web" / "catalogs" / "icmes.json").exists()
    assert (_data_dir / "users" / "vincent" / "profile.md").read_text() == "vincent profile"
    assert (settings.rag.chroma_dir / "marker").exists(), (
        "the legacy index moved to the configured dir"
    )
    assert not (legacy_root / "chroma").exists()

    # rerun: no crash, no duplication
    _run_migrate_storage()
    assert (_data_dir / "users" / "web" / "catalogs" / "icmes.json").exists()


async def test_periodic_cleanup_sweeps_on_every_tick_and_survives_a_failing_sweep(monkeypatch):
    """The web server swept once at startup and then ran for weeks; a demo machine held
    76 expired sessions of 269 MB each. The sweep must also outlive its own errors."""
    import asyncio

    import helioai.workspace as ws

    calls: list[int] = []

    def flaky_cleanup():
        calls.append(len(calls))
        if len(calls) == 1:
            raise OSError("disk went away")
        return 2

    monkeypatch.setattr(ws, "cleanup_old_runs", flaky_cleanup)
    task = asyncio.create_task(ws.cleanup_periodically(period_seconds=0.01))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if len(calls) >= 3:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(calls) >= 3


def test_notebook_and_mcp_sweep_expired_sessions_on_load(monkeypatch):
    """The CLI swept at startup; a notebook kernel and the MCP server never did."""
    from unittest.mock import MagicMock

    import helioai.mcp_server as ms
    import helioai.workspace as ws
    from helioai.interfaces.jupyter_magic import load_ipython_extension

    swept: list[str] = []
    monkeypatch.setattr(ws, "cleanup_old_runs", lambda: swept.append("swept") or 0)

    load_ipython_extension(MagicMock())
    assert swept == ["swept"]

    monkeypatch.setattr(ms, "serve_http", lambda host, port: None)
    monkeypatch.setattr(ms.sys, "argv", ["helioai-mcp", "--http"])
    ms.main()
    assert swept == ["swept", "swept"]
