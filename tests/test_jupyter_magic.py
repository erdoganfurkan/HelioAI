"""Tests for Jupyter IPython magics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_magic_state():
    """`_SESSION_ID` is the only module state left to restore.

    There used to be a cached `_llm` here too. It is gone: `_run_async` gives each
    cell its own event loop, and a cached async client kept sockets from a loop
    that had since closed, so the second `%%helioai` of a session died in the
    connection pool with `RuntimeError: Event loop is closed`.
    """
    import helioai.interfaces.jupyter_magic as magic

    original_session = magic._SESSION_ID
    yield
    magic._SESSION_ID = original_session


def _make_store(summaries=None, all_ids=None, messages=None):
    store = MagicMock()
    store.list_summaries.return_value = summaries or []
    store.all_sessions.return_value = all_ids or []
    store.get_or_create.return_value = messages or []
    return store


def _make_magic():
    from helioai.interfaces.jupyter_magic import HelioAIMagics

    return HelioAIMagics(None)


# --- load_ipython_extension ---


def test_load_extension_registers_magics():
    from helioai.interfaces.jupyter_magic import HelioAIMagics, load_ipython_extension

    ip = MagicMock()
    load_ipython_extension(ip)
    ip.register_magics.assert_called_once()
    assert ip.register_magics.call_args[0][0] is HelioAIMagics


# --- %helioai_session ---


def test_session_reset_changes_id(monkeypatch, capsys):
    import helioai.core.session as s
    import helioai.interfaces.jupyter_magic as magic

    monkeypatch.setattr(s, "store", _make_store())
    original_id = magic._SESSION_ID
    _make_magic().helioai_session("reset")
    assert original_id != magic._SESSION_ID
    assert "Session reset" in capsys.readouterr().out


def test_session_unknown_command(capsys):
    _make_magic().helioai_session("unknown")
    assert "Unknown command" in capsys.readouterr().out


# --- %helioai_provider ---


def _run_cell_and_capture_provider(magic, monkeypatch) -> list:
    """Run one `%%helioai` cell with the loop and the LLM factory stubbed out; return
    the `provider` arguments the factory received."""
    import helioai.interfaces.jupyter_magic as jm

    built: list = []

    def fake_build_llm_client(provider=None):
        built.append(provider)
        client = MagicMock()

        async def aclose():
            return None

        client.aclose = aclose
        return client

    async def fake_stream_chat(llm, user_id, session_id, text, *, restricted=True):
        yield {"event": "done", "data": {"n_iterations": 0}}

    monkeypatch.setattr("helioai.core.llm.factory.build_llm_client", fake_build_llm_client)
    monkeypatch.setattr("helioai.core.agent_loop.stream_chat", fake_stream_chat)
    monkeypatch.setattr(jm, "setup_logging", lambda *a, **k: None, raising=False)
    magic.helioai("", "plot IMF Bz")
    return built


def test_provider_switch_is_what_the_next_cell_builds(monkeypatch, capsys):
    """`%helioai_provider gemini` must change the client the next cell is built with.

    It used to set `HELIOAI_LLM_PROVIDER` in the environment — which `settings`
    reads exactly once, at import — so the switch printed a confirmation and changed
    nothing. The old test asserted the environment variable and so passed with the
    bug in place.
    """
    magic = _make_magic()
    magic.helioai_provider("gemini")
    assert "gemini" in capsys.readouterr().out

    built = _run_cell_and_capture_provider(magic, monkeypatch)
    assert built == ["gemini"]


def test_provider_unset_lets_the_factory_use_the_configured_default(monkeypatch):
    built = _run_cell_and_capture_provider(_make_magic(), monkeypatch)
    assert built == [None]


def test_provider_switch_does_not_touch_the_environment(monkeypatch):
    import os

    monkeypatch.delenv("HELIOAI_LLM_PROVIDER", raising=False)
    _make_magic().helioai_provider("gemini")
    assert "HELIOAI_LLM_PROVIDER" not in os.environ


def test_provider_without_argument_shows_the_current_one(capsys):
    from helioai.config import settings

    magic = _make_magic()
    magic.helioai_provider("")
    out = capsys.readouterr().out
    assert settings.llm.provider in out
    assert "Unknown provider" not in out

    magic.helioai_provider("groq")
    capsys.readouterr()
    magic.helioai_provider("")
    assert "groq" in capsys.readouterr().out


def test_provider_accepts_ollama(capsys):
    """The README advertises ollama; the magic used to reject it."""
    _make_magic().helioai_provider("ollama")
    assert "ollama" in capsys.readouterr().out.lower()
    assert "Unknown provider" not in capsys.readouterr().out


def test_provider_invalid_keeps_the_previous_choice(capsys):
    magic = _make_magic()
    magic.helioai_provider("groq")
    capsys.readouterr()
    magic.helioai_provider("openai")
    assert "Unknown provider" in capsys.readouterr().out
    assert magic._provider == "groq"


# --- %helioai_history ---


def test_history_empty(monkeypatch, capsys):
    import helioai.core.session as s

    monkeypatch.setattr(s, "store", _make_store())
    _make_magic().helioai_history("")
    assert "No history" in capsys.readouterr().out


def test_history_renders_html(monkeypatch):
    import helioai.core.session as s

    summaries = [
        {
            "session_id": "abc12345-xyz",
            "updated_at": "2026-05-31T10:00:00Z",
            "n_messages": 5,
            "first_message": "solar wind query",
        },
    ]
    monkeypatch.setattr(s, "store", _make_store(summaries))
    with patch("helioai.interfaces.jupyter_magic.display") as mock_display:
        _make_magic().helioai_history("")
        mock_display.assert_called_once()
        html_obj = mock_display.call_args[0][0]
        assert "abc12345" in html_obj.data
        assert "solar wind query" in html_obj.data
        assert "<table>" in html_obj.data


# --- %helioai_resume ---


def test_resume_valid_prefix(monkeypatch, capsys):
    import helioai.core.session as s
    import helioai.interfaces.jupyter_magic as magic

    full_id = "abc12345-0000-rest"
    monkeypatch.setattr(s, "store", _make_store(all_ids=[full_id], messages=[]))
    _make_magic().helioai_resume("abc12345")
    assert full_id == magic._SESSION_ID
    assert "Resumed" in capsys.readouterr().out


def test_resume_no_match(monkeypatch, capsys):
    import helioai.core.session as s

    monkeypatch.setattr(s, "store", _make_store(all_ids=[]))
    _make_magic().helioai_resume("nonexistent")
    assert "No session found" in capsys.readouterr().out


def test_resume_no_arg(capsys):
    _make_magic().helioai_resume("")
    assert "Usage" in capsys.readouterr().out


def test_resume_exact_match(monkeypatch, capsys):
    import helioai.core.session as s
    import helioai.interfaces.jupyter_magic as magic

    full_id = "exact-session-id"
    monkeypatch.setattr(
        s, "store", _make_store(all_ids=[full_id], messages=[MagicMock(), MagicMock()])
    )
    _make_magic().helioai_resume(full_id)
    assert full_id == magic._SESSION_ID
    assert "2 messages" in capsys.readouterr().out


# ── HTML card helpers ──────────────────────────────────────────────────────────


def test_param_card_html_contains_param_id():
    from helioai.interfaces.jupyter_magic import _param_card_html

    data = {
        "param_id": "amda/imf_gsm",
        "name": "IMF B GSM",
        "mission": "ACE",
        "instrument": "MAG",
        "units": "nT",
        "cadence": "16 s",
        "coord_sys": "GSM",
        "components": ["Bx", "By", "Bz"],
        "n_points": 3600,
    }
    html = _param_card_html(data)
    assert "amda/imf_gsm" in html
    assert "ACE" in html
    assert "nT" in html
    assert "#58a6ff" in html  # dark-theme border color


def test_param_card_html_escapes_xss():
    from helioai.interfaces.jupyter_magic import _param_card_html

    data = {"param_id": "<script>alert(1)</script>", "name": "", "mission": "", "units": ""}
    html = _param_card_html(data)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_catalog_card_html_contains_catalog_id():
    from helioai.interfaces.jupyter_magic import _catalog_card_html

    data = {
        "catalog_id": "amda/Richardson_Cane_ICME_list",
        "nb_events_total": 341,
        "nb_events_filtered": 20,
        "columns": ["start", "stop", "v_transit"],
    }
    html = _catalog_card_html(data)
    assert "Richardson_Cane" in html
    assert "20" in html
    assert "#3fb950" in html  # dark-theme border color


def test_render_jupyter_event_parameter_card_calls_display(monkeypatch):
    from helioai.interfaces import jupyter_magic as magic

    displayed = []
    monkeypatch.setattr(magic, "display", lambda x: displayed.append(x))

    magic._render_jupyter_event(
        {
            "event": "artifact",
            "data": {
                "kind": "parameter_card",
                "param_id": "amda/vsw",
                "name": "Solar wind speed",
                "mission": "ACE",
                "units": "km/s",
                "cadence": "64 s",
                "coord_sys": "",
                "components": [],
                "n_points": 100,
                "instrument": "",
            },
        }
    )
    assert len(displayed) == 1
    from IPython.display import HTML

    assert isinstance(displayed[0], HTML)


def test_get_llm_is_never_cached_across_cells():
    """Each cell must get its own client.

    `_run_async` runs every cell in a fresh `asyncio.run` loop. An async HTTP
    client binds its connection pool to the loop that first used it, so a client
    reused across cells holds sockets from a closed loop and the next request
    fails inside httpcore with `RuntimeError: Event loop is closed` — surfaced to
    the user as a bare `APIConnectionError: Connection error`.

    Returning a distinct object each call is the invariant that prevents it.
    """
    import helioai.interfaces.jupyter_magic as magic

    built = []

    def fake_build_llm_client(provider=None):
        client = object()
        built.append(client)
        return client

    with patch("helioai.core.llm.factory.build_llm_client", fake_build_llm_client):
        first = magic._get_llm()
        second = magic._get_llm()

    assert first is not second, "a cached client would carry a dead event loop into the next cell"
    assert len(built) == 2


def test_session_new_starts_fresh_without_deleting_the_previous_one(monkeypatch, capsys):
    """`reset` wipes the current session from the store. A demo that has just exported
    an analysis and wants a clean slate for the next question must not lose it —
    `new` mints an id and touches nothing.
    """
    import helioai.core.session as s
    import helioai.interfaces.jupyter_magic as magic

    store = _make_store()
    monkeypatch.setattr(s, "store", store)
    original_id = magic._SESSION_ID
    _make_magic().helioai_session("new")
    assert magic._SESSION_ID != original_id
    store.reset.assert_not_called()
    assert "New session" in capsys.readouterr().out
