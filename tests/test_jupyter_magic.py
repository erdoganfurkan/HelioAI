"""Tests for Jupyter IPython magics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_magic_state():
    import helioai.interfaces.jupyter_magic as magic

    original_session = magic._SESSION_ID
    original_llm = magic._llm
    yield
    magic._SESSION_ID = original_session
    magic._llm = original_llm


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
    assert magic._SESSION_ID != original_id
    assert "Session reset" in capsys.readouterr().out


def test_session_unknown_command(capsys):
    _make_magic().helioai_session("unknown")
    assert "Unknown command" in capsys.readouterr().out


# --- %helioai_provider ---


def test_provider_valid_switches_llm(monkeypatch, capsys):
    import helioai.interfaces.jupyter_magic as magic

    magic._llm = MagicMock()
    _make_magic().helioai_provider("gemini")
    import os

    assert os.environ.get("HELIOAI_LLM_PROVIDER") == "gemini"
    assert magic._llm is None
    assert "gemini" in capsys.readouterr().out


def test_provider_invalid(capsys):
    _make_magic().helioai_provider("openai")
    assert "Unknown provider" in capsys.readouterr().out


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
    assert magic._SESSION_ID == full_id
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
    assert magic._SESSION_ID == full_id
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
