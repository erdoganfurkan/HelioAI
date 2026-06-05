"""Tests for helioai.export — reproducible notebook generation."""

from __future__ import annotations

import nbformat
import pytest

import helioai.export as export_module
from helioai.core.llm.base import Message, ToolCall
from helioai.core.session import SessionStore
from helioai.export import export_session_notebook

_USER = "tester"
_SESSION = "sess-abc-123456"
_LABEL = "plot-imf-bz_abc123"


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """A fresh store on a tmp DB + a tmp workspace with one saved run."""
    store = SessionStore(tmp_path / "sessions.db")
    workspace = tmp_path / "workspace"
    (workspace / _LABEL).mkdir(parents=True)
    (workspace / _LABEL / "code_0.py").write_text(
        "var = spz.get_data('amda/imf', start, stop)\nparam_card(var, 'amda/imf')\n",
        encoding="utf-8",
    )

    history = [
        Message(role="user", content="Plot IMF Bz from ACE on 2005-01-17"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="get_timeseries",
                    arguments={"param_id": "amda/imf", "start": "2005-01-17", "stop": "2005-01-18"},
                ),
            ],
        ),
        Message(
            role="tool", tool_call_id="t1", content='{"param_id": "amda/imf", "preview": "..."}'
        ),
        Message(role="assistant", content="Here is the IMF Bz time series."),
    ]
    store.save(_USER, _SESSION, history)
    store.set_workspace_dir(_USER, _SESSION, _LABEL)

    monkeypatch.setattr(export_module, "store", store)
    monkeypatch.setattr(export_module.settings.workspace, "workspace_dir", workspace)
    return workspace


def test_export_produces_valid_notebook(wired) -> None:
    path = export_session_notebook(_USER, _SESSION)
    assert path.exists()
    assert path.suffix == ".ipynb"
    nb = nbformat.read(str(path), as_version=4)
    nbformat.validate(nb)


def test_export_includes_provenance(wired) -> None:
    path = export_session_notebook(_USER, _SESSION)
    nb = nbformat.read(str(path), as_version=4)
    first = nb.cells[0]
    assert first.cell_type == "markdown"
    assert "HelioAI session export" in first.source
    assert _SESSION in first.source
    assert "amda/imf" in first.source  # param id collected


def test_export_includes_saved_code(wired) -> None:
    path = export_session_notebook(_USER, _SESSION)
    nb = nbformat.read(str(path), as_version=4)
    code_sources = [c.source for c in nb.cells if c.cell_type == "code"]
    assert any("spz.get_data('amda/imf'" in s for s in code_sources)
    # setup cell defines the export/param_card shims so saved runs execute standalone
    assert any("def param_card" in s for s in code_sources)


def test_export_includes_conversation(wired) -> None:
    path = export_session_notebook(_USER, _SESSION)
    nb = nbformat.read(str(path), as_version=4)
    md = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "Plot IMF Bz from ACE" in md
    assert "Here is the IMF Bz time series." in md


def test_export_default_path_uses_label(wired) -> None:
    path = export_session_notebook(_USER, _SESSION)
    assert path.name == f"{_LABEL}.ipynb"


def test_export_custom_out_path(wired, tmp_path) -> None:
    out = tmp_path / "custom.ipynb"
    path = export_session_notebook(_USER, _SESSION, out_path=out)
    assert path == out
    assert out.exists()


def test_setup_cell_defines_clean() -> None:
    from helioai.export import _SETUP_CELL

    assert "def clean" in _SETUP_CELL


def test_exported_notebook_clean_shim_is_callable(wired, tmp_path) -> None:
    """A code cell using clean() should execute without NameError."""
    from helioai.export import _SETUP_CELL

    # Execute setup cell to populate namespace, then call clean()
    ns: dict = {}
    exec(_SETUP_CELL, ns)  # noqa: S102
    import numpy as np

    result = ns["clean"](np.array([1.0, 1e31, -1e31, float("inf"), float("-inf"), 2.0]))
    assert np.isnan(result[1])
    assert np.isnan(result[2])
    assert np.isnan(result[3])
    assert np.isnan(result[4])
    assert result[0] == pytest.approx(1.0)
    assert result[5] == pytest.approx(2.0)
