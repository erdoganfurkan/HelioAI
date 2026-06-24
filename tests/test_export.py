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
    # agent-only param_card() is stripped from standalone code
    assert all("param_card" not in s for s in code_sources)


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
    from helioai.export import _SETUP_CELL_BASE

    assert "def clean" in _SETUP_CELL_BASE


def test_exported_notebook_clean_shim_is_callable(wired, tmp_path) -> None:
    """A code cell using clean() should execute without NameError."""
    from helioai.export import _SETUP_CELL_BASE

    # Execute setup cell to populate namespace, then call clean()
    ns: dict = {}
    exec(_SETUP_CELL_BASE, ns)  # noqa: S102
    import numpy as np

    result = ns["clean"](np.array([1.0, 1e31, -1e31, float("inf"), float("-inf"), 2.0]))
    assert np.isnan(result[1])
    assert np.isnan(result[2])
    assert np.isnan(result[3])
    assert np.isnan(result[4])
    assert result[0] == pytest.approx(1.0)
    assert result[5] == pytest.approx(2.0)


# ── Chantier A: standalone rewrite ───────────────────────────────────────────


def test_rewrite_timeseries_load_data_to_get_data() -> None:
    from helioai.export import _rewrite_load_data_calls

    manifest = {
        "datasets": {
            "imf_gsm": {
                "kind": "timeseries",
                "param_id": "amda/imf_gsm",
                "start": "2005-01-17",
                "stop": "2005-01-18",
            }
        }
    }
    code = 'd = load_data("imf_gsm")\nplt.plot(d.time, d.values)'
    out = _rewrite_load_data_calls(code, manifest)
    assert 'spz.get_data("amda/imf_gsm", "2005-01-17", "2005-01-18")' in out
    assert "load_data(" not in out


def test_rewrite_unknown_dataset_left_intact() -> None:
    from helioai.export import _rewrite_load_data_calls

    code = 'd = load_data("mystery")'
    out = _rewrite_load_data_calls(code, {"datasets": {}})
    assert 'load_data("mystery")' in out  # never emit a wrong spz.get_data call
    assert "spz.get_data" not in out


def test_rewrite_event_collection_reconstructed() -> None:
    from helioai.export import _rewrite_load_data_calls

    manifest = {
        "datasets": {
            "shocks_events": {
                "kind": "event_collection",
                "param_id": "amda/imf_gsm",
                "events": [
                    {
                        "idx": 0,
                        "start": "2005-01-17T01:00",
                        "stop": "2005-01-17T02:00",
                        "status": "ok",
                    },
                    {
                        "idx": 1,
                        "start": "2005-02-01T00:00",
                        "stop": "2005-02-01T01:00",
                        "status": "no_data",
                    },
                    {
                        "idx": 2,
                        "start": "2005-03-10T05:00",
                        "stop": "2005-03-10T06:00",
                        "status": "ok",
                    },
                ],
            }
        }
    }
    code = 'evs = load_data("shocks_events")'
    out = _rewrite_load_data_calls(code, manifest)
    assert "load_data(" not in out
    assert 'spz.get_data("amda/imf_gsm"' in out
    assert "2005-01-17T01:00" in out and "2005-03-10T05:00" in out
    assert "2005-02-01T00:00" not in out  # no_data event excluded


def test_rewrite_event_collection_without_param_id_kept() -> None:
    from helioai.export import _rewrite_load_data_calls

    manifest = {"datasets": {"x_events": {"kind": "event_collection", "events": []}}}
    code = 'evs = load_data("x_events")'
    out = _rewrite_load_data_calls(code, manifest)
    assert 'load_data("x_events")' in out


# ── Chantier 2: to_standalone (strip agent-only + header) ────────────────────


def test_to_standalone_strips_agent_only_calls() -> None:
    from helioai.export import to_standalone

    code = (
        "b = clean(x)\n"
        'param_card(b, "amda/imf")\n'
        'document_method("MVAB", "Sonnerup (1998)", "min variance")\n'
        'export("beta", b)\n'
    )
    out = to_standalone(code, {"datasets": {}})
    assert "param_card" not in out
    assert "document_method" not in out
    assert "clean(x)" in out and 'export("beta", b)' in out  # scientific calls kept


def test_to_standalone_header_defines_used_helpers() -> None:
    from helioai.export import to_standalone

    code = "b = clean(spz.get_data('amda/imf', s, e).values)\nexport('b', b)\n"
    out = to_standalone(code, {"datasets": {}})
    assert "import speasy as spz" in out
    assert "import numpy as np" in out
    assert "def clean(" in out
    assert "def export(" in out
    import ast

    ast.parse(out)  # standalone code is syntactically valid


def test_to_standalone_no_header_for_notebook() -> None:
    from helioai.export import to_standalone

    code = "param_card(b, 'x')\nplt.plot(t, y)\n"
    out = to_standalone(code, {"datasets": {}}, with_header=False)
    assert "import" not in out  # setup cell carries imports
    assert "param_card" not in out


def test_setup_cell_drops_load_data_shim_when_all_rewritten(wired) -> None:
    # the wired run uses spz.get_data directly (no load_data) → shim must be absent
    path = export_session_notebook(_USER, _SESSION)
    nb = nbformat.read(str(path), as_version=4)
    code_sources = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "def clean" in code_sources  # base shims kept
    assert "def load_data" not in code_sources


# ── Chantier B: Methods section ──────────────────────────────────────────────


def test_methods_section_lists_recipes(monkeypatch, tmp_path) -> None:
    from helioai.export import build_notebook

    store = SessionStore(tmp_path / "sessions.db")
    workspace = tmp_path / "workspace"
    (workspace / _LABEL).mkdir(parents=True)

    history = [
        Message(role="user", content="Find a shock and compute theta_Bn"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="r1", name="load_recipe", arguments={"name": "theta_bn"})],
        ),
        Message(role="tool", tool_call_id="r1", content='{"name": "theta_bn", "code": "..."}'),
        Message(role="assistant", content="theta_Bn computed."),
    ]
    store.save(_USER, _SESSION, history)
    store.set_workspace_dir(_USER, _SESSION, _LABEL)
    monkeypatch.setattr(export_module, "store", store)
    monkeypatch.setattr(export_module.settings.workspace, "workspace_dir", workspace)

    nb = build_notebook(_USER, _SESSION)
    md = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "Methods & data acknowledgements" in md
    assert "theta_bn" in md
    assert "Schwartz" in md  # reference pulled from the recipe header
