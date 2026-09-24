"""Tests for helioai.export — reproducible notebook generation."""

from __future__ import annotations

import json
import sys

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
    workspace = tmp_path / "users" / _USER / "workspace"
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
    monkeypatch.setattr(export_module.settings, "data_dir", tmp_path)
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


def test_export_reports_helioai_version_without_distribution_metadata(wired, monkeypatch) -> None:
    """The distribution is `helioai-agent`, so `version("helioai")` finds nothing.

    On a fresh install that lookup raises and `_version` degrades to "unknown" —
    silently, in every exported notebook. The dev venv still carries the old
    `helioai` metadata, so the miss is simulated here; both the provenance header
    and the Methods footer must fall back on `helioai.__version__` instead.
    """
    import helioai

    def _no_dist(pkg):
        raise export_module.PackageNotFoundError(pkg)

    monkeypatch.setattr(export_module, "version", _no_dist)
    path = export_session_notebook(_USER, _SESSION)
    md = "\n".join(
        c.source for c in nbformat.read(str(path), as_version=4).cells if c.cell_type == "markdown"
    )
    assert f"**helioai:** {helioai.__version__}" in md
    assert f"helioai {helioai.__version__}._" in md
    assert "helioai unknown" not in md
    assert "**helioai:** unknown" not in md


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
    assert 'fetch_series("amda/imf_gsm", "2005-01-17", "2005-01-18")' in out
    assert "load_data(" not in out


def test_rewrite_unknown_dataset_left_intact() -> None:
    from helioai.export import _rewrite_load_data_calls

    code = 'd = load_data("mystery")'
    out = _rewrite_load_data_calls(code, {"datasets": {}})
    assert 'load_data("mystery")' in out  # never emit a wrong fetch call
    assert "fetch_series" not in out


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
    assert 'fetch_events("amda/imf_gsm"' in out
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


def test_to_standalone_does_not_duplicate_existing_imports() -> None:
    from helioai.export import to_standalone

    # Sandbox code already ships its own numpy/matplotlib imports.
    code = "import numpy as np\nimport matplotlib.pyplot as plt\n\nplt.plot(np.arange(3))\n"
    out = to_standalone(code, {"datasets": {}})
    assert out.count("import numpy as np") == 1
    assert out.count("import matplotlib.pyplot as plt") == 1

    # combined form: `import numpy as np, matplotlib.pyplot as plt`
    combined = "import numpy as np, matplotlib.pyplot as plt\n\nplt.plot(np.arange(3))\n"
    out = to_standalone(combined, {"datasets": {}})
    assert out.count("import numpy as np") == 1
    assert out.count("matplotlib.pyplot as plt") == 1


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
    workspace = tmp_path / "users" / _USER / "workspace"
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
    monkeypatch.setattr(export_module.settings, "data_dir", tmp_path)

    nb = build_notebook(_USER, _SESSION)
    md = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "Methods & data acknowledgements" in md
    assert "theta_bn" in md
    assert "Schwartz" in md  # reference pulled from the recipe header


def test_physics_helper_reemitted_standalone():
    from helioai.export import to_standalone

    code = "theta, r = mp_shue1998(2.0, 0.0)\nprint(float(r[0]))"
    out = to_standalone(code, {})
    assert "import numpy as np" in out
    assert "def mp_shue1998" in out
    ns: dict = {}
    exec(out, ns)


def test_transform_helper_reemitted_with_geopack_note():
    from helioai.export import to_standalone

    out = to_standalone("v = transform_coords('2019-01-01', [1, 2, 3], 'gse', 'gsm')", {})
    assert "pip install geopack" in out
    assert "def _epoch_seconds" in out
    assert "def transform_coords" in out


# ── E6: the exported code must compute what the sandbox computed ─────────────


def _fake_speasy(values, fillval, *, columns=("Bx",), unit="nT"):
    """A stand-in `spz` whose get_data returns raw archive values with a declared FILLVAL."""
    from types import SimpleNamespace

    import numpy as np

    def get_data(param_id, start, stop=None):
        def _var(n):
            return SimpleNamespace(
                time=np.arange(n).astype("datetime64[s]"),
                values=np.array(values, dtype=float),
                meta={"FILLVAL": fillval},
                columns=list(columns),
                unit=unit,
            )

        if isinstance(start, list):
            return [_var(len(values)) for _ in start]
        return _var(len(values))

    return SimpleNamespace(get_data=get_data)


def test_exported_timeseries_blanks_declared_fill_like_the_sandbox_did(monkeypatch) -> None:
    """Audit probe: the same code gave mean 5.0 in the sandbox and 50002.45 once exported.

    get_timeseries blanks the declared FILLVAL before persisting, so the sandbox never
    sees 99999.9. A rewrite to a bare spz.get_data() hands the raw sentinel back to the
    very same arithmetic. Same source, different data, different number — silently.
    """
    import numpy as np

    from helioai.export import to_standalone

    manifest = {
        "datasets": {
            "density": {
                "kind": "timeseries",
                "param_id": "cda/EXAMPLE/Np",
                "start": "2005-01-17",
                "stop": "2005-01-18",
            }
        }
    }
    code = 'd = load_data("density")\nresult = float(np.nanmean(d.values))\n'
    out = to_standalone(code, manifest)
    monkeypatch.setitem(
        sys.modules, "speasy", _fake_speasy([5.0, 99999.8984375], [99999.8984375], unit="cm-3")
    )
    ns: dict = {}
    exec(out, ns)  # noqa: S102

    assert ns["result"] == pytest.approx(5.0)
    assert ns["d"].units == "cm-3", "load_data exposed .units; the standalone must too"
    assert ns["d"].param_id == "cda/EXAMPLE/Np"
    assert np.isnan(ns["d"].values[1])


def test_exported_event_collection_blanks_declared_fill(monkeypatch) -> None:
    from helioai.export import to_standalone

    manifest = {
        "datasets": {
            "b_events": {
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
                        "start": "2005-03-10T05:00",
                        "stop": "2005-03-10T06:00",
                        "status": "ok",
                    },
                ],
            }
        }
    }
    code = 'evs = load_data("b_events")\nmeans = [float(np.nanmean(e.values)) for e in evs]\n'
    out = to_standalone(code, manifest)
    monkeypatch.setitem(sys.modules, "speasy", _fake_speasy([2.0, -1e31, 4.0], -1e31))
    ns: dict = {}
    exec(out, ns)  # noqa: S102

    assert ns["means"] == [pytest.approx(3.0), pytest.approx(3.0)]
    assert ns["evs"][0].start == "2005-01-17T01:00"
    assert ns["evs"][1].units == "nT"


def test_exported_helpers_accept_what_the_sandbox_helpers_accept() -> None:
    """Audit probe: export(units=), export(dict), magnitude() and interp_to() all ran in
    the sandbox and all failed once exported — TypeError, TypeError, NameError, NameError.
    """
    import numpy as np

    from helioai.export import to_standalone

    code = (
        "b = np.array([[3., 4., 0.], [np.nan, np.nan, np.nan]])\n"
        "mag = magnitude(b)\n"
        "t1 = np.array(['2015-03-17T00:00:00', '2015-03-17T00:02:00'], dtype='datetime64[s]')\n"
        "t2 = np.array(['2015-03-17T00:01:00'], dtype='datetime64[s]')\n"
        "mid = interp_to(t2, t1, np.array([10., 30.]))\n"
        'export("field", [5.0], units="nT")\n'
        'export("summary", {"ratio": 2.5, "note": "text"})\n'
    )
    out = to_standalone(code, {"datasets": {}})
    ns: dict = {}
    exec(out, ns)  # noqa: S102

    assert ns["mag"][0] == pytest.approx(5.0)
    assert np.isnan(ns["mag"][1]), "a three-component gap stays a gap, never 0 nT"
    assert ns["mid"][0] == pytest.approx(20.0)


def test_strip_known_imports_keeps_names_the_header_does_not_provide() -> None:
    """Audit probe: `from numpy import mean` was stripped because its root is numpy,
    but the header only binds `np` — the cell then died on NameError.
    """
    from helioai.export import to_standalone

    code = "from numpy import mean\nimport numpy\nresult = mean([1., 3.]) + numpy.float64(0)\n"
    out = to_standalone(code, {"datasets": {}})
    ns: dict = {}
    exec(out, ns)  # noqa: S102
    assert ns["result"] == pytest.approx(2.0)

    redundant = "import numpy as np\nfrom scipy import signal\nx = np.arange(2); signal.welch\n"
    out = to_standalone(redundant, {"datasets": {}})
    assert out.count("import numpy as np") == 1
    assert out.count("from scipy import signal") == 1


def test_failed_attempts_are_not_executable_cells(monkeypatch, tmp_path) -> None:
    """Audit probe: code_N.py is written before it runs, so a failed attempt followed by
    its fix exported as two executable cells — and "Run All" stopped on the first.
    """
    from helioai.export import build_notebook

    store = SessionStore(tmp_path / "sessions.db")
    workspace = tmp_path / "users" / _USER / "workspace" / _LABEL
    workspace.mkdir(parents=True)
    (workspace / "code_0.py").write_text('raise ValueError("first attempt")\n', encoding="utf-8")
    (workspace / "code_1.py").write_text("result = 2 + 2\n", encoding="utf-8")

    history = [
        Message(role="user", content="compute"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="r0", name="run_python", arguments={"code": "..."})],
        ),
        Message(
            role="tool",
            tool_call_id="r0",
            content=json.dumps(
                {"error": "ValueError: first attempt", "code_path": str(workspace / "code_0.py")}
            ),
        ),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="r1", name="run_python", arguments={"code": "..."})],
        ),
        Message(
            role="tool",
            tool_call_id="r1",
            content=json.dumps(
                {"stdout": "", "exports": {}, "code_path": str(workspace / "code_1.py")}
            ),
        ),
        Message(role="assistant", content="4"),
    ]
    store.save(_USER, _SESSION, history)
    store.set_workspace_dir(_USER, _SESSION, _LABEL)
    monkeypatch.setattr(export_module, "store", store)
    monkeypatch.setattr(export_module.settings, "data_dir", tmp_path)

    nb = build_notebook(_USER, _SESSION)
    code_cells = [c.source for c in nb.cells if c.cell_type == "code"]
    markdown = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")

    assert not any("first attempt" in s for s in code_cells), "a failed run must not block Run All"
    assert any("result = 2 + 2" in s for s in code_cells)
    assert "first attempt" in markdown, "the failed attempt stays in the record, as prose"
    ns: dict = {}
    for src in code_cells:
        exec(src, ns)  # noqa: S102
    assert ns["result"] == 4


def test_methods_section_sees_recipes_loaded_by_a_sub_agent(monkeypatch, tmp_path) -> None:
    """Audit probe: a recipe loaded inside a data_analyst run reaches the lead only as a
    `recipe_used` artifact in the task result — and the export ignored those.
    """
    import json

    from helioai.export import _collect_recipes

    history = [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="task-1", name="task", arguments={"agent_role": "data_analyst"})
            ],
        ),
        Message(
            role="tool",
            tool_call_id="task-1",
            content=json.dumps(
                {
                    "summary": "done",
                    "artifacts": [
                        {
                            "tool": "load_recipe",
                            "kind": "recipe_used",
                            "name": "theta_bn",
                            "reference": "Schwartz (1998), ISSI SR-001",
                            "description": "Shock normal angle by coplanarity.",
                        }
                    ],
                }
            ),
        ),
    ]
    found = _collect_recipes(history)
    assert [r["name"] for r in found] == ["theta_bn"]
    assert found[0]["reference"] == "Schwartz (1998), ISSI SR-001"


def test_narrative_does_not_credit_the_reader_with_an_automated_correction(wired, monkeypatch):
    """The correction the loop injects rides in a `user` message (the only role the
    providers forward from history). In the exported conversation it must read as
    HelioAI's note, not as something the researcher typed."""
    from helioai import export as export_module

    history = export_module.store.get_or_create(_USER, _SESSION)
    history.append(Message(role="assistant", content="Use cda/BOGUS/id."))
    history.append(
        Message(
            role="user",
            content="⚠️ AUTOMATED CORRECTION — not in the catalogue",
            origin="correction",
        )
    )
    history.append(Message(role="assistant", content="Use the real id."))
    export_module.store.save(_USER, _SESSION, history)

    nb = nbformat.read(str(export_session_notebook(_USER, _SESSION)), as_version=4)
    narrative = next(c.source for c in nb.cells if c.source.startswith("## Conversation"))
    assert "**You:** ⚠️ AUTOMATED CORRECTION" not in narrative
    assert "_Automated note (correction):_ ⚠️ AUTOMATED CORRECTION" in narrative
    assert "**You:** Plot IMF Bz from ACE on 2005-01-17" in narrative
