"""Export a session as a reproducible Jupyter notebook.

A research result that cannot be re-run is worthless. The agent already saves
every sandbox run as `code_N.py` in the session workspace; this module bundles
those runs plus the conversation into a self-contained, re-executable `.ipynb`
with a provenance header (parameter ids, time, library versions).

The exported notebook opens with a setup cell that provides the same imports
and the `export()` / `clean()` / `param_card()` shims the sandbox defines, so
each saved run cell executes standalone in a normal Jupyter kernel.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from helioai.config import settings
from helioai.core.session import store
from helioai.datastore import read_manifest

_PARAM_ID_RE = re.compile(r"\b(?:amda|cda|csa|ssc)/[\w./-]+")
_LOAD_DATA_RE = re.compile(r"""load_data\(\s*(["'])([a-z0-9_]+)\1\s*\)""")

# Base setup cell — imports + the export()/clean()/param_card() shims that the
# saved run cells always rely on. The load_data shim is appended separately
# (see _LOAD_DATA_SHIM) only when some cell still references load_data after the
# standalone rewrite.
_SETUP_CELL_BASE = '''\
# HelioAI export — environment setup (mirrors the sandbox preamble)
import warnings
warnings.filterwarnings("ignore")
import json, types
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import speasy as spz
from scipy import signal, stats, fft
try:
    import plasmapy.formulary as pf
    import astropy.units as u
except ImportError:
    pf = None

_HELIOAI_DATA_DIR = Path("{data_dir}")


def export(name, data):
    """Shim for the sandbox export(): prints a numeric summary."""
    arr = np.asarray(data, dtype=float)
    print(f"{{name}}: shape={{arr.shape}} min={{np.nanmin(arr):.4g}} "
          f"max={{np.nanmax(arr):.4g}} mean={{np.nanmean(arr):.4g}}")


def clean(values):
    """Shim for the sandbox clean(): convert CDF fill values (|x|>=1e30) and infinities to NaN."""
    arr = np.asarray(values, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    arr[np.abs(arr) >= 1e30] = np.nan
    return arr


def param_card(var, param_id):
    """Shim for the sandbox param_card(): prints parameter metadata."""
    print(f"{{param_id}}: {{getattr(var, 'name', '')}} "
          f"[{{getattr(var, 'unit', '')}}] n_points={{len(var.time)}}")


def document_method(name, reference="", method=""):
    """Shim for the sandbox document_method(): prints the recorded method + reference."""
    print(f"method: {{name}}" + (f" — {{reference}}" if reference else ""))
'''

_LOAD_DATA_SHIM = '''

def load_data(name):
    """Load a dataset saved by get_timeseries / get_events_timeseries during the session."""
    mfile = _HELIOAI_DATA_DIR / "manifest.json"
    if not mfile.exists():
        raise FileNotFoundError(
            f"manifest not found at {mfile} — copy the session data/ folder next to this notebook"
        )
    manifest = json.loads(mfile.read_text())
    entry = manifest.get("datasets", {}).get(name)
    if entry is None:
        available = sorted(manifest.get("datasets", {}).keys())
        raise KeyError(f"unknown dataset {name!r} — available: {available}")
    z = np.load(_HELIOAI_DATA_DIR / entry["file"], allow_pickle=False)
    if entry["kind"] == "timeseries":
        ns = types.SimpleNamespace()
        ns.time = z["time"]
        ns.values = z["values"]
        ns.columns = entry.get("columns", [])
        ns.units = entry.get("units", "")
        ns.param_id = entry.get("param_id", "")
        return ns
    elif entry["kind"] == "event_collection":
        result = []
        for em in entry.get("events", []):
            if em.get("status") != "ok":
                continue
            i = em["idx"]
            ev = types.SimpleNamespace()
            ev.time = z[f"t{i}"]
            ev.values = z[f"v{i}"]
            ev.start = em["start"]
            ev.stop = em["stop"]
            ev.units = entry.get("units", "")
            result.append(ev)
        return result
    raise ValueError(f"unknown dataset kind {entry['kind']!r}")
'''


def _rewrite_load_data_calls(code_src: str, manifest: dict) -> str:
    """Rewrite load_data("name") into a standalone spz.get_data(...) call.

    Frontier rewrite: the stored sandbox code keeps load_data(); this produces the
    standalone version shown to the reader (/code panel + notebook export).

    - timeseries dataset with param_id/start/stop → spz.get_data("<id>", "<start>", "<stop>")
      (SpeasyVariable exposes .time/.values/.columns; .units/.param_id are not mirrored —
      ponytail: rare in generated code, nbconvert --execute catches any breakage)
    - event_collection with param_id → spz.get_data("<id>", [["<s>","<e>"], ...]) over the
      events that actually had data (speasy multi-interval call, same as get_events_timeseries)
    - unknown / unreconstructable dataset → left intact (never emit a wrong call)
    """
    datasets = manifest.get("datasets", {})
    kept_load_data = {"flag": False}

    def _sub(m: re.Match) -> str:
        name = m.group(2)
        entry = datasets.get(name)
        if not entry:
            kept_load_data["flag"] = True
            return m.group(0)
        if entry.get("kind") == "timeseries" and all(
            entry.get(k) for k in ("param_id", "start", "stop")
        ):
            return f'spz.get_data("{entry["param_id"]}", "{entry["start"]}", "{entry["stop"]}")'
        if entry.get("kind") == "event_collection" and entry.get("param_id"):
            intervals = [
                [ev["start"], ev["stop"]]
                for ev in entry.get("events", [])
                if ev.get("status") == "ok"
            ]
            if intervals:
                return f'spz.get_data("{entry["param_id"]}", {intervals!r})'
        kept_load_data["flag"] = True
        return m.group(0)

    rewritten = _LOAD_DATA_RE.sub(_sub, code_src)
    if kept_load_data["flag"]:
        rewritten = "# some datasets kept as load_data() — see data/manifest.json\n" + rewritten
    return rewritten


def _version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "unknown"


def _collect_param_ids(history) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for m in history:
        blobs = [m.content or ""]
        for tc in m.tool_calls or []:
            blobs.append(str(tc.arguments))
        for blob in blobs:
            for pid in _PARAM_ID_RE.findall(blob):
                if pid not in seen:
                    seen.add(pid)
                    found.append(pid)
    return found


def _collect_recipes(history) -> list[dict]:
    """Return the methods/recipes used during the session, for the Methods section.

    Two sources:
      - load_recipe tool calls → reference/description read from the recipe file header
      - document_method() cards in run_python results (methods computed outside a recipe)
    Each entry: {"name", "reference", "description"}. Order preserved, de-duplicated by name.
    """
    from helioai.tools.recipes import _parse_header

    recipes_dir = settings.recipes.recipes_dir
    found: list[dict] = []
    seen: set[str] = set()

    def _add(name: str, reference: str, description: str) -> None:
        if not name or name in seen:
            return
        seen.add(name)
        found.append({"name": name, "reference": reference, "description": description})

    for m in history:
        for tc in m.tool_calls or []:
            if tc.name != "load_recipe":
                continue
            args = tc.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (ValueError, TypeError):
                    args = {}
            name = (args or {}).get("name")
            reference = description = ""
            if name:
                path = recipes_dir / f"{name}.py"
                if path.is_file():
                    meta = _parse_header(path.read_text(encoding="utf-8"))
                    reference = meta.get("reference", "")
                    description = meta.get("description", "")
            _add(name, reference, description)

        if m.role == "tool" and m.content:
            try:
                data = json.loads(m.content)
            except (ValueError, TypeError):
                continue
            cards = data.get("cards", []) if isinstance(data, dict) else []
            for card in cards:
                if isinstance(card, dict) and card.get("kind") == "method_used":
                    _add(card.get("name", ""), card.get("reference", ""), card.get("method", ""))
    return found


def _code_files(workspace_dir: Path) -> list[Path]:
    """Return code_N.py files sorted by run index."""
    files = list(workspace_dir.glob("code_*.py"))

    def _idx(p: Path) -> int:
        parts = p.stem.split("_")
        return int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0

    return sorted(files, key=_idx)


def build_notebook(user_id: str, session_id: str):
    """Build an nbformat notebook object for a session (no file I/O)."""
    import nbformat as nbf

    history = store.get_or_create(user_id, session_id)
    label = store.get_workspace_dir(user_id, session_id)
    workspace_dir = Path(settings.workspace.workspace_dir) / label if label else None

    nb = nbf.v4.new_notebook()
    cells = []

    # Provenance
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    param_ids = _collect_param_ids(history)
    prov = [
        "# HelioAI session export",
        "",
        f"- **Generated:** {now}",
        f"- **Session:** `{session_id}`",
        f"- **speasy:** {_version('speasy')} · **plasmapy:** {_version('plasmapy')} "
        f"· **helioai:** {_version('helioai')}",
    ]
    if param_ids:
        prov.append("- **Parameters referenced:** " + ", ".join(f"`{p}`" for p in param_ids))
    prov.append("\n> Run all cells top-to-bottom to reproduce the analysis.")
    cells.append(nbf.v4.new_markdown_cell("\n".join(prov)))

    # Methods & data acknowledgements — recipes used + references
    recipes = _collect_recipes(history)
    if recipes or param_ids:
        methods = ["## Methods & data acknowledgements", ""]
        for r in recipes:
            line = f"- **{r['name']}**"
            if r["description"]:
                line += f" — {r['description']}"
            if r["reference"]:
                line += f"  \n  _Reference:_ {r['reference']}"
            methods.append(line)
        if param_ids:
            methods.append(
                "- **Data:** " + ", ".join(f"`{p}`" for p in param_ids) + " (via speasy)"
            )
        methods.append(
            f"\n_Libraries: speasy {_version('speasy')}, plasmapy {_version('plasmapy')}, "
            f"helioai {_version('helioai')}._"
        )
        cells.append(nbf.v4.new_markdown_cell("\n".join(methods)))

    # Standalone rewrite of every saved run (load_data → spz.get_data), so cells run
    # without the session data/ folder. Compute first to know if the load_data shim is
    # still needed in the setup cell.
    manifest = read_manifest(workspace_dir) if workspace_dir else {"datasets": {}}
    runs: list[tuple[str, str]] = []
    if workspace_dir and workspace_dir.exists():
        for p in _code_files(workspace_dir):
            src = _rewrite_load_data_calls(p.read_text(encoding="utf-8"), manifest)
            runs.append((p.name, src))
    shim_needed = any("load_data(" in src for _, src in runs)

    data_dir = (workspace_dir / "data") if workspace_dir else Path("data")
    setup_cell = _SETUP_CELL_BASE.format(data_dir=str(data_dir))
    if shim_needed:
        setup_cell += _LOAD_DATA_SHIM
    cells.append(nbf.v4.new_code_cell(setup_cell))

    # Conversation narrative
    convo: list[str] = ["## Conversation"]
    for m in history:
        text = (m.content or "").strip()
        if m.role == "user" and text:
            convo.append(f"**You:** {text}")
        elif m.role == "assistant" and text:
            convo.append(f"**HelioAI:** {text}")
    if len(convo) > 1:
        cells.append(nbf.v4.new_markdown_cell("\n\n".join(convo)))

    # Reproducible analysis: every saved run, in execution order
    if runs:
        cells.append(nbf.v4.new_markdown_cell("## Reproducible analysis"))
        for name, src in runs:
            cells.append(nbf.v4.new_markdown_cell(f"### {name}"))
            cells.append(nbf.v4.new_code_cell(src))

    nb["cells"] = cells
    return nb


def export_session_notebook(user_id: str, session_id: str, out_path: Path | None = None) -> Path:
    """Write the session as a .ipynb and return its path.

    Default location: <workspace>/<label>.ipynb (or <workspace>/<session>.ipynb).
    """
    import nbformat as nbf

    nb = build_notebook(user_id, session_id)

    if out_path is None:
        label = store.get_workspace_dir(user_id, session_id) or session_id
        root = Path(settings.workspace.workspace_dir)
        root.mkdir(parents=True, exist_ok=True)
        out_path = root / f"{label}.ipynb"
    out_path = Path(out_path)
    nbf.write(nb, str(out_path))
    return out_path
