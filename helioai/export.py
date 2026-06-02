"""Export a session as a reproducible Jupyter notebook.

A research result that cannot be re-run is worthless. The agent already saves
every sandbox run as `code_N.py` in the session workspace; this module bundles
those runs plus the conversation into a self-contained, re-executable `.ipynb`
with a provenance header (parameter ids, time, library versions).

The exported notebook opens with a setup cell that provides the same imports
and the `export()` / `param_card()` shims the sandbox defines, so each saved
run cell executes standalone in a normal Jupyter kernel.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from helioai.config import settings
from helioai.core.session import store

_PARAM_ID_RE = re.compile(r"\b(?:amda|cda|csa|ssc)/[\w./-]+")

_SETUP_CELL = '''\
# HelioAI export — environment setup (mirrors the sandbox preamble)
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import speasy as spz
from scipy import signal, stats, fft
try:
    import plasmapy.formulary as pf
    import astropy.units as u
except ImportError:
    pf = None


def export(name, data):
    """Shim for the sandbox export(): prints a numeric summary."""
    arr = np.asarray(data, dtype=float)
    print(f"{name}: shape={arr.shape} min={np.nanmin(arr):.4g} "
          f"max={np.nanmax(arr):.4g} mean={np.nanmean(arr):.4g}")


def param_card(var, param_id):
    """Shim for the sandbox param_card(): prints parameter metadata."""
    print(f"{param_id}: {getattr(var, 'name', '')} "
          f"[{getattr(var, 'unit', '')}] n_points={len(var.time)}")
'''


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

    cells.append(nbf.v4.new_code_cell(_SETUP_CELL))

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
    if workspace_dir and workspace_dir.exists():
        code_files = _code_files(workspace_dir)
        if code_files:
            cells.append(nbf.v4.new_markdown_cell("## Reproducible analysis"))
            for p in code_files:
                src = p.read_text(encoding="utf-8")
                cells.append(nbf.v4.new_markdown_cell(f"### {p.name}"))
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
