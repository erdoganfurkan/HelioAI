"""Export a session as a reproducible Jupyter notebook.

A research result that cannot be re-run is worthless. The agent already saves
every sandbox run as `code_N.py` in the session workspace; this module bundles
those runs plus the conversation into a self-contained, re-executable `.ipynb`
with a provenance header (parameter ids, time, library versions).

The saved runs are rewritten to standalone code (`to_standalone`): load_data()
becomes a direct `spz.get_data(...)`, the agent-only `param_card()`/
`document_method()` calls are dropped, and a minimal header supplies the
imports plus real `clean()`/`export()` helpers — so each cell runs in a plain
Jupyter kernel with no HelioAI sandbox around it.
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

# Real helpers emitted into standalone code (no HelioAI sandbox needed). These
# are plain strings — never .format()-ed — so their f-string braces stay intact.
_CLEAN_DEF = '''\
def clean(values):
    """Convert CDF fill values (|x|>=1e30) and infinities to NaN."""
    arr = np.asarray(values, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    arr[np.abs(arr) >= 1e30] = np.nan
    return arr'''

_EXPORT_DEF = '''\
def export(name, data):
    """Print a numeric summary of an array."""
    arr = np.asarray(data, dtype=float)
    print(f"{name}: shape={arr.shape} min={np.nanmin(arr):.4g} "
          f"max={np.nanmax(arr):.4g} mean={np.nanmean(arr):.4g}")'''

# Notebook setup cell — full imports + the real clean()/export() helpers. The
# load_data shim is appended separately only when a cell still references it.
_SETUP_CELL_BASE = (
    """\
# HelioAI export — environment setup
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


"""
    + _CLEAN_DEF
    + "\n\n\n"
    + _EXPORT_DEF
    + "\n"
)

# Appended to the setup cell only when some dataset could not be rewritten to a
# direct spz.get_data() call (the {data_dir} line is built separately, never
# .format()-ed here, so the f-string braces below survive).
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


def _match_paren(s: str, i: int) -> int | None:
    """Index of the ')' matching the '(' at s[i], skipping string literals."""
    depth = 0
    quote: str | None = None
    while i < len(s):
        c = s[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _drop_call_statements(code: str, name: str) -> str:
    """Remove `name(...)` statements (balanced parens, multi-line aware)."""
    pat = re.compile(rf"(?m)^[ \t]*{re.escape(name)}\s*\(")
    while True:
        m = pat.search(code)
        if not m:
            return code
        open_paren = code.index("(", m.start())
        end = _match_paren(code, open_paren)
        if end is None:
            return code  # malformed — leave as-is rather than loop forever
        nl = code.find("\n", end)
        nl = len(code) if nl == -1 else nl + 1
        code = code[: m.start()] + code[nl:]


def _strip_agent_only_calls(code: str) -> str:
    """Drop agent/UI-only calls that have no scientific effect in standalone code."""
    for fn in ("param_card", "document_method"):
        code = _drop_call_statements(code, fn)
    return code


def _standalone_header(code: str) -> str:
    """Imports + real clean()/export() helpers needed by `code`, conditionally."""
    needs_np = "np." in code or re.search(r"\b(clean|export)\s*\(", code)
    imports: list[str] = []
    if needs_np:
        imports.append("import numpy as np")
    if "plt." in code or "matplotlib" in code:
        imports.append("import matplotlib.pyplot as plt")
    if "spz." in code:
        imports.append("import speasy as spz")
    scipy_mods = [m for m in ("signal", "stats", "fft") if re.search(rf"\b{m}\.", code)]
    if scipy_mods:
        imports.append(f"from scipy import {', '.join(scipy_mods)}")
    if "pf." in code:
        imports.append("import plasmapy.formulary as pf")
    if re.search(r"\bu\.", code):
        imports.append("import astropy.units as u")

    # ponytail: drop any import the body already declares verbatim (sandbox code
    # ships its own `import numpy as np` etc.); exact-line match, good enough.
    imports = [imp for imp in imports if not re.search(rf"(?m)^{re.escape(imp)}\s*$", code)]

    blocks: list[str] = []
    if imports:
        blocks.append("\n".join(imports))
    if re.search(r"\bclean\s*\(", code):
        blocks.append(_CLEAN_DEF)
    if re.search(r"\bexport\s*\(", code):
        blocks.append(_EXPORT_DEF)
    return "\n\n\n".join(blocks)


def to_standalone(code_src: str, manifest: dict, *, with_header: bool = True) -> str:
    """Turn a saved sandbox run into standalone, re-executable code.

    Strips agent-only calls, rewrites load_data() → spz.get_data(), and (unless
    embedded in a notebook that already has a setup cell) prepends the imports
    plus real clean()/export() helpers it needs.
    """
    body = _rewrite_load_data_calls(_strip_agent_only_calls(code_src), manifest)
    if not with_header:
        return body
    header = _standalone_header(body)
    return f"{header}\n\n\n{body}" if header else body


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

    from helioai.workspace import user_home

    history = store.get_or_create(user_id, session_id)
    label = store.get_workspace_dir(user_id, session_id)
    workspace_dir = user_home(user_id) / "workspace" / label if label else None

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
            src = to_standalone(p.read_text(encoding="utf-8"), manifest, with_header=False)
            runs.append((p.name, src))
    shim_needed = any("load_data(" in src for _, src in runs)

    data_dir = (workspace_dir / "data") if workspace_dir else Path("data")
    setup_cell = _SETUP_CELL_BASE
    if shim_needed:
        setup_cell += (
            f"\n\nimport json, types\nfrom pathlib import Path\n"
            f"_HELIOAI_DATA_DIR = Path({str(data_dir)!r})\n" + _LOAD_DATA_SHIM
        )
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
        from helioai.workspace import user_home

        label = store.get_workspace_dir(user_id, session_id) or session_id
        root = user_home(user_id) / "workspace"
        root.mkdir(parents=True, exist_ok=True)
        out_path = root / f"{label}.ipynb"
    out_path = Path(out_path)
    nbf.write(nb, str(out_path))
    return out_path
