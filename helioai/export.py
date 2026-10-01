"""Export a session as a reproducible Jupyter notebook.

A research result that cannot be re-run is worthless. The agent already saves
every sandbox run as `code_N.py` in the session workspace; this module bundles
those runs plus the conversation into a self-contained, re-executable `.ipynb`
with a provenance header (parameter ids, time, library versions).

The saved runs are rewritten to standalone code (`to_standalone`): load_data()
becomes a `fetch_series(...)` call that wraps `spz.get_data` and blanks the declared
fill value exactly as the session did, the agent-only `param_card()`/
`document_method()` calls are dropped, and a minimal header supplies the imports
plus real `clean()`/`export()`/`magnitude()`/`interp_to()` helpers — so each cell
runs in a plain Jupyter kernel with no HelioAI sandbox around it.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import NamedTuple

from helioai.config import settings
from helioai.core.session import store
from helioai.datastore import read_manifest

_PARAM_ID_RE = re.compile(r"\b(?:amda|cda|csa|ssc)/[\w./-]+")
_LOAD_DATA_RE = re.compile(r"""load_data\(\s*(["'])([a-z0-9_]+)\1\s*\)""")

# Real helpers emitted into standalone code (no HelioAI sandbox needed). These
# are plain strings — never .format()-ed — so their f-string braces stay intact.
#
# clean/export/magnitude/interp_to mirror the sandbox preamble in tools/sandbox.py.
# They are duplicated rather than imported because the exported notebook must run
# with no HelioAI installed; tests/test_export.py holds them to the same behaviour.
_CLEAN_DEF = '''\
def clean(values):
    """Convert CDF fill values (|x|>=1e30) and infinities to NaN."""
    arr = np.asarray(values, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    arr[np.abs(arr) >= 1e30] = np.nan
    return arr'''

_EXPORT_DEF = '''\
def export(name, data, units=""):
    """Print a numeric summary of an array; a dict of numbers is summarised key by key.
    A value that is not numeric is printed as it is — the session's export() kept it
    rather than raising, and the notebook must not fail where the run did not."""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, int, float)) and not isinstance(value, bool):
                export(f"{name}.{key}", value, units)
        return
    try:
        arr = np.asarray(data, dtype=float)
    except (TypeError, ValueError):
        print(f"{name}: {data!r}")
        return
    unit = f" {units}" if units else ""
    print(f"{name}: shape={arr.shape} min={np.nanmin(arr):.4g}{unit} "
          f"max={np.nanmax(arr):.4g}{unit} mean={np.nanmean(arr):.4g}{unit}")'''

_MAGNITUDE_DEF = '''\
def magnitude(vectors):
    """|V| of an N×3 array where a data gap stays NaN (nansum would read it as 0)."""
    arr = clean(vectors)
    if arr.ndim not in (1, 2) or arr.shape[-1] != 3:
        raise ValueError(
            f"magnitude expects an (N, 3) array of vector components, got shape {arr.shape}: "
            "pass the three components (for instance b[:, :3]); a spectrogram or a scalar "
            "series has no vector magnitude"
        )
    return np.sqrt(np.sum(arr**2, axis=-1))'''

_INTERP_TO_DEF = '''\
def interp_to(t_target, t_source, values):
    """Resample onto t_target; a point that would lean on a missing sample stays NaN."""
    tt = np.asarray(t_target)
    ts = np.asarray(t_source)
    if np.issubdtype(tt.dtype, np.datetime64):
        tt = tt.astype("datetime64[ns]").astype("float64")
    if np.issubdtype(ts.dtype, np.datetime64):
        ts = ts.astype("datetime64[ns]").astype("float64")
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        good = np.isfinite(v)
        if not good.any():
            return np.full(tt.shape, np.nan)
        out = np.interp(tt, ts[good], v[good], left=np.nan, right=np.nan)
        touched = np.interp(tt, ts, (~good).astype(float), left=1.0, right=1.0)
        out[touched > 0.0] = np.nan
        return out
    cols = [interp_to(t_target, t_source, v[:, i]) for i in range(v.shape[1])]
    return np.column_stack(cols)'''

# What load_data() gave the sandbox was the archive series with the declared FILLVAL
# already blanked to NaN (get_timeseries does that once, before persisting). A rewrite
# to a bare spz.get_data() hands the raw sentinel to the very same arithmetic: the same
# code gave a mean of 5.0 in the sandbox and 50002.4 exported. These wrappers redo the
# blanking with the same three conventions as datastore.fill_mask, and mirror the
# attributes load_data exposed (.units, .param_id) that SpeasyVariable does not.
_FETCH_DEFS = '''\
def _blank_declared_fill(var):
    values = np.array(var.values, dtype=float)
    bad = ~np.isfinite(values) | (np.abs(values) >= 1e30)
    fillval = (getattr(var, "meta", {}) or {}).get("FILLVAL")
    if fillval is not None:
        try:
            fv = float(np.asarray(fillval).ravel()[0])
            if np.isfinite(fv):
                bad |= np.isclose(values, fv, rtol=1e-6)
        except (TypeError, ValueError, IndexError):
            pass
    values[bad] = np.nan
    return values, float(bad.mean() * 100) if bad.size else 0.0


def fetch_series(param_id, start, stop):
    """spz.get_data() with the declared FILLVAL blanked to NaN, as the session saw it."""
    var = spz.get_data(param_id, start, stop)
    if var is None:
        raise ValueError(f"no data returned for {param_id!r} between {start} and {stop}")
    values, missing_pct = _blank_declared_fill(var)
    return types.SimpleNamespace(
        time=np.asarray(var.time),
        values=values,
        columns=list(getattr(var, "columns", None) or []),
        units=str(getattr(var, "unit", "") or ""),
        param_id=param_id,
        missing_pct=missing_pct,
    )


def fetch_events(param_id, intervals):
    """One fill-blanked series per [start, stop] interval, skipping empty events."""
    events = []
    for (start, stop), var in zip(intervals, spz.get_data(param_id, intervals) or []):
        if var is None or len(getattr(var, "time", [])) == 0:
            continue
        values, _ = _blank_declared_fill(var)
        events.append(
            types.SimpleNamespace(
                time=np.asarray(var.time),
                values=values,
                start=start,
                stop=stop,
                units=str(getattr(var, "unit", "") or ""),
            )
        )
    return events'''

_HELPER_DEFS = "\n\n\n".join((_CLEAN_DEF, _EXPORT_DEF, _MAGNITUDE_DEF, _INTERP_TO_DEF, _FETCH_DEFS))

# Notebook setup cell — full imports + the real helpers. The load_data shim is
# appended separately only when a cell still references it.
_SETUP_CELL_BASE = (
    """\
# HelioAI export — environment setup
import types
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
    + _HELPER_DEFS
    + "\n"
)

# Appended to the setup cell when a saved run was `run_recipe`. The recipe's source sits
# once in the notebook, in a collapsed cell before its first use; each run keeps only the
# lines that were its own — the bindings and the call — and hands them to this helper,
# which does what the sandbox did: run the source on a namespace where the inputs are
# bound and `__name__` is not `__main__`.
_RUN_RECIPE_DEF = '''

RECIPE_SOURCES = {}


def run_recipe(name, inputs=None, call=None):
    """Run a shipped recipe as the session ran it: its source, RECIPE_SOURCES[name],
    executed on a copy of this namespace with the inputs bound and __name__ set so a
    demo behind `if __name__ == "__main__":` stays off. `call` is the expression the
    session evaluated after the recipe, if any."""
    ns = dict(globals())
    ns.update(inputs or {})
    ns["__name__"] = "recipe"
    exec(compile(RECIPE_SOURCES[name], f"<recipe {name}>", "exec"), ns)
    if call:
        result = eval(call, ns)
        print(repr(result))
        return result
    return ns'''

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
    manifest = json.loads(mfile.read_text(encoding="utf-8"))
    datasets = manifest.get("datasets", {})
    key = name
    if key not in datasets:
        # The manifest keys a dataset by the slug of its id's last component, but the
        # session (and this notebook's prose) names the full product id. Resolve through
        # the recorded param_id: matching on the suffix alone would hand back another
        # mission's data whenever two products share a parameter name.
        norm = lambda s: re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
        if "/" in name:
            matches = [k for k, e in datasets.items() if e.get("param_id") == name]
        else:
            matches = [
                k for k, e in datasets.items()
                if norm(str(e.get("param_id", "")).rstrip("/").split("/")[-1]) == norm(name)
            ]
        if len(matches) == 1:
            key = matches[0]
        elif matches:
            windows = ", ".join(
                f"{k} ({datasets[k].get('start')} -> {datasets[k].get('stop')})"
                for k in sorted(matches)
            )
            raise KeyError(
                f"ambiguous dataset {name!r}: {windows} -- load one of these names explicitly"
            )
    entry = datasets.get(key)
    if entry is None:
        available = sorted(datasets.keys())
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
    """Rewrite load_data("name") into a standalone fetch_series(...)/fetch_events(...) call.

    Frontier rewrite: the stored sandbox code keeps load_data(); this produces the
    standalone version shown to the reader (/code panel + notebook export).

    - timeseries dataset with param_id/start/stop → fetch_series("<id>", "<start>", "<stop>")
    - event_collection with param_id → fetch_events("<id>", [["<s>","<e>"], ...]) over the
      events that actually had data (same multi-interval call as get_events_timeseries)
    - unknown / unreconstructable dataset → left intact (never emit a wrong call)

    The targets wrap spz.get_data() rather than being it: the sandbox read data with the
    declared FILLVAL already blanked, and a bare get_data() would feed the raw sentinel
    to the same arithmetic (see `_FETCH_DEFS`).
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
            return f'fetch_series("{entry["param_id"]}", "{entry["start"]}", "{entry["stop"]}")'
        if entry.get("kind") == "event_collection" and entry.get("param_id"):
            intervals = [
                [ev["start"], ev["stop"]]
                for ev in entry.get("events", [])
                if ev.get("status") == "ok"
            ]
            if intervals:
                return f'fetch_events("{entry["param_id"]}", {intervals!r})'
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


_KNOWN_IMPORT_ROOTS = {"numpy", "matplotlib", "speasy", "scipy", "plasmapy", "astropy"}

# The names the header / setup cell actually binds. An import line is dropped only when
# every name it introduces is one of these: `import numpy as np` is redundant with the
# header, `from numpy import mean` is not — stripping it left the cell on a NameError.
_HEADER_BINDINGS = {"np", "plt", "spz", "signal", "stats", "fft", "pf", "u"}


def _import_roots(stmt: str) -> set[str]:
    if stmt.startswith("from "):
        return {stmt.split()[1].split(".")[0]}
    roots = set()
    for part in stmt[len("import ") :].split(","):
        name = part.strip().split(" as ")[0].strip().split(".")[0]
        if name:
            roots.add(name)
    return roots


def _import_bindings(stmt: str) -> set[str]:
    """Names an import statement introduces into the namespace."""
    if stmt.startswith("from "):
        _, _, names = stmt.partition(" import ")
    else:
        names = stmt[len("import ") :]
    bound = set()
    for part in names.split(","):
        part = part.strip().strip("()")
        if not part:
            continue
        if " as " in part:
            bound.add(part.split(" as ")[1].strip())
        else:
            bound.add(part.split(".")[0])
    return bound


def _strip_known_imports(code: str) -> str:
    """Drop top-level import lines the header already provides, binding for binding.

    Handles combined forms (`import numpy as np, matplotlib.pyplot as plt`) the
    header would otherwise duplicate. The header is the single source of imports,
    but only for the names it actually defines.
    """
    out = []
    for line in code.splitlines():
        s = line.strip()
        if line[:1] not in (" ", "\t") and (s.startswith("import ") or s.startswith("from ")):
            roots = _import_roots(s)
            bindings = _import_bindings(s)
            if roots and roots <= _KNOWN_IMPORT_ROOTS and bindings <= _HEADER_BINDINGS:
                continue
        out.append(line)
    return "\n".join(out)


class _Uses(NamedTuple):
    """What a cell reaches for: attribute roots it does not bind itself, and names it calls."""

    roots: frozenset[str]
    calls: frozenset[str]


def _uses(code: str) -> _Uses:
    """Read the names a cell uses off its syntax tree, not off its text.

    `"np." in code` and `\\bu\\.` saw comments and string literals — a `# convert with u.nT`
    pulled astropy into a notebook that never used it — and a local variable named `stats`
    pulled scipy. The tree ignores comments and strings, and a name the cell binds itself
    (assignment, loop target, def, import) is not a module it needs. Code that does not
    parse falls back to the textual reading: an unparsable cell is already broken, and a
    superfluous import is the least of its problems.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        roots = frozenset(re.findall(r"\b([A-Za-z_]\w*)\.", code))
        calls = frozenset(re.findall(r"\b([A-Za-z_]\w*)\s*\(", code))
        return _Uses(roots, calls)
    bound: set[str] = set()
    roots: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            roots.add(node.value.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.add(node.func.id)
    return _Uses(frozenset(roots - bound), frozenset(calls))


def _standalone_header(code: str) -> str:
    """Imports + real helpers needed by `code`, conditionally."""
    uses = _uses(code)
    fetches = bool(uses.calls & {"fetch_series", "fetch_events"})
    helpers = {
        "clean",
        "export",
        "magnitude",
        "interp_to",
        "transform_coords",
        "mp_shue1998",
        "bs_jelinek2012",
    }
    imports: list[str] = []
    if "np" in uses.roots or uses.calls & helpers or fetches:
        imports.append("import numpy as np")
    if uses.roots & {"plt", "matplotlib"}:
        imports.append("import matplotlib.pyplot as plt")
    if "spz" in uses.roots or fetches:
        imports.append("import speasy as spz")
    if fetches:
        imports.append("import types")
    scipy_mods = [m for m in ("signal", "stats", "fft") if m in uses.roots]
    if scipy_mods:
        imports.append(f"from scipy import {', '.join(scipy_mods)}")
    if "pf" in uses.roots:
        imports.append("import plasmapy.formulary as pf")
    if "u" in uses.roots:
        imports.append("import astropy.units as u")

    blocks: list[str] = []
    if imports:
        blocks.append("\n".join(imports))
    if uses.calls & {"clean", "magnitude"}:
        blocks.append(_CLEAN_DEF)
    if "export" in uses.calls:
        blocks.append(_EXPORT_DEF)
    if "magnitude" in uses.calls:
        blocks.append(_MAGNITUDE_DEF)
    if "interp_to" in uses.calls:
        blocks.append(_INTERP_TO_DEF)
    if fetches:
        blocks.append(_FETCH_DEFS)
    blocks.extend(_physics_helper_defs(code, uses))
    return "\n\n\n".join(blocks)


def _physics_helper_defs(code: str, uses: _Uses | None = None) -> list[str]:
    """Source of the sandbox physics helpers the code calls, for standalone reuse."""
    calls = (uses or _uses(code)).calls
    used = [h for h in ("transform_coords", "mp_shue1998", "bs_jelinek2012") if h in calls]
    if not used:
        return []
    import inspect

    from helioai.tools import sandbox_helpers

    blocks: list[str] = []
    if "transform_coords" in used:
        blocks.append(
            "# requires: pip install geopack\n" + inspect.getsource(sandbox_helpers._epoch_seconds)
        )
    blocks.extend(inspect.getsource(getattr(sandbox_helpers, h)) for h in used)
    return blocks


def to_standalone(code_src: str, manifest: dict, *, with_header: bool = True) -> str:
    """Turn a saved sandbox run into standalone, re-executable code.

    Strips agent-only calls, rewrites load_data() → fetch_series()/fetch_events(), and
    (unless embedded in a notebook that already has a setup cell) prepends the imports
    plus the real helpers it needs.

    Args:
        code_src: A `code_N.py` saved by the sandbox.
        manifest: The session manifest from `read_manifest()` — provides the
            param_id and window behind each `load_data()` name.
        with_header: Prepend the standalone imports/helpers header.

    Example:
        >>> manifest = {"datasets": {"imf_gsm": {"kind": "timeseries",
        ...     "param_id": "amda/imf_gsm", "start": "2005-01-17", "stop": "2005-01-18"}}}
        >>> to_standalone('d = load_data("imf_gsm")\\nparam_card(d, "amda/imf_gsm")\\n',
        ...               manifest, with_header=False)
        'd = fetch_series("amda/imf_gsm", "2005-01-17", "2005-01-18")'
    """
    body = _strip_known_imports(
        _rewrite_load_data_calls(_strip_agent_only_calls(code_src), manifest)
    )
    if not with_header:
        return body
    header = _standalone_header(body)
    return f"{header}\n\n\n{body}" if header else body


def _version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "unknown"


def _helioai_version() -> str:
    """Our own version, read from the package rather than the distribution.

    `version("helioai")` looked up the *distribution* name, which is `helioai-agent`
    on PyPI — the import name and the distribution name are allowed to differ, and
    here they must. The lookup degraded to "unknown" instead of failing, so the miss
    would have been silent, in the provenance header of every exported notebook.
    """
    from helioai import __version__

    return __version__


_CATALOG_CITATIONS = {
    "helio4cast/icmecat": (
        "HELIO4CAST ICMECAT v2.3 (https://helioforecast.space/icmecat) — "
        "Moestl et al. (2017), Space Weather 15, doi:10.1002/2017SW001614"
    ),
}


def _collect_catalog_refs(history) -> list[str]:
    """Citations of community catalogs referenced anywhere in the session."""
    blobs: list[str] = []
    for m in history:
        blobs.append(m.content or "")
        for tc in m.tool_calls or []:
            blobs.append(str(tc.arguments))
    joined = "\n".join(blobs)
    return [cite for cat_id, cite in _CATALOG_CITATIONS.items() if cat_id in joined]


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

    Three sources:
      - load_recipe and run_recipe tool calls → reference/description read from the
        recipe file header
      - document_method() cards in run_python results (methods computed outside a recipe)
      - `recipe_used` artifacts carried back in a sub-agent's task result — a recipe the
        data_analyst loaded never appears as a lead tool call, only there
    Each entry: {"name", "reference", "description"}. Order preserved, de-duplicated by name.
    """
    from helioai.tools.recipes import _parse_header

    recipes_dir = settings.recipes.recipes_dir
    found: list[dict] = []
    seen: set[str] = set()

    def _header(name: str) -> tuple[str, str]:
        path = recipes_dir / f"{name}.py"
        if not path.is_file():
            return "", ""
        meta = _parse_header(path.read_text(encoding="utf-8"))
        return meta.get("reference", ""), meta.get("description", "")

    def _add(name: str, reference: str, description: str) -> None:
        if not name or name in seen:
            return
        seen.add(name)
        found.append({"name": name, "reference": reference, "description": description})

    for m in history:
        for tc in m.tool_calls or []:
            if tc.name not in ("load_recipe", "run_recipe"):
                continue
            args = tc.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (ValueError, TypeError):
                    args = {}
            name = (args or {}).get("name")
            reference, description = _header(name) if name else ("", "")
            _add(name, reference, description)

        if m.role == "tool" and m.content:
            try:
                data = json.loads(m.content)
            except (ValueError, TypeError):
                continue
            if not isinstance(data, dict):
                continue
            for card in data.get("cards", []):
                if isinstance(card, dict) and card.get("kind") == "method_used":
                    _add(card.get("name", ""), card.get("reference", ""), card.get("method", ""))
            for art in data.get("artifacts", []):
                if isinstance(art, dict) and art.get("kind") == "recipe_used":
                    name = art.get("name", "")
                    reference, description = art.get("reference", ""), art.get("description", "")
                    if name and not (reference and description):
                        h_ref, h_desc = _header(name)
                        reference, description = reference or h_ref, description or h_desc
                    _add(name, reference, description)
    return found


def _failed_code_names(history) -> set[str]:
    """File names of the sandbox runs the session itself reported as failed.

    `code_N.py` is written before it runs, so the workspace holds every attempt, the ones
    that raised included. The only record of which did is in the history: a run_python
    result carrying `error` and `code_path`, or — for a sub-agent's runs, invisible to the
    lead as tool calls — a `failed` code artifact inside the task result.
    """
    failed: set[str] = set()
    for m in history:
        if m.role != "tool" or not m.content:
            continue
        try:
            data = json.loads(m.content)
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("error") and data.get("code_path"):
            failed.add(Path(data["code_path"]).name)
        for art in data.get("artifacts", []):
            if isinstance(art, dict) and art.get("kind") == "code" and art.get("failed"):
                if art.get("code_path"):
                    failed.add(Path(art["code_path"]).name)
    return failed


def _code_files(workspace_dir: Path) -> list[Path]:
    """Return code_N.py files sorted by run index."""
    files = list(workspace_dir.glob("code_*.py"))

    def _idx(p: Path) -> int:
        parts = p.stem.split("_")
        return int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0

    return sorted(files, key=_idx)


_RECIPE_RUN_HEADER = re.compile(r"^# run_recipe: (?P<name>[A-Za-z_]\w*) — ")
_RECIPE_MARK = "# ── recipe {name} ──"
_CALL_MARK = "# ── call ──"
_BINDING_RE = re.compile(r"^(?P<name>[A-Za-z_]\w*) = \(", re.M)


class _RecipeRun(NamedTuple):
    name: str
    bindings: str
    source: str
    call: str | None
    preamble: str = ""


def _split_recipe_run(src: str) -> _RecipeRun | None:
    """Take a saved `run_recipe` script apart along the markers `recipe_script` wrote.

    A `run_recipe` run is five lines of the model's own — the input bindings and an
    optional call — around a recipe of two to six hundred lines inserted verbatim; a
    session that ran `superposed_epoch` five times exported the same 385 lines five
    times, and a notebook whose analysis was 30 lines a cell became 1 900 lines of
    repeated recipe. The exported notebook keeps each recipe once (see `build_notebook`)
    and each run keeps its own lines. Anything not laid out exactly as `recipe_script`
    lays it out — a hand-edited file, an older format — returns None and is exported as
    it is.

    Args:
        src: The run's code, raw or after `to_standalone`; the markers survive both, and
            the note `to_standalone` may put above the script is kept as a preamble.

    Returns:
        The recipe name, the binding lines, the recipe source, the call expression
        (None when the run had none) and the preamble, or None when this is not a
        `run_recipe` run.
    """
    lines = src.split("\n")
    start = next((i for i, ln in enumerate(lines) if _RECIPE_RUN_HEADER.match(ln)), None)
    if start is None or any(not ln.startswith("#") for ln in lines[:start] if ln.strip()):
        return None
    m = _RECIPE_RUN_HEADER.match(lines[start])
    assert m is not None
    name = m.group("name")
    preamble = "\n".join(ln for ln in lines[:start] if ln.strip())
    src = "\n".join(lines[start:])
    head, sep, rest = src.partition(f"\n{_RECIPE_MARK.format(name=name)}\n")
    if not sep:
        return None
    source, sep, tail = rest.partition(f"\n{_CALL_MARK}\n")
    call = None
    if sep:
        lines = [ln for ln in tail.splitlines() if ln.strip()]
        opener = "_recipe_result = ("
        if not lines or not lines[0].startswith(opener) or not lines[0].endswith(")"):
            return None
        call = lines[0][len(opener) : -1]
    bindings = [
        ln
        for ln in head.splitlines()[1:]
        if ln.strip() and not ln.startswith("#") and ln != '__name__ = "recipe"'
    ]
    return _RecipeRun(name, "\n".join(bindings), source.strip("\n"), call, preamble)


def _recipe_source_cells(run: _RecipeRun, raw_source: str | None):
    """The two cells that carry a recipe once: a line of provenance, then the source in a
    collapsed cell (`jupyter.source_hidden`, tag `recipe-source`) assigned into
    `RECIPE_SOURCES` for `run_recipe` to execute.

    The provenance line says whether what ran is the recipe shipped with the helioai
    installed at export time: the digest of the recipe text as it ran (the raw run file,
    before the notebook's standalone rewrites) against the digest of the shipped file.
    """
    import nbformat as nbf

    shipped = settings.recipes.recipes_dir / f"{run.name}.py"
    ran = hashlib.sha256((raw_source or run.source).rstrip("\n").encode()).hexdigest()
    if shipped.exists():
        now = hashlib.sha256(shipped.read_text(encoding="utf-8").rstrip("\n").encode()).hexdigest()
        same = "identical to" if now == ran else "**different from**"
        provenance = (
            f"{same} the `{run.name}` shipped with helioai {_helioai_version()} "
            f"(sha256 of the text as it ran: `{ran[:16]}…`)"
        )
    else:
        provenance = (
            f"sha256 of the text as it ran: `{ran[:16]}…`; no `{run.name}.py` is shipped "
            f"with helioai {_helioai_version()}"
        )
    n_lines = run.source.count("\n") + 1
    md = (
        f"#### Recipe `{run.name}` — as it ran, once for every use below\n\n"
        f"The source is in the collapsed cell that follows ({n_lines} lines), {provenance}. "
        "Imports the setup cell already makes and sandbox-only calls are removed, as in every "
        "cell of this notebook; nothing else is changed. Expand the cell to read it."
    )
    quote = "'" * 3
    body = run.source
    if quote not in body and not body.endswith("\\"):
        literal = f"r{quote}\n{body}\n{quote}"
    else:
        literal = json.dumps(body)
    code = f"RECIPE_SOURCES[{run.name!r}] = {literal}\n"
    metadata = {"jupyter": {"source_hidden": True}, "tags": ["recipe-source"]}
    return [nbf.v4.new_markdown_cell(md), nbf.v4.new_code_cell(code, metadata=metadata)]


def _recipe_call_cell(run: _RecipeRun, where: str = "collapsed cell above") -> str:
    """A `run_recipe` run as a cell of its own lines: the bindings as they ran, then the
    call to the helper with those names and the session's call expression."""
    names = _BINDING_RE.findall(run.bindings)
    inputs = "{" + ", ".join(f"{n!r}: {n}" for n in names) + "}"
    call = f", call={run.call!r}" if run.call else ""
    lines = [f"# run_recipe: {run.name} — the inputs, then the recipe as shipped ({where})"]
    if run.preamble:
        lines.insert(0, run.preamble)
    if run.bindings:
        lines.append(run.bindings)
    lines.append(f"run_recipe({run.name!r}, inputs={inputs}{call})")
    return "\n".join(lines) + "\n"


def _digest(text: str) -> str:
    return hashlib.sha256(text.rstrip("\n").encode()).hexdigest()


def recipe_run_view(code_src: str, manifest: dict) -> str | None:
    """A `run_recipe` script as the code panel shows it: the run's own lines, runnable,
    with the recipe read from the installed helioai instead of repeated.

    The saved script carries the recipe verbatim — it is the record of what the sandbox
    ran, and stays whole on disk — so the panel of a θ_Bn run showed 700 lines of which
    the model wrote five. The notebook export already keeps each recipe once; this is the
    same cut for a single script. It holds only when what ran is, to the byte, the recipe
    the installed package ships (`importlib.resources` reads that file, not the
    `HELIOAI_RECIPES_DIR` override): otherwise the view would run another recipe than the
    session did, and None sends the caller back to the full script. The digest is written
    into the view, which warns when a later helioai ships a different text.

    Args:
        code_src: A saved `code_N.py`.
        manifest: The session manifest, for the `load_data` → `fetch_series` rewrite.

    Returns:
        The short standalone script, or None when `code_src` is not a `run_recipe` run of
        a shipped, unmodified recipe.
    """
    from helioai.config import _PKG_RECIPES

    raw = _split_recipe_run(code_src)
    if raw is None:
        return None
    shipped = _PKG_RECIPES / f"{raw.name}.py"
    if not shipped.is_file():
        return None
    digest = _digest(raw.source)
    if _digest(shipped.read_text(encoding="utf-8")) != digest:
        return None
    run = _split_recipe_run(to_standalone(code_src, manifest, with_header=False))
    if run is None:
        return None
    body = _recipe_call_cell(run, where="read from the installed helioai above")
    # The recipe calls export() and numpy itself; the header must supply both.
    header = _standalone_header(body + "\nexport('', np.zeros(0))\n")
    n_lines = raw.source.count("\n") + 1
    loader = f"""\
# Recipe `{raw.name}` ({n_lines} lines) is not repeated here: what ran is identical to the one
# shipped with helioai {_helioai_version()} (sha256 {digest[:16]}…), so it is read from the
# installed package. The full script as it ran stays in the session workspace, and the
# notebook export carries the recipe once, collapsed.
import hashlib
import warnings
from importlib.resources import files
{_RUN_RECIPE_DEF.strip()}


_source = files("helioai").joinpath("data", "recipes", "{raw.name}.py").read_text(encoding="utf-8")
if hashlib.sha256(_source.rstrip("\\n").encode()).hexdigest() != "{digest}":
    warnings.warn("the installed helioai ships another {raw.name} than the one this run used")
RECIPE_SOURCES[{raw.name!r}] = _source"""
    return f"{header}\n\n\n{loader}\n\n\n{body}"


def build_notebook(user_id: str, session_id: str):
    """Build an nbformat notebook object for a session, without touching disk.

    Split from `export_session_notebook` so the web download can stream a
    notebook it never writes.

    Args:
        user_id: Storage owner.
        session_id: Session to render.

    Returns:
        An `nbformat` NotebookNode: provenance header, then one runnable cell
        per saved sandbox run, rewritten to standalone speasy calls.
    """
    import nbformat as nbf

    from helioai.workspace import user_home

    history = store.get_or_create(user_id, session_id)
    label = store.get_workspace_dir(user_id, session_id)
    workspace_dir = user_home(user_id) / "workspace" / label if label else None

    nb = nbf.v4.new_notebook()
    cells = []

    # Provenance
    now = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    param_ids = _collect_param_ids(history)
    prov = [
        "# HelioAI session export",
        "",
        f"- **Generated:** {now}",
        f"- **Session:** `{session_id}`",
        f"- **speasy:** {_version('speasy')} · **plasmapy:** {_version('plasmapy')} "
        f"· **helioai:** {_helioai_version()}",
    ]
    if param_ids:
        prov.append("- **Parameters referenced:** " + ", ".join(f"`{p}`" for p in param_ids))
    prov.append("\n> Run all cells top-to-bottom to reproduce the analysis.")
    cells.append(nbf.v4.new_markdown_cell("\n".join(prov)))

    # Methods & data acknowledgements — recipes used + references
    recipes = _collect_recipes(history)
    catalog_refs = _collect_catalog_refs(history)
    if recipes or param_ids or catalog_refs:
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
        for cite in catalog_refs:
            methods.append(f"- **Event catalog:** {cite}")
        methods.append(
            f"\n_Libraries: speasy {_version('speasy')}, plasmapy {_version('plasmapy')}, "
            f"helioai {_helioai_version()}._"
        )
        cells.append(nbf.v4.new_markdown_cell("\n".join(methods)))

    # Standalone rewrite of every saved run (load_data → fetch_series), so cells run
    # without the session data/ folder. Compute first to know if the load_data shim is
    # still needed in the setup cell. Runs the session reported as failed are kept as
    # prose, not as executable cells: code_N.py is written before it runs, and a raised
    # attempt exported as a cell stopped "Run All" before the corrected one ever ran.
    manifest = read_manifest(workspace_dir) if workspace_dir else {"datasets": {}}
    failed_names = _failed_code_names(history)
    runs: list[tuple[str, str]] = []
    failed_runs: list[tuple[str, str]] = []
    raw_sources: dict[str, str] = {}
    if workspace_dir and workspace_dir.exists():
        for p in _code_files(workspace_dir):
            raw = p.read_text(encoding="utf-8")
            raw_sources[p.name] = raw
            src = to_standalone(raw, manifest, with_header=False)
            (failed_runs if p.name in failed_names else runs).append((p.name, src))
    shim_needed = any("load_data(" in src for _, src in runs)
    recipe_runs = {name: _split_recipe_run(src) for name, src in runs}

    data_dir = (workspace_dir / "data") if workspace_dir else Path("data")
    setup_cell = _SETUP_CELL_BASE
    if shim_needed:
        setup_cell += (
            f"\n\nimport json, re\nfrom pathlib import Path\n"
            f"_HELIOAI_DATA_DIR = Path({str(data_dir)!r})\n" + _LOAD_DATA_SHIM
        )
    if any(recipe_runs.values()):
        setup_cell += _RUN_RECIPE_DEF
    cells.append(nbf.v4.new_code_cell(setup_cell))

    # Conversation narrative
    convo: list[str] = ["## Conversation"]
    for m in history:
        text = (m.content or "").strip()
        if m.role == "user" and text and m.origin:
            # Injected by the loop, not typed by the reader: kept — it explains why the
            # next answer changed — but not attributed to them.
            convo.append(f"_Automated note ({m.origin}):_ {text}")
        elif m.role == "user" and text:
            convo.append(f"**You:** {text}")
        elif m.role == "assistant" and text:
            convo.append(f"**HelioAI:** {text}")
    if len(convo) > 1:
        cells.append(nbf.v4.new_markdown_cell("\n\n".join(convo)))

    # Reproducible analysis: every saved run, in execution order
    if runs:
        cells.append(nbf.v4.new_markdown_cell("## Reproducible analysis"))
        recipes_placed: set[str] = set()
        for name, src in runs:
            cells.append(nbf.v4.new_markdown_cell(f"### {name}"))
            run = recipe_runs.get(name)
            if run is None:
                cells.append(nbf.v4.new_code_cell(src))
                continue
            if run.name not in recipes_placed:
                recipes_placed.add(run.name)
                raw = _split_recipe_run(raw_sources.get(name, ""))
                cells.extend(_recipe_source_cells(run, raw.source if raw else None))
            cells.append(nbf.v4.new_code_cell(_recipe_call_cell(run)))

    if failed_runs:
        cells.append(
            nbf.v4.new_markdown_cell(
                "## Attempts that did not run\n\n"
                "These scripts raised during the session and were superseded by the cells "
                "above. They are kept for the record, not for execution."
            )
        )
        for name, src in failed_runs:
            cells.append(nbf.v4.new_markdown_cell(f"### {name}\n\n```python\n{src}\n```"))

    nb["cells"] = cells
    return nb


def export_session_notebook(user_id: str, session_id: str, out_path: Path | None = None) -> Path:
    """Write the session as a .ipynb and return its path.

    Args:
        user_id: Storage owner.
        session_id: Session to export.
        out_path: Destination. Defaults to `<workspace>/<label>.ipynb`, falling
            back to the session id when the workspace has no label.

    Returns:
        The path written.

    Example:
        >>> export_session_notebook("cli", "8f3aa012-...")
        PosixPath('.../data/users/cli/workspace/plot-imf-bz_8f3aa0/plot-imf-bz_8f3aa0.ipynb')

        The notebook opens with a provenance header (parameter ids, library versions),
        then one runnable cell per saved sandbox run, rewritten to standalone speasy calls.
    """
    import nbformat as nbf

    nb = build_notebook(user_id, session_id)

    if out_path is None:
        from helioai.workspace import safe_id, user_home

        label = store.get_workspace_dir(user_id, session_id) or session_id
        root = user_home(user_id) / "workspace"
        root.mkdir(parents=True, exist_ok=True)
        out_path = root / f"{safe_id(label)}.ipynb"
    out_path = Path(out_path)
    nbf.write(nb, str(out_path))
    return out_path
