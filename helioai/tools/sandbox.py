"""Python sandbox: execute user/LLM-generated code in an isolated subprocess.

Security model:
  - Runs in a fresh subprocess (separate memory, no shared globals)
  - Hard timeout (default 30s) — kills the process if exceeded
  - stdout/stderr captured and returned
  - No network isolation (speasy needs network access) — trust LLM-generated code

Pre-imports available in sandbox: speasy, plasmapy, numpy, scipy, matplotlib, astropy
Figures are saved to a temp directory; paths are returned (not base64).
Use export(name, array) to share numerical data with the LLM.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
from functools import lru_cache
from pathlib import Path

_MAX_TIMEOUT_S = 300.0  # hard ceiling regardless of the caller-supplied timeout

# Environment whitelist for the sandbox subprocess. Only these (and the prefixes
# below) are passed through; everything else — notably LLM provider API keys —
# is dropped so LLM-generated code cannot read secrets from os.environ.
_ENV_KEEP = frozenset(
    {
        "PATH",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "TERM",
        "TZ",
        "TMPDIR",
        "TMP",
        "TEMP",
        "VIRTUAL_ENV",
        "MPLBACKEND",
        # Windows essentials. Absent on POSIX, so listing them unconditionally
        # changes nothing there. SYSTEMROOT is not optional: Winsock cannot locate
        # its service providers without it and WSAStartup fails with
        # `OSError: [WinError 10106]` on the first import that touches sockets —
        # which is every sandbox run, since the preamble imports speasy.
        # The APPDATA pair matters too, or matplotlib finds no home and rebuilds
        # its font cache on every single call.
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "APPDATA",
        "LOCALAPPDATA",
        "USERPROFILE",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    }
)
_ENV_KEEP_PREFIXES = ("XDG_", "LC_", "SPEASY_", "SPEDAS_", "PYTHON")

# ponytail: fork-bomb cap, generous so numpy/scipy thread pools still spawn.
_MAX_PROCS = 4096


def _sandbox_env(home: str = "/tmp") -> dict[str, str]:
    """Filtered os.environ for the sandbox — secrets stripped (see _ENV_KEEP).

    HOME is explicitly set (default /tmp, or the plot_dir when bwrap is active).
    The bwrap sandbox mounts /tmp as a writable tmpfs while the rest of the
    filesystem is read-only. Libraries that create caches at import time
    (speasy SQLite, matplotlib font cache) need a writable home directory.
    """
    env = {
        k: v for k, v in os.environ.items() if k in _ENV_KEEP or k.startswith(_ENV_KEEP_PREFIXES)
    }
    env.setdefault("MPLBACKEND", "Agg")
    env["HOME"] = home
    # Redirect all XDG base dirs under the writable home. Otherwise a host
    # XDG_DATA_HOME/XDG_CONFIG_HOME (kept via the XDG_ prefix) leaks through and
    # points speasy's diskcache index at a path bwrap mounts read-only → the
    # SQLite index open fails with "attempt to write a readonly database".
    env["XDG_CACHE_HOME"] = os.path.join(home, ".cache")
    env["XDG_DATA_HOME"] = os.path.join(home, ".local", "share")
    env["XDG_CONFIG_HOME"] = os.path.join(home, ".config")
    # matplotlib ignores XDG on Windows, so point it at the writable home
    # explicitly rather than letting it fall back to a fresh temp directory and
    # rebuild the font cache on every single run.
    env["MPLCONFIGDIR"] = os.path.join(home, ".cache", "matplotlib")
    # Keeping the inventory fresh is the host's job, not the sandbox's. The copy seeded
    # by `_seed_speasy_inventory` is only ever as old as the host's own, which speasy
    # refreshes on its usual cycle outside here; letting each spawn re-validate it over
    # the network instead costs a round trip per run (11 s against 5 s measured) and
    # buys nothing, since every session starts from that same copy.
    env["SPEASY_INVENTORIES_CACHE_RETENTION_DAYS"] = "365"
    return env


def _seed_speasy_inventory(home: str) -> None:
    """Give a fresh sandbox home the speasy inventory the host already built.

    The sandbox hands every session a new HOME, so `XDG_DATA_HOME` is new too, so
    `import speasy` finds no inventory and rebuilds it — 35 MB fetched from the
    providers, well past the default run timeout. The spawn is then SIGKILLed
    mid-build, the index stays incomplete, and the next spawn starts over: a session
    can burn every one of its `run_python` calls without executing a line of user
    code. A `print("hello")` timing out at 30 s is what this looks like from outside.

    Copying is deliberate rather than sharing one directory across sessions: the
    index is a diskcache SQLite that the sandbox must be able to write, and under
    bwrap only the session's own directory is really on disk (`data_dir` is masked by
    a tmpfs). A copy needs no new bind mount and no cross-session locking.

    Best-effort by construction — no host inventory, no permission, no space, and the
    session simply pays the rebuild as before. It must never be the reason a run fails.
    """
    src = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share", "speasy")
    dst = Path(home, ".local", "share", "speasy")
    if dst.exists() or not src.is_dir():
        return
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
    except (OSError, shutil.Error) as e:
        from helioai.logging_config import get_logger

        get_logger(__name__).debug("speasy_inventory_not_seeded", src=str(src), error=str(e))


def _kill_proc_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill the subprocess and its whole session (grandchildren included)."""
    if sys.platform != "win32":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            pass
    proc.kill()


_BWARP = shutil.which("bwrap")


def _bwrap_works() -> bool:
    """Check that bubblewrap can actually set up a namespace on this host.

    `shutil.which` finds the binary but the kernel may not allow user
    namespaces (Debian default, Docker seccomp profiles, etc.).
    """
    if not _BWARP or sys.platform != "linux":
        return False
    try:
        proc = subprocess.run(
            [_BWARP, "--ro-bind", "/", "/", "true"],
            capture_output=True,
            timeout=5,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _build_sandbox_cmd(plot_dir: str, full_code: str) -> list[str]:
    """Build the sandbox execution command.

    Prefers bubblewrap (bwrap) for PID-namespace isolation — prevents the
    sandbox from reading /proc/<server-pid>/environ even when the server and
    sandbox share the same host UID. Falls back to plain python + preexec_fn
    when bwrap is not functional (local dev, restrictive seccomp profiles).
    """
    if _bwrap_works():
        from helioai.config import _ROOT, settings

        data_dir = str(settings.data_dir)
        cmd = [
            _BWARP,
            "--unshare-pid",
            "--unshare-ipc",
            "--ro-bind",
            "/",
            "/",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            # Empty tmpfs over data/ hides users.json (password hashes), sessions.db
            # and every other user's workspace/catalogs — the whole point of the sandbox.
            "--tmpfs",
            data_dir,
        ]
        # Mask the .env file — it holds the LLM provider API keys and is otherwise
        # readable through --ro-bind / / even though it's absent from the subprocess env.
        env_file = _ROOT / ".env"
        if env_file.exists():
            cmd += ["--ro-bind", "/dev/null", str(env_file)]
        # Re-bind THIS session's workspace writable LAST, so no earlier tmpfs
        # (/tmp or data/) can mask it — plot_dir may live under either.
        cmd += ["--bind", plot_dir, plot_dir]
        # Start IN the workspace. Without this the cwd is inherited from the server,
        # which is the repo root under --ro-bind / /, so `open("out.py", "w")` failed
        # with EROFS — the standalone-script export in examples/02 could not write
        # its file no matter how the model was prompted.
        cmd += ["--chdir", plot_dir]
        cmd += [sys.executable, "-c", full_code]
        return cmd
    return [sys.executable, "-c", full_code]


def _preexec_fn() -> callable | None:
    """preexec_fn for the non-bwrap path — limits + privilege drop."""
    if sys.platform == "win32":
        return None
    return _set_subprocess_limits


def _drop_privileges() -> None:
    """Drop to the dedicated sandbox user (or nobody) after fork, before exec.

    Runs in the child via preexec_fn. When the sandbox subprocess shares the
    server's UID it can read /proc/<ppid>/environ (API keys) and other server
    files — this call closes that hole. Degrades silently when unavailable.
    """
    try:
        import pwd

        sb = pwd.getpwnam("helioai-sandbox")
        os.setgid(sb.pw_gid)
        os.setuid(sb.pw_uid)
        return
    except (KeyError, PermissionError):
        pass
    try:
        nb = pwd.getpwnam("nobody")
        os.setgid(nb.pw_gid)
        os.setuid(nb.pw_uid)
    except Exception:
        pass


def _isolation_gap() -> str | None:
    """Why the fallback path will not drop privileges, or None if it will.

    Separate from the warning so it can be asserted on directly: whether a log line
    lands in stdout, stderr or the stdlib capture depends on who configured logging
    first, which is not what this needs to be right about.
    """
    if not hasattr(os, "geteuid"):
        return "no POSIX privilege model on this platform"
    if os.geteuid() != 0:
        return "server is not root, so setuid in the child cannot succeed"
    try:
        import pwd

        pwd.getpwnam("helioai-sandbox")
    except KeyError:
        return "user 'helioai-sandbox' does not exist"
    return None


@lru_cache(maxsize=1)
def _warn_if_not_isolated() -> None:
    """Say once, in the parent, that the fallback path isolates nothing.

    Reaching this function already means bwrap is unavailable or non-functional, so
    generated code has no filesystem isolation whatever the privilege drop does —
    that is worth saying even when the drop succeeds, which is exactly the Docker
    case (root plus the helioai-sandbox user, and nothing else).

    `_drop_privileges` itself runs in the forked child via preexec_fn — it cannot log
    (structlog after fork, before exec) and it swallows its own failure, which is
    correct there and invisible everywhere else. The condition is knowable before the
    spawn, so it is checked before the spawn.
    """
    from helioai.logging_config import get_logger

    get_logger(__name__).warning(
        "sandbox_not_isolated",
        detail="bubblewrap unavailable: generated code has no filesystem isolation",
        privileges=_isolation_gap() or "dropped to helioai-sandbox",
        remedy="install bubblewrap",
    )


def _set_subprocess_limits() -> None:
    """Apply resource limits + drop to sandbox user (pre-exec hook).

    Called via preexec_fn — runs in the child process after fork, before exec.
    Degrades silently on non-Linux or permission error.

    RLIMIT_AS (virtual memory) is intentionally not set: numpy/scipy/speasy use
    large sparse mmap regions at import time that can exceed any safe threshold,
    causing OSError on import rather than at actual allocation. The hard timeout
    already handles runaway CPU usage.
    """
    _drop_privileges()
    try:
        import resource

        # 200 MB max file write — prevents disk exhaustion from large figure dumps
        _200MB = 200 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (_200MB, _200MB))
        # Cap process count — fork-bomb defense in depth (timeout + killpg also apply)
        _, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        cap = _MAX_PROCS if hard == resource.RLIM_INFINITY else min(_MAX_PROCS, hard)
        resource.setrlimit(resource.RLIMIT_NPROC, (cap, hard))
    except Exception:
        pass


_SANDBOX_PREAMBLE = """\
import sys as _sys, io as _io, warnings, os
warnings.filterwarnings('ignore')
os.environ.setdefault('MPLBACKEND', 'Agg')

# Ensure cache directories exist before library init.  Libraries that create
# SQLite caches at import time (speasy uses diskcache) need their parent dirs
# to exist.  With bwrap the filesystem is read-only except for HOME and /tmp
# so we must pre-create them here.
_hm = os.environ.get('HOME', '/tmp')
for _d in (os.path.join(_hm, '.cache', 'speasy'),
           os.path.join(_hm, '.local', 'share', 'speasy', 'index'),
           os.path.join(_hm, '.config', 'speasy'),
           os.path.join(_hm, '.matplotlib')):
    os.makedirs(_d, exist_ok=True)

import json
import re as _re

# Suppress stdout during library init — speasy prints network errors at import time
_saved_stdout = _sys.stdout
_sys.stdout = _io.StringIO()
try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # speasy on first attribute access only. Importing it refreshes an inventory
    # over the network, so a print("hello") used to depend on CDAWeb being
    # reachable — that is what cascaded into 23 CI timeouts. The prompts tell the
    # model to use load_data() rather than spz.get_data, so almost no run pays it.
    class _LazySpeasy:
        _mod = None

        def __getattr__(self, name):
            if _LazySpeasy._mod is None:
                _out = _sys.stdout
                _sys.stdout = _io.StringIO()
                try:
                    import speasy
                    _LazySpeasy._mod = speasy
                finally:
                    _sys.stdout = _out
            return getattr(_LazySpeasy._mod, name)

    spz = _LazySpeasy()

    try:
        import plasmapy
        import plasmapy.formulary as pf
        import astropy.units as u
    except ImportError:
        pf = None

    try:
        import scipy
        from scipy import signal, stats, fft
    except ImportError:
        scipy = None

    try:
        from helioai.tools.sandbox_helpers import transform_coords, mp_shue1998, bs_jelinek2012
    except Exception:
        transform_coords = mp_shue1998 = bs_jelinek2012 = None
finally:
    _sys.stdout = _saved_stdout

__sandbox_figure_paths = []
__sandbox_exports = {}
__sandbox_cards = []

_orig_show = plt.show
def _capture_show():
    path = os.path.join(__sandbox_plot_dir, f"fig_{__sandbox_run_idx}_{len(__sandbox_figure_paths)}.png")
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.savefig(path[:-4] + ".pdf", dpi=300, bbox_inches='tight')
    __sandbox_figure_paths.append(path)
    plt.clf()
plt.show = _capture_show


def export(name, data, units=""):
    \"\"\"Export numerical data for LLM interpretation. Call instead of or in addition to plt.show().

    Pass `units` whenever the quantity has one: the value is kept in the session
    provenance ledger, and without a unit a ratio of 2.53 and 2.53 nT are the same number.

    A dict of summary numbers is exported key by key as `name.key`. Passing one used to
    fail — float() on a dict raises — and the failure was silent, so the numbers reached
    the reply through the error repr and nothing else: computed, published, untraceable.
    \"\"\"
    if isinstance(data, dict):
        for _k, _v in data.items():
            if isinstance(_v, (dict, int, float)) and not isinstance(_v, bool):
                export(str(name) + "." + str(_k), _v, units)
        return
    try:
        arr = np.asarray(data, dtype=float)
        flat = arr.flatten()
        finite = flat[np.isfinite(flat)]
        __sandbox_exports[name] = {
            "units": str(units),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": float(np.nanmin(arr)) if finite.size else None,
            "max": float(np.nanmax(arr)) if finite.size else None,
            "mean": float(np.nanmean(arr)) if finite.size else None,
            "std": float(np.nanstd(arr)) if finite.size else None,
            "n_finite": int(finite.size),
            "n_nan": int(flat.size - finite.size),
            "sample": [float(x) for x in flat[:8].tolist()],
        }
    except Exception as e:
        __sandbox_exports[name] = {"error": str(e), "repr": repr(data)[:200]}


def clean(values):
    \"\"\"Blank infinities and ~1e31 fill values to NaN.

    You do NOT need this for load_data() results: get_timeseries already blanks
    fill to NaN using each dataset's declared FILLVAL, which is the only way to
    catch conventions like Wind/SWE's 99999.9 that this function cannot see.
    Use it only on arrays you fetched yourself inside the sandbox.
    \"\"\"
    arr = np.asarray(values, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    arr[np.abs(arr) >= 1e30] = np.nan
    return arr


def magnitude(vectors):
    \"\"\"|V| of an N×3 array, where a data gap stays a gap.

    `np.sqrt(np.nansum(v**2, axis=1))` is the idiom that gets hand-written here run after
    run, and it is wrong in the one way that matters: `nansum` reads a missing sample as
    zero, so a three-component gap becomes a magnitude of exactly 0 nT. That value plots,
    it survives every finite check, and `np.diff` sees the recovery out of the gap as the
    largest jump in the interval — which is how a shock detector came to fire on a 90-second
    hole in Wind/MFI and report a shock 3.5 minutes early.
    \"\"\"
    arr = clean(vectors)
    return np.sqrt(np.sum(arr**2, axis=-1))


def interp_to(t_target, t_source, values):
    \"\"\"Resample `values` (sampled at `t_source`) onto the `t_target` time base.

    Putting two instruments on a common clock is the most frequent operation here, and
    hand-rolling it fails the same two ways every run: `np.timedelta64` has no
    `.total_seconds()` (that is stdlib's timedelta), and `np.interp` is 1-D only, so a
    3-component field raises \"object too deep for desired array\".

    Handles datetime64 axes and 2-D values. A target point whose linear estimate would
    depend on a missing sample comes back NaN rather than bridged: interpolating across
    a gap invents measurements, and that is how a data hole becomes a plotted feature.
    \"\"\"
    tt = np.asarray(t_target)
    ts = np.asarray(t_source)
    if np.issubdtype(tt.dtype, np.datetime64):
        tt = tt.astype('datetime64[ns]').astype('float64')
    if np.issubdtype(ts.dtype, np.datetime64):
        ts = ts.astype('datetime64[ns]').astype('float64')
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        good = np.isfinite(v)
        if not good.any():
            return np.full(tt.shape, np.nan)
        out = np.interp(tt, ts[good], v[good], left=np.nan, right=np.nan)
        # Any weight on a missing sample disqualifies the point — not a majority of it.
        touched = np.interp(tt, ts, (~good).astype(float), left=1.0, right=1.0)
        out[touched > 0.0] = np.nan
        return out
    cols = [interp_to(t_target, t_source, v[:, i]) for i in range(v.shape[1])]
    return np.column_stack(cols)


def load_data(name):
    \"\"\"Load a dataset saved by get_timeseries or get_events_timeseries.

    Fill values are ALREADY NaN — do not call clean() on .values, and do not
    filter on magnitude. Use np.isnan() to mask, or the nan-aware reductions
    (np.nanmean, np.nanmax, ...). Timeseries carry .missing_pct, the percentage
    of samples with no measurement.
    \"\"\"
    import json as _json, types as _types
    if not _re.fullmatch(r"[a-z0-9_]+", str(name)):
        raise ValueError(f"invalid dataset name {name!r} — use only lowercase letters, digits and underscores")
    _ddir = os.path.join(__sandbox_plot_dir, "data")
    _mfile = os.path.join(_ddir, "manifest.json")
    if not os.path.exists(_mfile):
        raise FileNotFoundError("no dataset manifest found — run get_timeseries or get_events_timeseries first")
    _manifest = _json.loads(open(_mfile).read())
    _entry = _manifest.get("datasets", {}).get(name)
    if _entry is None:
        _available = sorted(_manifest.get("datasets", {}).keys())
        raise KeyError(f"unknown dataset {name!r} — available: {_available}")
    _fpath = os.path.join(_ddir, _entry["file"])
    _z = np.load(_fpath, allow_pickle=False)
    if _entry["kind"] == "timeseries":
        _ns = _types.SimpleNamespace()
        _ns.time = _z["time"]
        _ns.values = _z["values"]
        _ns.columns = _entry.get("columns", [])
        _ns.units = _entry.get("units", "")
        _ns.param_id = _entry.get("param_id", "")
        _ns.missing_pct = _entry.get("missing_pct", 0.0)
        return _ns
    elif _entry["kind"] == "event_collection":
        _events_meta = _entry.get("events", [])
        _result = []
        for _em in _events_meta:
            if _em.get("status") != "ok":
                continue
            _i = _em["idx"]
            _ev = _types.SimpleNamespace()
            _ev.time = _z[f"t{_i}"]
            _ev.values = _z[f"v{_i}"]
            _ev.start = _em["start"]
            _ev.stop = _em["stop"]
            _ev.units = _entry.get("units", "")
            _result.append(_ev)
        return _result
    raise ValueError(f"unknown dataset kind {_entry['kind']!r}")


def save_path(name):
    \"\"\"Absolute path to write `name` into this session's own workspace.

    A bare relative path also lands here — the sandbox's cwd already is this directory —
    but a script that changes directory, or a path you build once and hand to something
    else, needs the absolute form. Never construct that path yourself: only this exact
    directory is writable. Under bwrap, everything else under the data root is an empty
    tmpfs overlay that looks writable and is not — a run that hardcoded one directory up
    printed "Wrote <path>", the write itself did not raise, and the file was gone the
    moment the sandbox process exited.\"\"\"
    return os.path.join(__sandbox_plot_dir, str(name))


def document_method(name, reference="", method=""):
    \"\"\"Record a scientific method/algorithm used (provenance). Call when you compute a derived
    quantity outside a recipe (e.g. MVAB inline). Shown in the UI and the exported Methods section.\"\"\"
    __sandbox_cards.append({
        "kind": "method_used",
        "name": str(name),
        "reference": str(reference),
        "method": str(method),
    })


def param_card(var, param_id: str) -> None:
    \"\"\"Emit a parameter metadata card for display in the UI. Call after spz.get_data().\"\"\"
    try:
        t = var.time
        cadence = ""
        if len(t) > 1:
            deltas = np.diff(t.astype("datetime64[ms]").astype(float))
            med_ms = float(np.median(deltas))
            if med_ms >= 3_600_000:
                cadence = f"{med_ms / 3_600_000:.4g} h"
            elif med_ms >= 60_000:
                cadence = f"{med_ms / 60_000:.4g} min"
            elif med_ms >= 1_000:
                cadence = f"{med_ms / 1_000:.4g} s"
            else:
                cadence = f"{med_ms:.4g} ms"
        meta = getattr(var, "meta", {}) or {}
        parts = param_id.split("/")
        columns = list(getattr(var, "columns", None) or [])
        coord_sys = ""
        for _key in ("COORDINATE_SYSTEM", "COORDINATE_SYSTEM_NAME", "FRAME", "FRAME_ORIGIN"):
            _val = str(meta.get(_key, "") or "").strip()
            if _val:
                coord_sys = _val[:20]
                break
        if not coord_sys:
            _name = str(getattr(var, "name", "") or "")
            _haystack = " ".join(filter(None, [param_id, _name] + columns))
            _m = _re.search(
                r"\b(GSE|GSM|RTN|HCI|HAE|HEE|HEEQ|GCI|SSE|VSO|MSO|MFA|FAC)\b",
                _haystack, _re.IGNORECASE
            )
            if _m:
                coord_sys = _m.group(0).upper()
        __sandbox_cards.append({
            "kind": "parameter_card",
            "param_id": param_id,
            "name": str(getattr(var, "name", "") or ""),
            "mission": parts[1] if len(parts) > 1 else parts[0],
            "instrument": str(meta.get("FIELDNAM", "") or "")[:80],
            "units": str(getattr(var, "unit", "") or ""),
            "cadence": cadence,
            "coord_sys": coord_sys,
            "components": columns,
            "n_points": len(t),
        })
    except Exception:
        pass

"""

_SANDBOX_POSTAMBLE = """
import sys, json
_out = {"figure_paths": __sandbox_figure_paths, "exports": __sandbox_exports, "cards": __sandbox_cards}
print("__HELIOAI_RESULT__" + json.dumps(_out))
"""

# Lines injected ahead of the agent's code: the two __sandbox_* assignments plus the
# preamble. Derived, never hard-coded — the preamble grows, and a stale constant would
# silently start pointing tracebacks at the wrong line again.
_PREAMBLE_LINES = 2 + len(_SANDBOX_PREAMBLE.splitlines())

_TRACEBACK_FRAME = re.compile(r'File "<string>", line (\d+)')


def _rewrite_traceback(stderr: str) -> str:
    """Renumber `File "<string>", line N` frames onto the agent's own code.

    The traceback counts from the top of the assembled script, so a one-line typo was
    reported ~212 lines below where the agent could see it — and the `code_N.py` written
    to the workspace holds the agent's lines alone, so its numbering did not match either.
    Frames inside the preamble keep their raw number, tagged, since they are ours.
    """

    def fix(m: re.Match) -> str:
        line = int(m.group(1))
        if line <= _PREAMBLE_LINES:
            return f'File "<sandbox preamble>", line {line}'
        return f'File "your code", line {line - _PREAMBLE_LINES}'

    return _TRACEBACK_FRAME.sub(fix, stderr)


def _error_summary(stderr: str, returncode: int) -> str:
    """The exception line itself, which is what the agent needs to fix its code.

    `Code exited with code 1` said nothing and was the only field kept once the result
    aged out of context, so the same typo came back three times in one session.
    """
    lines = [ln.strip() for ln in stderr.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        if not ln.startswith(("File ", "Traceback", "^", "~")):
            return ln
    return f"Code exited with code {returncode}"


async def run_python(
    code: str, timeout: float = 60.0, _plot_dir: str | None = None, _run_idx: int | None = None
) -> dict:
    """Execute Python code in an isolated subprocess.

    Args:
        code: Python source code to execute. Has access to speasy (spz), plasmapy (pf),
              numpy (np), scipy, matplotlib (Agg — plt.show() saves to disk),
              astropy units (u).
              Call export(name, array) to share numerical results with the LLM.
        timeout: maximum execution time in seconds — clamped to _MAX_TIMEOUT_S
        _plot_dir: injected by the agent loop — workspace dir for this run.
                   Not exposed in the LLM tool schema.

    Returns dict with:
        - stdout: captured text output
        - stderr: captured errors/warnings
        - figure_paths: list of absolute paths to saved PNG files
        - exports: dict of named numerical summaries (from export() calls)
        - error: error message if execution failed
    """
    timeout = min(timeout, _MAX_TIMEOUT_S)
    if _plot_dir is None:
        from helioai.workspace import get_run_dir_for_sandbox

        _plot_dir = get_run_dir_for_sandbox()
    run_idx = _run_idx if _run_idx is not None else 0
    plot_dir = _plot_dir
    from helioai.logging_config import get_logger as _get_logger

    _get_logger(__name__).info("sandbox_plot_dir", plot_dir=plot_dir, run_idx=run_idx)
    code_file = Path(plot_dir, f"code_{run_idx}.py")
    dedented_code = textwrap.dedent(code)
    code_file.write_text(dedented_code, encoding="utf-8")
    n_lines = len(dedented_code.splitlines())
    plot_dir_line = f"__sandbox_plot_dir = {plot_dir!r}\n__sandbox_run_idx = {run_idx!r}\n"
    full_code = (
        plot_dir_line + _SANDBOX_PREAMBLE + textwrap.dedent(code) + "\n" + _SANDBOX_POSTAMBLE
    )

    cmd = _build_sandbox_cmd(plot_dir, full_code)
    using_bwrap = cmd[0].endswith("bwrap") if cmd else False

    try:
        if using_bwrap:
            _seed_speasy_inventory(plot_dir)
            sandbox_env = _sandbox_env(home=plot_dir)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=sandbox_env,
                start_new_session=True,
            )
        else:
            _warn_if_not_isolated()
            sandbox_env = _sandbox_env()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=sandbox_env,
                start_new_session=True,
                preexec_fn=_preexec_fn(),
                cwd=plot_dir,  # same working directory as the bwrap path's --chdir
            )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            _kill_proc_tree(proc)
            stdout_bytes, stderr_bytes = await proc.communicate()
            return {
                "error": f"Execution timed out after {timeout}s",
                "stdout": stdout_bytes.decode("utf-8", errors="replace")[-2000:],
                "stderr": stderr_bytes.decode("utf-8", errors="replace")[-2000:],
            }

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        figure_paths: list[str] = []
        exports: dict = {}
        cards: list[dict] = []
        clean_stdout_lines: list[str] = []
        for line in stdout.splitlines():
            if line.startswith("__HELIOAI_RESULT__"):
                try:
                    payload = json.loads(line[len("__HELIOAI_RESULT__") :])
                    figure_paths = payload.get("figure_paths", [])
                    exports = payload.get("exports", {})
                    cards = payload.get("cards", [])
                except json.JSONDecodeError:
                    pass
            else:
                clean_stdout_lines.append(line)

        _MAX_STDOUT = 4000
        clean_stdout = "\n".join(clean_stdout_lines).strip()
        if len(clean_stdout) > _MAX_STDOUT:
            clean_stdout = (
                clean_stdout[:_MAX_STDOUT]
                + f"\n[stdout truncated — {len(clean_stdout)} chars total; use export() for numerical data]"
            )

        if proc.returncode != 0:
            agent_stderr = _rewrite_traceback(stderr.strip())
            return {
                "error": _error_summary(agent_stderr, proc.returncode),
                "stdout": clean_stdout,
                "stderr": agent_stderr,
                "figure_paths": figure_paths,
                "exports": exports,
                "cards": cards,
                "code_path": str(code_file),
                "n_lines": n_lines,
            }

        return {
            "stdout": clean_stdout,
            "stderr": stderr.strip() if stderr.strip() else None,
            "figure_paths": figure_paths,
            "n_figures": len(figure_paths),
            "exports": exports,
            "cards": cards,
            "code_path": str(code_file),
            "n_lines": n_lines,
        }

    except Exception as e:
        return {"error": f"Sandbox error: {e}"}
