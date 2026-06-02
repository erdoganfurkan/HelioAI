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
import sys
import textwrap
from pathlib import Path


def _set_subprocess_limits() -> None:
    """Apply resource limits inside the sandbox subprocess (Linux only).

    Called via preexec_fn — runs in the child process after fork, before exec.
    Degrades silently on non-Linux or permission error.

    RLIMIT_AS (virtual memory) is intentionally not set: numpy/scipy/speasy use
    large sparse mmap regions at import time that can exceed any safe threshold,
    causing OSError on import rather than at actual allocation. The hard timeout
    already handles runaway CPU usage.
    """
    try:
        import resource

        # 200 MB max file write — prevents disk exhaustion from large figure dumps
        _200MB = 200 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (_200MB, _200MB))
    except Exception:
        pass

_SANDBOX_PREAMBLE = """\
import warnings
warnings.filterwarnings('ignore')
import os
os.environ.setdefault('MPLBACKEND', 'Agg')

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import speasy as spz
except ImportError:
    spz = None

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

__sandbox_figure_paths = []
__sandbox_exports = {}
__sandbox_cards = []

_orig_show = plt.show
def _capture_show():
    path = os.path.join(__sandbox_plot_dir, f"fig_{__sandbox_run_idx}_{len(__sandbox_figure_paths)}.png")
    plt.savefig(path, dpi=100, bbox_inches='tight')
    __sandbox_figure_paths.append(path)
    plt.clf()
plt.show = _capture_show


def export(name, data):
    \"\"\"Export numerical data for LLM interpretation. Call instead of or in addition to plt.show().\"\"\"
    try:
        arr = np.asarray(data, dtype=float)
        flat = arr.flatten()
        finite = flat[np.isfinite(flat)]
        __sandbox_exports[name] = {
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
        __sandbox_cards.append({
            "kind": "parameter_card",
            "param_id": param_id,
            "name": str(getattr(var, "name", "") or ""),
            "mission": parts[1] if len(parts) > 1 else parts[0],
            "instrument": str(meta.get("FIELDNAM", "") or "")[:80],
            "units": str(getattr(var, "unit", "") or ""),
            "cadence": cadence,
            "components": list(getattr(var, "columns", None) or []),
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


async def run_python(
    code: str, timeout: float = 60.0, _plot_dir: str | None = None, _run_idx: int | None = None
) -> dict:
    """Execute Python code in an isolated subprocess.

    Args:
        code: Python source code to execute. Has access to speasy (spz), plasmapy (pf),
              numpy (np), scipy, matplotlib (Agg — plt.show() saves to disk),
              astropy units (u).
              Call export(name, array) to share numerical results with the LLM.
        timeout: maximum execution time in seconds
        _plot_dir: injected by the agent loop — workspace dir for this run.
                   Not exposed in the LLM tool schema.

    Returns dict with:
        - stdout: captured text output
        - stderr: captured errors/warnings
        - figure_paths: list of absolute paths to saved PNG files
        - exports: dict of named numerical summaries (from export() calls)
        - error: error message if execution failed
    """
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

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            full_code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=_set_subprocess_limits if sys.platform != "win32" else None,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return {"error": f"Execution timed out after {timeout}s", "stdout": "", "stderr": ""}

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

        clean_stdout = "\n".join(clean_stdout_lines).strip()

        if proc.returncode != 0:
            return {
                "error": f"Code exited with code {proc.returncode}",
                "stdout": clean_stdout,
                "stderr": stderr.strip(),
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
