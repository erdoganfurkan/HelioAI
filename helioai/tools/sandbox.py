"""Python sandbox: execute user/LLM-generated code in an isolated subprocess.

Security model:
  - Runs in a fresh subprocess (separate memory, no shared globals)
  - Hard timeout (default 30s) — kills the process if exceeded
  - stdout/stderr captured and returned
  - No network isolation (speasy needs network access) — trust LLM-generated code

Pre-imports available in sandbox: speasy, plasmapy, numpy, scipy, matplotlib, astropy
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path

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

__sandbox_figures = []
_orig_show = plt.show
def _capture_show():
    import io, base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    __sandbox_figures.append(base64.b64encode(buf.read()).decode())
    plt.clf()
plt.show = _capture_show

"""

_SANDBOX_POSTAMBLE = """
import sys, json
_out = {"figures": __sandbox_figures}
print("__HELIOAI_RESULT__" + json.dumps(_out))
"""


async def run_python(code: str, timeout: float = 30.0) -> dict:
    """Execute Python code in an isolated subprocess.

    Args:
        code: Python source code to execute. Has access to speasy, plasmapy,
              numpy, scipy, matplotlib (Agg backend). plt.show() captures
              the figure as base64 PNG.
        timeout: maximum execution time in seconds

    Returns dict with:
        - stdout: captured text output
        - stderr: captured errors/warnings
        - figures: list of base64-encoded PNG strings
        - error: error message if execution failed
    """
    full_code = _SANDBOX_PREAMBLE + textwrap.dedent(code) + "\n" + _SANDBOX_POSTAMBLE

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", full_code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return {"error": f"Execution timed out after {timeout}s", "stdout": "", "stderr": ""}

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        # Extract structured result
        figures: list[str] = []
        clean_stdout_lines: list[str] = []
        for line in stdout.splitlines():
            if line.startswith("__HELIOAI_RESULT__"):
                try:
                    payload = json.loads(line[len("__HELIOAI_RESULT__"):])
                    figures = payload.get("figures", [])
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
                "figures": figures,
            }

        return {
            "stdout": clean_stdout,
            "stderr": stderr.strip() if stderr.strip() else None,
            "figures": figures,
            "n_figures": len(figures),
        }

    except Exception as e:
        return {"error": f"Sandbox error: {e}"}
