"""Tests for helioai.tools.sandbox.run_python."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from helioai.tools import sandbox
from helioai.tools.sandbox import run_python


async def test_simple_stdout_captured() -> None:
    result = await run_python("print('hello sandbox')")
    assert result.get("error") is None
    assert result["stdout"] == "hello sandbox"


async def test_returncode_zero_no_error() -> None:
    result = await run_python("x = 1 + 1")
    assert "error" not in result or result.get("error") is None


async def test_timeout_hard() -> None:
    result = await run_python("import time; time.sleep(100)", timeout=5.0)
    assert "error" in result
    assert "timed out" in result["error"].lower()


async def test_syntax_error_returns_error_not_exception() -> None:
    result = await run_python("def broken(:\n    pass")
    assert "error" in result


async def test_runtime_error_returns_error() -> None:
    result = await run_python("raise ValueError('bad input')")
    assert "error" in result


async def test_export_returns_stats() -> None:
    code = """
import numpy as np
arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
export('my_array', arr)
"""
    result = await run_python(code)
    assert result.get("error") is None
    exports = result.get("exports", {})
    assert "my_array" in exports
    stats = exports["my_array"]
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(5.0)
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["n_finite"] == 5
    assert stats["shape"] == [5]


async def test_export_scalar() -> None:
    code = "export('ratio', 2.97)"
    result = await run_python(code)
    assert result.get("error") is None
    stats = result["exports"]["ratio"]
    assert stats["mean"] == pytest.approx(2.97)


async def test_plt_show_creates_png_on_disk() -> None:
    code = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.plot(np.arange(10), np.random.rand(10))
plt.show()
"""
    result = await run_python(code)
    assert result.get("error") is None
    figure_paths = result.get("figure_paths", [])
    assert len(figure_paths) >= 1
    assert os.path.exists(figure_paths[0])
    assert figure_paths[0].endswith(".png")


async def test_plt_show_creates_pdf_alongside_png() -> None:
    code = """
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
plt.show()
"""
    result = await run_python(code)
    assert result.get("error") is None
    figure_paths = result.get("figure_paths", [])
    assert len(figure_paths) >= 1
    png_path = figure_paths[0]
    pdf_path = png_path[:-4] + ".pdf"
    assert os.path.exists(png_path)
    assert os.path.exists(pdf_path), f"PDF not found at {pdf_path}"
    assert png_path.endswith(".png")
    assert pdf_path.endswith(".pdf")


async def test_multiple_plt_show_creates_multiple_files() -> None:
    code = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.plot([1, 2])
plt.show()
plt.plot([3, 4])
plt.show()
"""
    result = await run_python(code)
    assert result.get("error") is None
    assert result.get("n_figures", 0) == 2


async def test_numpy_available_in_sandbox() -> None:
    result = await run_python("import numpy as np; print(np.__version__)")
    assert result.get("error") is None
    assert result["stdout"]


async def test_stdout_and_exports_coexist() -> None:
    code = """
print('output line')
export('val', 42.0)
"""
    result = await run_python(code)
    assert result.get("error") is None
    assert result["stdout"] == "output line"
    assert "val" in result["exports"]


async def test_stdout_truncated_at_4000_chars() -> None:
    code = "print('x' * 10000)"
    result = await run_python(code)
    assert result.get("error") is None
    assert len(result["stdout"]) <= 4100
    assert "truncated" in result["stdout"]


async def test_clean_masks_fill_values() -> None:
    code = """
import numpy as np
arr = np.array([1.0, -1e31, 9.96e36, 2.0, float('inf'), float('-inf')])
cleaned = clean(arr)
export('cleaned', cleaned)
"""
    result = await run_python(code)
    assert result.get("error") is None
    stats = result["exports"]["cleaned"]
    assert stats["n_nan"] >= 4
    assert stats["max"] is not None and stats["max"] < 1e29
    assert stats["min"] is not None and stats["min"] > -1e29


@pytest.mark.asyncio
async def test_superposed_epoch_recipe_end_to_end(tmp_path) -> None:
    """End-to-end: synthetic manifest + npz → superposed_epoch recipe → figure + exports."""
    # Build synthetic events: 20 events, each a half-sine pulse in Bz
    rng = np.random.default_rng(0)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    arrays: dict[str, object] = {}
    events_meta = []
    base_t = np.datetime64("2005-01-17T00:00:00", "s")
    for i in range(20):
        n = int(rng.integers(30, 60))
        t = np.array([base_t + np.timedelta64(j * 60, "s") for j in range(n)])
        v = np.sin(np.linspace(0, np.pi, n)) + rng.normal(0, 0.1, n)
        arrays[f"t{i}"] = t
        arrays[f"v{i}"] = v
        events_meta.append({"idx": i, "start": str(t[0]), "stop": str(t[-1]), "status": "ok"})

    npz_path = data_dir / "bz_events.npz"
    np.savez_compressed(npz_path, **arrays)

    manifest = {
        "datasets": {
            "bz_events": {
                "kind": "event_collection",
                "file": "bz_events.npz",
                "param_id": "amda/imf_bz",
                "units": "nT",
                "n_events": 20,
                "events": events_meta,
                "source": "test",
                "created": "0",
            }
        }
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Load recipe source from data/recipes/superposed_epoch.py
    recipe_path = Path(__file__).parent.parent / "data" / "recipes" / "superposed_epoch.py"
    recipe_src = recipe_path.read_text(encoding="utf-8")

    # Strip the standalone-demo comment block and append events loading
    setup = "events = load_data('bz_events')\n"
    code = setup + recipe_src

    result = await run_python(code, _plot_dir=str(tmp_path))
    assert result.get("error") is None, result.get("stderr", "")
    assert "epoch_median" in result.get("exports", {}), result
    assert result["exports"]["epoch_median"]["n_finite"] == 100  # n_grid default
    assert result.get("n_figures", 0) >= 1


async def test_api_keys_not_leaked_to_sandbox(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "secret-groq-123")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret-azure-456")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai-789")
    code = (
        "import os\n"
        "print(os.environ.get('GROQ_API_KEY', 'MISSING'))\n"
        "print(os.environ.get('AZURE_OPENAI_API_KEY', 'MISSING'))\n"
        "print(os.environ.get('OPENAI_API_KEY', 'MISSING'))\n"
    )
    result = await run_python(code)
    assert result.get("error") is None, result.get("stderr", "")
    assert "secret-" not in result["stdout"]
    assert result["stdout"].count("MISSING") == 3


async def test_sandbox_runs_as_different_uid() -> None:
    """H1 — sandbox must be PID-isolated from the server.

    Functional bwrap → PID namespace isolation.
    helioai-sandbox user + root → UID isolation via preexec_fn.
    Falls back to skip on dev machines without either.
    """
    from helioai.tools.sandbox import _bwrap_works

    if not _bwrap_works():
        try:
            import pwd

            pwd.getpwnam("helioai-sandbox")
        except (KeyError, ImportError):
            pytest.skip("bwrap not functional and no helioai-sandbox user — H1 not verifiable")

    import os as _os

    server_pid = _os.getpid()
    code = (
        f"import os\n"
        f"try:\n"
        f"    with open('/proc/{server_pid}/environ', 'rb') as f:\n"
        f"        data = f.read(); print('READABLE:' + str(len(data)))\n"
        f"except (PermissionError, FileNotFoundError):\n"
        f"    print('BLOCKED')\n"
    )
    result = await run_python(code)
    assert "BLOCKED" in result["stdout"], (
        f"sandbox can read /proc/{server_pid}/environ — API keys exposed"
    )


async def test_sandbox_cannot_read_server_proc() -> None:
    """H1+H4 — sandbox must not read server /proc/<pid>/environ.

    With bwrap (--unshare-pid) or setuid sandbox user, /proc entries
    from the server's PID namespace are invisible or permission-blocked.
    Skips when neither mechanism is available.
    """
    from helioai.tools.sandbox import _bwrap_works

    if not _bwrap_works():
        try:
            import pwd

            pwd.getpwnam("helioai-sandbox")
        except (KeyError, ImportError):
            pytest.skip("bwrap not functional and no helioai-sandbox user — H1 not verifiable")

    import os as _os

    server_pid = _os.getpid()
    code = (
        f"try:\n"
        f"    with open('/proc/{server_pid}/environ', 'rb') as f:\n"
        f"        data = f.read()\n"
        f"    print('READABLE:' + str(len(data)))\n"
        f"except PermissionError:\n"
        f"    print('BLOCKED')\n"
        f"except FileNotFoundError:\n"
        f"    print('BLOCKED')\n"
    )
    result = await run_python(code)
    assert "BLOCKED" in result["stdout"], (
        f"sandbox can read server /proc/{server_pid}/environ — API keys exposed"
    )


async def test_home_not_leaked_to_sandbox(monkeypatch) -> None:
    """H4 — sandbox HOME must be a scratch dir, never the server's real home."""
    monkeypatch.setenv("HOME", "/home/secret-user")
    code = "import os; print(os.environ.get('HOME', 'MISSING'))"
    result = await run_python(code)
    assert result.get("error") is None, result.get("stderr", "")
    home = result["stdout"].strip()
    assert home != "/home/secret-user", "server HOME leaked into sandbox"
    assert home == "/tmp" or home.startswith("/tmp/"), (
        f"sandbox HOME={home!r}, expected a /tmp-scoped scratch dir"
    )


@pytest.mark.skipif(
    not sandbox._bwrap_works(), reason="bwrap unavailable — no filesystem isolation to assert"
)
async def test_sandbox_cannot_read_secrets(tmp_path, monkeypatch) -> None:
    """H1/H2 — a bwrap sandbox must not read data/ secrets nor the .env file.

    data/ is masked by an empty tmpfs (only this session's workspace is re-bound),
    and .env is masked to /dev/null. Without both, run_python could exfiltrate the
    password hashes (users.json) and the LLM provider API keys (.env).
    """
    from helioai.config import _ROOT, settings

    data_dir = tmp_path / "data"
    plot_dir = data_dir / "users" / "u" / "workspace" / "lbl"
    plot_dir.mkdir(parents=True)
    secret = "SUPER_SECRET_HASH_9f83b2c1"
    (data_dir / "users.json").write_text(f'{{"admin": {{"pw": "{secret}"}}}}')
    monkeypatch.setattr(settings, "data_dir", data_dir)

    users_json = str(data_dir / "users.json")
    env_file = str(_ROOT / ".env")
    code = (
        "def _try(p):\n"
        "    try:\n"
        "        with open(p) as f: return f.read()\n"
        "    except Exception as e: return 'BLOCKED:' + type(e).__name__\n"
        f"print('USERS=' + _try({users_json!r}))\n"
        f"print('ENV=' + _try({env_file!r}))\n"
    )
    result = await run_python(code, _plot_dir=str(plot_dir))
    assert result.get("error") is None, result.get("stderr", "")
    out = result["stdout"]
    assert secret not in out, "password hash readable from sandbox — data/ not masked"
    assert "AZURE" not in out and "API_KEY" not in out, ".env keys readable from sandbox"


async def test_registry_rejects_private_args() -> None:
    from helioai.tools.registry import ToolRegistry

    reg = ToolRegistry()

    @reg.register("echo", "echo", {"type": "object", "properties": {}})
    async def _echo(**kwargs):
        return kwargs

    out = await reg.call_tool("echo", {"x": 1, "_plot_dir": "/etc"})
    assert "rejected private argument" in out
    assert "_plot_dir" in out


async def test_registry_allows_trusted_private_args() -> None:
    from helioai.tools.registry import ToolRegistry

    reg = ToolRegistry()

    @reg.register("echo", "echo", {"type": "object", "properties": {}})
    async def _echo(**kwargs):
        return kwargs

    out = await reg.call_tool("echo", {"x": 1}, trusted={"_plot_dir": "/safe"})
    assert "rejected private argument" not in out
    assert "/safe" in out
