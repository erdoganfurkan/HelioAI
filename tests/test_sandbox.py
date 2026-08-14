"""Tests for helioai.tools.sandbox.run_python."""

from __future__ import annotations

import asyncio
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


async def test_timeout_is_capped_server_side(monkeypatch) -> None:
    """Real bug: an arbitrarily large caller-supplied timeout was honored as-is —
    nothing clamped it before asyncio.wait_for, so a single run could tie up a
    sandbox subprocess far longer than intended."""
    real_wait_for = asyncio.wait_for
    seen_timeouts: list[float] = []

    async def spy_wait_for(aw, timeout=None, **kw):
        seen_timeouts.append(timeout)
        return await real_wait_for(aw, timeout=timeout, **kw)

    monkeypatch.setattr(sandbox.asyncio, "wait_for", spy_wait_for)

    await run_python("print('ok')", timeout=10_000_000.0)

    assert seen_timeouts, "asyncio.wait_for was not called"
    assert seen_timeouts[0] <= sandbox._MAX_TIMEOUT_S


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

    # Load the shipped recipe through settings rather than a hardcoded path, so
    # relocating the recipe directory cannot silently break this test.
    from helioai.config import _PKG_RECIPES

    recipe_src = (_PKG_RECIPES / "superposed_epoch.py").read_text(encoding="utf-8")

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


async def test_physics_helpers_available_in_sandbox() -> None:
    result = await run_python(
        "theta, r = mp_shue1998(2.0, 0.0, theta_deg=0.0)\n"
        "t2, rbs = bs_jelinek2012(2.0, theta_deg=0.0)\n"
        "v = transform_coords('2019-01-01T00:00:00', [10.0, 2.0, 3.0], 'gse', 'gsm')\n"
        "print(round(float(r[0]), 2), round(float(rbs[0]), 2), round(float(v[0]), 2))",
        timeout=120.0,
    )
    assert result.get("error") is None, result.get("stderr", "")
    assert "10.25 13.51 10.0" in result["stdout"]


# ── environment allowlist portability ──────────────────────────────────────────


def test_sandbox_env_passes_windows_essentials(monkeypatch) -> None:
    """Windows needs more than the POSIX set to start a usable interpreter.

    Every sandbox test failed on Windows CI with
    `OSError: [WinError 10106] The requested service provider could not be loaded
    or initialized` — WSAEPROVIDERFAILEDINIT. Winsock cannot find its service
    providers without SYSTEMROOT, and the sandbox preamble imports speasy, which
    touches sockets. Matplotlib also rebuilt its font cache on every run for want
    of LOCALAPPDATA.
    """
    fake = {
        "SYSTEMROOT": r"C:\Windows",
        "SYSTEMDRIVE": "C:",
        "WINDIR": r"C:\Windows",
        "COMSPEC": r"C:\Windows\system32\cmd.exe",
        "PATHEXT": ".COM;.EXE;.BAT",
        "APPDATA": r"C:\Users\me\AppData\Roaming",
        "LOCALAPPDATA": r"C:\Users\me\AppData\Local",
        "USERPROFILE": r"C:\Users\me",
        "NUMBER_OF_PROCESSORS": "8",
        "PROCESSOR_ARCHITECTURE": "AMD64",
    }
    for key, value in fake.items():
        monkeypatch.setenv(key, value)

    env = sandbox._sandbox_env()

    missing = [k for k in fake if k not in env]
    assert not missing, f"these would break the Windows sandbox: {missing}"
    assert env["SYSTEMROOT"] == r"C:\Windows"


def test_sandbox_env_still_strips_secrets(monkeypatch) -> None:
    """Widening the allowlist must not have opened a hole for credentials."""
    for key in (
        "AZURE_OPENAI_API_KEY",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "ADS_API_TOKEN",
        "HELIOAI_DEV_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
    ):
        monkeypatch.setenv(key, "s3cret")

    env = sandbox._sandbox_env()

    leaked = [k for k, v in env.items() if v == "s3cret"]
    assert not leaked, f"secrets reachable from generated code: {leaked}"


def test_sandbox_env_pins_the_matplotlib_cache_under_home(tmp_path) -> None:
    """matplotlib ignores XDG on Windows, so MPLCONFIGDIR must be explicit.

    Otherwise it falls back to a fresh temporary directory and rebuilds the font
    cache on every run — the "Matplotlib created a temporary cache directory"
    line that preceded every Windows CI failure.
    """
    env = sandbox._sandbox_env(home=str(tmp_path))
    assert env["MPLCONFIGDIR"].startswith(str(tmp_path))
    assert env["XDG_CACHE_HOME"].startswith(str(tmp_path))


async def test_sandbox_runs_inside_the_session_workspace(tmp_path, monkeypatch):
    """Agent code must be able to write files, and they must land in the workspace.

    Regression: the cwd was inherited from the server — the repo root, which bwrap
    mounts read-only — so `open(..., "w")` failed with EROFS. Act VI of examples/02
    asks the agent to save a standalone script and could never succeed, whatever
    the prompt or the token budget. On the non-bwrap fallback there is no read-only
    mount at all, so the same cwd meant agent code could write into the repo.
    """
    import helioai.workspace as ws
    from helioai.tools.sandbox import run_python

    ws.set_user("t_cwd")
    ws.set_session("s_cwd")

    result = await run_python(
        'import os\nprint(os.getcwd())\nopen("artifact.txt", "w").write("written by agent code")\n'
    )

    assert not result.get("error"), result
    session_dir = ws.get_session_dir()
    assert (session_dir / "artifact.txt").read_text() == "written by agent code"
    assert str(session_dir) in (result.get("stdout") or "")


@pytest.mark.asyncio
async def test_traceback_line_points_at_the_agents_own_code():
    """A traceback numbered from the assembled script is unusable to the agent.

    The preamble sits ~212 lines above the snippet, so an undefined name on user line 3
    was reported as line 215 — and code_N.py on disk holds only the user's lines.
    """
    result = await run_python("x = 1\ny = 2\nprint(x[nope])")
    assert "error" in result
    assert "NameError" in result["error"], result["error"]
    assert 'File "your code", line 3' in result["stderr"], result["stderr"]
    assert "line 21" not in result["stderr"]


@pytest.mark.asyncio
async def test_error_field_names_the_exception_not_the_exit_code():
    result = await run_python("raise ValueError('bad input')")
    assert result["error"] == "ValueError: bad input"


@pytest.mark.asyncio
async def test_preamble_offset_is_derived_not_hard_coded():
    """If the preamble grows and the offset does not, tracebacks silently drift again."""
    from helioai.tools.sandbox import _PREAMBLE_LINES, _SANDBOX_PREAMBLE

    assert _PREAMBLE_LINES == 2 + len(_SANDBOX_PREAMBLE.splitlines())


@pytest.mark.asyncio
async def test_failed_run_still_reports_the_script_path():
    result = await run_python("raise RuntimeError('boom')")
    assert result["code_path"].endswith(".py")
    assert Path(result["code_path"]).read_text().strip() == "raise RuntimeError('boom')"


@pytest.mark.asyncio
async def test_interp_to_handles_the_two_ways_hand_rolling_it_fails():
    """Both failures came from one Act II run, on consecutive turns.

    `np.timedelta64` has no `.total_seconds()`, and `np.interp` on a 3-component field
    raises "object too deep for desired array".
    """
    code = """
import numpy as np
t1 = np.array(['2015-03-17T00:00:00','2015-03-17T00:01:00','2015-03-17T00:02:00'], dtype='datetime64[s]')
t2 = np.array(['2015-03-17T00:00:30','2015-03-17T00:01:30'], dtype='datetime64[s]')
export('scalar', interp_to(t2, t1, np.array([10., 20., 30.])))
export('vector', interp_to(t2, t1, np.array([[1.,2.,3.],[2.,4.,6.],[3.,6.,9.]])))
export('gap', interp_to(t2, t1, np.array([1.0, np.nan, 3.0])))
export('outside', interp_to(np.array(['2015-03-17T09:00:00'], dtype='datetime64[s]'), t1,
                            np.array([10., 20., 30.])))
"""
    result = await run_python(code)
    assert result.get("error") is None, result.get("stderr", "")
    ex = result["exports"]
    assert ex["scalar"]["mean"] == pytest.approx(20.0)
    assert ex["vector"]["shape"] == [2, 3]
    assert ex["gap"]["n_finite"] == 0, "a gap must not be bridged — that invents data"
    assert ex["outside"]["n_finite"] == 0, "outside the source range is NaN, not clamped"


@pytest.mark.asyncio
async def test_save_path_writes_inside_the_session_directory(tmp_path):
    """A hand-built path one level up from the session directory writes to a bwrap
    tmpfs overlay: the write does not raise, stdout can claim success, and the file
    is gone the moment the sandbox process exits. `save_path` must resolve inside
    the directory this run actually has bound writable.
    """
    code = """
p = save_path('standalone.py')
open(p, 'w').write('# hello')
print(p)
"""
    result = await run_python(code, _plot_dir=str(tmp_path), _run_idx=0)
    assert result.get("error") is None, result.get("stderr", "")
    written = tmp_path / "standalone.py"
    assert written.is_file(), result["stdout"]
    assert written.read_text() == "# hello"
    assert result["stdout"].strip() == str(written)


@pytest.mark.asyncio
async def test_magnitude_leaves_a_data_gap_as_a_gap():
    """The hand-written idiom turned a 90 s hole in Wind/MFI into |B| = 0 nT.

    `np.diff` then read the recovery out of that hole as the largest jump of the
    interval, and the shock detector of Act IV locked onto it — publishing a shock
    time 3.5 minutes early and a spacecraft lag of -464.5 s instead of -260.5 s.
    """
    code = """
import numpy as np
b = np.array([[3., 4., 0.], [np.nan, np.nan, np.nan], [6., 8., 0.]])
export('good', magnitude(b))
export('n_finite', np.array([np.sum(np.isfinite(magnitude(b)))]))
export('gap_is_nan', np.array([1.0 if np.isnan(magnitude(b)[1]) else 0.0]))
export('nansum_idiom_gives_zero', np.array([np.sqrt(np.nansum(b[1]**2))]))
export('fill_blanked', magnitude(np.array([[1e31, 1e31, 1e31]])))
"""
    result = await run_python(code)
    assert result.get("error") is None, result.get("stderr", "")
    ex = result["exports"]
    assert ex["good"]["min"] == pytest.approx(5.0)
    assert ex["good"]["max"] == pytest.approx(10.0)
    assert ex["n_finite"]["mean"] == 2
    assert ex["gap_is_nan"]["mean"] == 1.0
    assert ex["nansum_idiom_gives_zero"]["mean"] == 0.0, "this is the trap being replaced"
    assert ex["fill_blanked"]["n_finite"] == 0, "1e31 fill must not become a magnitude"


def test_a_fresh_sandbox_home_gets_the_hosts_speasy_inventory(tmp_path, monkeypatch):
    """Regression: a session could burn every run_python without executing user code.

    The sandbox hands each session a new HOME, so speasy found no inventory and
    rebuilt it — longer than the run timeout, so the spawn was killed mid-build, so
    the index stayed incomplete, so the next spawn started over. Five run_python calls
    were lost to it in one Act, including a bare `print("hello")` at 30 s.
    """
    from helioai.tools.sandbox import _seed_speasy_inventory

    host = tmp_path / "host"
    (host / "speasy" / "index").mkdir(parents=True)
    (host / "speasy" / "index" / "cache.db").write_text("inventory", encoding="utf-8")
    monkeypatch.setenv("XDG_DATA_HOME", str(host))

    home = tmp_path / "session"
    home.mkdir()
    _seed_speasy_inventory(str(home))

    assert (home / ".local" / "share" / "speasy" / "index" / "cache.db").read_text(
        encoding="utf-8"
    ) == "inventory"


def test_seeding_never_overwrites_an_inventory_the_session_already_built(tmp_path, monkeypatch):
    """The session's own index is the fresher one; clobbering it would lose its warm state."""
    from helioai.tools.sandbox import _seed_speasy_inventory

    host = tmp_path / "host"
    (host / "speasy").mkdir(parents=True)
    (host / "speasy" / "cache.db").write_text("host", encoding="utf-8")
    monkeypatch.setenv("XDG_DATA_HOME", str(host))

    home = tmp_path / "session"
    own = home / ".local" / "share" / "speasy"
    own.mkdir(parents=True)
    (own / "cache.db").write_text("session", encoding="utf-8")

    _seed_speasy_inventory(str(home))
    assert (own / "cache.db").read_text(encoding="utf-8") == "session"


def test_seeding_is_silent_when_the_host_has_no_inventory(tmp_path, monkeypatch):
    """No host inventory is the fresh-install case: pay the rebuild, never raise."""
    from helioai.tools.sandbox import _seed_speasy_inventory

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "nothing-here"))
    home = tmp_path / "session"
    home.mkdir()
    _seed_speasy_inventory(str(home))  # must not raise
    assert not (home / ".local" / "share" / "speasy").exists()


async def test_speasy_is_not_imported_until_it_is_used() -> None:
    """The preamble imported speasy at every spawn, so `print("hello")` needed CDAWeb.

    That is the structural reason 23 sandbox tests cascaded into timeouts on one slow
    CI runner (2026-08-13) and the same shape as the 2026-07-17 incident. Asserting on
    sys.modules rather than on elapsed time keeps this deterministic on a loaded runner
    — the exact condition that made the old failure look like a code regression.
    """
    result = await run_python("import sys; print('speasy' in sys.modules)")
    assert result.get("error") is None
    assert result["stdout"] == "False"


async def test_spz_still_resolves_when_actually_touched() -> None:
    """Laziness must not break the escape hatch: spz.<attr> still reaches speasy."""
    result = await run_python("print(type(spz.get_data).__name__)")
    assert result.get("error") is None
    assert result["stdout"] in {"function", "method"}


def test_reports_when_the_fallback_path_isolates_nothing(monkeypatch):
    """_drop_privileges swallows its failure in the forked child, where it must.

    The consequence was that a host with neither bubblewrap nor root ran
    model-written code as the server, with nothing said anywhere. The condition is
    knowable in the parent, so it is asserted in the parent.
    """
    import helioai.tools.sandbox as sb

    # raising=False: os.geteuid does not exist on Windows, and monkeypatch refuses to
    # set an absent attribute — which failed the Windows job while the code under test
    # was fine, since it checks hasattr first.
    monkeypatch.setattr(sb.os, "geteuid", lambda: 1000, raising=False)
    assert "not root" in (sb._isolation_gap() or "")


def test_isolation_gap_answers_on_a_platform_with_no_privilege_model(monkeypatch):
    """Windows has no os.geteuid. The check must answer, not raise — asserted here
    rather than discovered on the one CI job that is allowed to fail."""
    import helioai.tools.sandbox as sb

    monkeypatch.delattr(sb.os, "geteuid", raising=False)
    assert "POSIX" in (sb._isolation_gap() or "")


def test_the_isolation_warning_is_said_once_not_per_run(monkeypatch):
    """Every run_python on a dev box takes this path — a per-run warning is noise."""
    import helioai.tools.sandbox as sb

    monkeypatch.setattr(sb.os, "geteuid", lambda: 1000, raising=False)
    sb._warn_if_not_isolated.cache_clear()
    try:
        for _ in range(3):
            sb._warn_if_not_isolated()
        assert sb._warn_if_not_isolated.cache_info().misses == 1
    finally:
        sb._warn_if_not_isolated.cache_clear()


def test_the_fallback_warns_even_when_privileges_do_drop(monkeypatch):
    """Reaching the fallback means no bwrap, so no filesystem isolation — say it.

    The Docker image is precisely this case: root, with the helioai-sandbox user
    present, so the privilege drop works and the earlier version of this warning
    stayed silent about the isolation that is missing.
    """
    import helioai.logging_config as lc
    import helioai.tools.sandbox as sb

    monkeypatch.setattr(sb, "_isolation_gap", lambda: None)
    said = []

    class _Rec:
        def warning(self, event, **kw):
            said.append((event, kw))

    monkeypatch.setattr(lc, "get_logger", lambda _n: _Rec())

    sb._warn_if_not_isolated.cache_clear()
    try:
        sb._warn_if_not_isolated()
    finally:
        sb._warn_if_not_isolated.cache_clear()

    assert said, "the fallback path said nothing at all"
    event, kw = said[0]
    assert event == "sandbox_not_isolated"
    assert "filesystem isolation" in kw["detail"]
