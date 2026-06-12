"""Tests for helioai.tools.sandbox.run_python."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

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
