"""Tests for helioai.tools.sandbox.run_python."""

from __future__ import annotations

import os

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
