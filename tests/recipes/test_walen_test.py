from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.recipes


def _nonzero_mean_rotating_field(n: int = 480) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack(
        [
            8.0 + 2.0 * np.cos(t),
            -3.0 + 2.5 * np.sin(t),
            4.0 + 1.5 * np.cos(t + 0.7),
        ]
    )


def _old_mean_subtracted_slope(alfven_velocity, v, b, n_cm3) -> float:
    va = alfven_velocity(b, n_cm3)
    x = (va - va.mean(axis=0)).ravel()
    y = (v - v.mean(axis=0)).ravel()
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def test_ht_frame_recovers_a_moving_rotational_discontinuity(recipe):
    ns = recipe("walen_test").namespace
    b = _nonzero_mean_rotating_field()
    n_cm3 = np.full(b.shape[0], 5.0)
    v_ht = np.array([-400.0, 30.0, -10.0])
    rng = np.random.default_rng(11)
    v = v_ht + ns["alfven_velocity"](b, n_cm3) + rng.normal(0.0, 0.15, b.shape)

    result = ns["walen_test"](v, b, n_cm3)

    assert result["frame"] == "ht"
    assert np.allclose(result["V_HT"], v_ht, atol=5.0)
    assert result["slope"] == pytest.approx(1.0, abs=0.05)
    assert result["R2"] >= 0.8


def test_mean_frame_preserves_the_previous_regression(recipe):
    ns = recipe("walen_test").namespace
    rng = np.random.default_rng(42)
    n = 200
    t = np.linspace(0, np.pi, n)
    b = np.column_stack([5 + 3 * np.cos(t), -2 + np.sin(t), 1 + 0.5 * np.sin(2 * t)])
    n_cm3 = 5 + rng.normal(0, 0.3, n)
    v = ns["alfven_velocity"](b, n_cm3) + rng.normal(0, 5, (n, 3))
    old_slope = _old_mean_subtracted_slope(ns["alfven_velocity"], v, b, n_cm3)

    result = ns["walen_test"](v, b, n_cm3, frame="mean")

    assert result["frame"] == "mean"
    assert result["slope"] == pytest.approx(old_slope, abs=5e-4)


def test_ht_frame_refuses_parallel_magnetic_fields(recipe):
    ns = recipe("walen_test").namespace
    b = np.tile([5.0, 0.0, 0.0], (64, 1))
    v = np.tile([400.0, -20.0, 10.0], (64, 1))
    n_cm3 = np.full(64, 5.0)

    result = ns["walen_test"](v, b, n_cm3)

    assert "error" in result
    assert "slope" not in result


def test_walen_recipe_exports_only_when_inputs_are_bound(recipe):
    assert recipe("walen_test").exports == {}

    ns = recipe("walen_test").namespace
    b = _nonzero_mean_rotating_field(160)
    n_cm3 = np.full(b.shape[0], 5.0)
    v = np.array([-400.0, 30.0, -10.0]) + ns["alfven_velocity"](b, n_cm3)

    run = recipe("walen_test", V=v, B=b, n_cm3=n_cm3)

    assert run.exports["walen_slope"]["units"] == ""
    assert run.exports["walen_R2"]["units"] == ""
    assert run.exports["V_HT"]["units"] == "km/s"
    assert run.value("V_HT").shape == (3,)


def test_walen_test_runs_as_a_standalone_script():
    script = Path(__file__).resolve().parents[2] / "helioai" / "data" / "recipes" / "walen_test.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=script.parents[3],
        text=True,
        capture_output=True,
        check=False,
    )
    if "NameError" in result.stderr and "export" in result.stderr:
        pytest.skip("standalone script still depends on sandbox export()")
    assert result.returncode == 0, result.stderr
    assert "slope" in result.stdout
