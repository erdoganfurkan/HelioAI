from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.recipes

MU0 = 4 * np.pi * 1e-7
MP = 1.6726e-27
CM3_TO_M3 = 1e6


def _nonzero_mean_rotating_field(n: int = 480) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack(
        [
            8.0 + 2.0 * np.cos(t),
            -3.0 + 2.5 * np.sin(t),
            4.0 + 1.5 * np.cos(t + 0.7),
        ]
    )


def _alfven_velocity(b_nT, n_cm3):
    b_T = np.asarray(b_nT, dtype=float) * 1e-9
    n_m3 = np.asarray(n_cm3, dtype=float) * CM3_TO_M3
    return b_T / np.sqrt(MU0 * n_m3[:, None] * MP) / 1e3


def _old_mean_subtracted_slope(v, b, n_cm3) -> float:
    va = _alfven_velocity(b, n_cm3)
    x = (va - va.mean(axis=0)).ravel()
    y = (v - v.mean(axis=0)).ravel()
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _velocity_with_target_slope_and_r2(b, n_cm3, slope, r2):
    va = _alfven_velocity(b, n_cm3)
    x = va - va.mean(axis=0)
    rng = np.random.default_rng(17)
    noise = rng.normal(0.0, 1.0, b.shape)
    noise -= noise.mean(axis=0)
    x_flat = x.ravel()
    noise_flat = noise.ravel()
    noise_flat -= np.dot(noise_flat, x_flat) / np.dot(x_flat, x_flat) * x_flat
    noise = noise_flat.reshape(b.shape)
    noise -= noise.mean(axis=0)
    noise_flat = noise.ravel()
    noise_flat -= np.dot(noise_flat, x_flat) / np.dot(x_flat, x_flat) * x_flat
    noise = noise_flat.reshape(b.shape)
    target_noise_norm = abs(slope) * np.sqrt(np.dot(x_flat, x_flat) * (1.0 / r2 - 1.0))
    noise *= target_noise_norm / np.linalg.norm(noise)
    return slope * va + noise


def test_independent_alfven_velocity_helper_has_the_expected_normalisation():
    va = _alfven_velocity(np.array([[5.0, 0.0, 0.0]]), np.array([5.0]))
    assert va[0, 0] == pytest.approx(48.77, abs=0.05)


def test_ht_frame_recovers_a_moving_rotational_discontinuity(recipe):
    ns = recipe("walen_test").namespace
    b = _nonzero_mean_rotating_field()
    n_cm3 = np.full(b.shape[0], 5.0)
    v_ht = np.array([-400.0, 30.0, -10.0])
    rng = np.random.default_rng(11)
    v = v_ht + _alfven_velocity(b, n_cm3) + rng.normal(0.0, 0.15, b.shape)

    result = ns["walen_test"](v, b, n_cm3)

    assert result["frame"] == "ht"
    assert np.allclose(result["V_HT"], v_ht, atol=5.0)
    assert result["slope"] == pytest.approx(1.0, abs=0.05)
    assert result["R2"] >= 0.8
    assert result["ht_correlation"] > 0.99
    assert result["ht_residual"] < 0.05
    assert set(result["component_slopes"]) == {"x", "y", "z"}
    assert all(s == pytest.approx(1.0, abs=0.05) for s in result["component_slopes"].values())


def test_negative_walen_slope_is_still_a_rotational_discontinuity(recipe):
    ns = recipe("walen_test").namespace
    b = _nonzero_mean_rotating_field()
    n_cm3 = np.full(b.shape[0], 5.0)
    v = np.array([-400.0, 30.0, -10.0]) - _alfven_velocity(b, n_cm3)

    result = ns["walen_test"](v, b, n_cm3)

    assert result["slope"] == pytest.approx(-1.0, abs=0.02)
    assert result["R2"] == pytest.approx(1.0, abs=1e-4)
    assert (
        result["interpretation"]
        == "consistent with a rotational discontinuity (Walén relation satisfied)"
    )


def test_super_alfvenic_slope_is_not_called_a_rotational_discontinuity(recipe):
    ns = recipe("walen_test").namespace
    b = _nonzero_mean_rotating_field()
    n_cm3 = np.full(b.shape[0], 5.0)
    v = np.array([-400.0, 30.0, -10.0]) + 10.0 * _alfven_velocity(b, n_cm3)

    result = ns["walen_test"](v, b, n_cm3)

    assert result["slope"] == pytest.approx(10.0, abs=0.02)
    assert result["R2"] == pytest.approx(1.0, abs=1e-4)
    assert not result["interpretation"].startswith("consistent with a rotational discontinuity")
    assert "super-Alfvénic correlation" in result["interpretation"]


def test_high_slope_with_low_correlation_is_not_a_rotational_discontinuity(recipe):
    ns = recipe("walen_test").namespace
    b = _nonzero_mean_rotating_field()
    n_cm3 = np.full(b.shape[0], 5.0)
    v = _velocity_with_target_slope_and_r2(b, n_cm3, slope=1.0, r2=0.3)

    result = ns["walen_test"](v, b, n_cm3, frame="mean")

    assert result["slope"] == pytest.approx(1.0, abs=0.02)
    assert result["R2"] == pytest.approx(0.3, abs=0.02)
    assert result["interpretation"] == "not Alfvénic"


def test_partial_slope_also_requires_correlation_quality(recipe):
    ns = recipe("walen_test").namespace
    b = _nonzero_mean_rotating_field()
    n_cm3 = np.full(b.shape[0], 5.0)
    v = _velocity_with_target_slope_and_r2(b, n_cm3, slope=0.5, r2=0.3)

    result = ns["walen_test"](v, b, n_cm3, frame="mean")

    assert result["slope"] == pytest.approx(0.5, abs=0.02)
    assert result["R2"] == pytest.approx(0.3, abs=0.02)
    assert result["interpretation"] == "not Alfvénic"


def test_poor_ht_frame_warns_that_the_walen_slope_is_not_meaningful(recipe):
    ns = recipe("walen_test").namespace
    b = _nonzero_mean_rotating_field()
    n_cm3 = np.full(b.shape[0], 5.0)
    rng = np.random.default_rng(23)
    v = np.array([-400.0, 30.0, -10.0]) + _alfven_velocity(b, n_cm3)
    v += rng.normal(0.0, 600.0, b.shape)

    result = ns["walen_test"](v, b, n_cm3)

    assert result["ht_correlation"] < 0.9
    assert "HT frame is poor" in result["interpretation"]
    assert "Walén slope not meaningful" in result["interpretation"]


def test_mean_frame_preserves_the_previous_regression(recipe):
    ns = recipe("walen_test").namespace
    rng = np.random.default_rng(42)
    n = 200
    t = np.linspace(0, np.pi, n)
    b = np.column_stack([5 + 3 * np.cos(t), -2 + np.sin(t), 1 + 0.5 * np.sin(2 * t)])
    n_cm3 = 5 + rng.normal(0, 0.3, n)
    v = _alfven_velocity(b, n_cm3) + rng.normal(0, 5, (n, 3))
    old_slope = _old_mean_subtracted_slope(v, b, n_cm3)

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

    b = _nonzero_mean_rotating_field(160)
    n_cm3 = np.full(b.shape[0], 5.0)
    v = np.array([-400.0, 30.0, -10.0]) + _alfven_velocity(b, n_cm3)

    run = recipe("walen_test", V=v, B=b, n_cm3=n_cm3)

    assert run.exports["walen_slope"]["units"] == ""
    assert run.exports["walen_R2"]["units"] == ""
    assert run.exports["V_HT"]["units"] == "km/s"
    assert run.exports["ht_residual"]["units"] == ""
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
    assert result.returncode == 0, result.stderr
    assert "slope" in result.stdout
