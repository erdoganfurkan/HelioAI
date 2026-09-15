from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from tests.recipes.conftest import RECIPES_DIR

pytestmark = pytest.mark.recipes


def _anisotropic_cloud(n=300):
    rng = np.random.default_rng(7)
    return np.column_stack(
        [
            rng.normal(0, 10, n),
            rng.normal(0, np.sqrt(10), n),
            rng.normal(0, np.sqrt(0.1), n),
        ]
    )


def _orthogonal_four_point_cloud(variances):
    sx, sy, sz = np.sqrt(variances)
    return np.array(
        [
            [sx, sy, sz],
            [sx, -sy, -sz],
            [-sx, sy, -sz],
            [-sx, -sy, sz],
        ]
    )


def test_mvab_normal_uncertainty_matches_sonnerup_scheible(recipe):
    mvab = recipe("mvab").namespace["mvab"]
    b = _anisotropic_cloud()
    result = mvab(b)

    lam1 = result["lambda_max"]
    lam2 = result["lambda_int"]
    lam3 = result["lambda_min"]
    expected_rad = np.sqrt(lam3 / (300 - 1) * lam2 / (lam2 - lam3) ** 2)
    expected_max_rad = np.sqrt(lam3 / (300 - 1) * lam1 / (lam1 - lam3) ** 2)
    b_mean = b.mean(axis=0)
    expected_dBn = np.sqrt(
        lam3 / (300 - 1)
        + (expected_rad * np.dot(b_mean, result["n_int"])) ** 2
        + (expected_max_rad * np.dot(b_mean, result["n_max"])) ** 2
    )

    assert result["dphi_min_int_deg"] < 2.0
    assert result["dphi_min_int_deg"] == pytest.approx(np.degrees(expected_rad), rel=1e-6)
    assert result["dphi_min_max_deg"] == pytest.approx(
        np.degrees(expected_max_rad),
        rel=1e-6,
    )
    assert result["dBn_nT"] == pytest.approx(expected_dBn, rel=1e-6)


def test_mvab_uses_sonnerup_scheible_covariance_for_normal_component_uncertainty(recipe):
    mvab = recipe("mvab").namespace["mvab"]
    b = _orthogonal_four_point_cloud(np.array([100.0, 10.0, 1.0]))
    result = mvab(b)

    centered = b - b.mean(axis=0)
    covariance = centered.T @ centered / len(b)
    lam3 = np.linalg.eigvalsh(covariance)[0]
    expected = np.sqrt(lam3 / (len(b) - 1))

    assert result["dBn_nT"] == pytest.approx(expected, abs=1e-6)


def test_mvab_warns_when_fewer_than_thirty_samples(recipe):
    mvab = recipe("mvab").namespace["mvab"]
    result = mvab(_anisotropic_cloud(n=12))

    assert result["warning"] == "fewer than 30 samples — eigenvalue ratios unreliable"


def test_mvab_reports_a_planar_cloud_as_unique_but_not_estimable(recipe):
    mvab = recipe("mvab").namespace["mvab"]
    result = mvab(_orthogonal_four_point_cloud(np.array([100.0, 10.0, 0.0])))

    assert result["lambda_min"] == pytest.approx(0.0, abs=1e-12)
    assert result["quality"].startswith("planar")
    assert np.isnan(result["dphi_min_int_deg"])
    assert np.isnan(result["dphi_min_max_deg"])


def test_mvab_reports_repeated_smallest_eigenvalues_as_degenerate(recipe):
    mvab = recipe("mvab").namespace["mvab"]
    result = mvab(_orthogonal_four_point_cloud(np.array([100.0, 0.0, 0.0])))

    assert result["lambda_min"] == pytest.approx(0.0, abs=1e-12)
    assert result["quality"].startswith("degenerate")
    assert np.isnan(result["dphi_min_int_deg"])


def test_mvab_returns_an_error_for_non_finite_input(recipe):
    mvab = recipe("mvab").namespace["mvab"]
    b = _anisotropic_cloud()
    b[4, 1] = np.nan

    result = mvab(b)

    assert result == {"error": "B must contain only finite values"}


def test_mvab_run_block_only_exports_when_b_is_bound(recipe):
    assert recipe("mvab").exports == {}

    run = recipe("mvab", B=_anisotropic_cloud())

    assert run.exports["mvab_ratio_int_min"]["units"] == ""
    assert run.exports["mvab_lambda_min"]["units"] == "nT2"
    assert run.exports["mvab_dphi_min_int"]["units"] == "deg"
    assert run.exports["mvab_dphi_min_max"]["units"] == "deg"
    assert run.exports["mvab_dBn"]["units"] == "nT"
    assert run.exports["mvab_normal"]["units"] == ""
    assert run.value("mvab_normal").shape == (3,)


def test_mvab_standalone_demo_runs_as_a_script():
    completed = subprocess.run(
        [sys.executable, str(RECIPES_DIR / "mvab.py")],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
