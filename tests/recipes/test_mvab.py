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


def test_mvab_warns_when_fewer_than_thirty_samples(recipe):
    mvab = recipe("mvab").namespace["mvab"]
    result = mvab(_anisotropic_cloud(n=12))

    assert result["warning"] == "fewer than 30 samples — eigenvalue ratios unreliable"


def test_mvab_classifies_numerically_zero_minimum_variance_as_degenerate(recipe):
    mvab = recipe("mvab").namespace["mvab"]
    rng = np.random.default_rng(3)
    b = np.column_stack(
        [
            rng.normal(0, 10, 200),
            rng.normal(0, np.sqrt(10), 200),
            np.zeros(200),
        ]
    )

    result = mvab(b)

    assert result["lambda_min"] == pytest.approx(0.0, abs=1e-12)
    assert result["quality"].startswith("degenerate")
    assert not result["quality"].startswith("well-determined")


def test_mvab_run_block_only_exports_when_b_is_bound(recipe):
    assert recipe("mvab").exports == {}

    run = recipe("mvab", B=_anisotropic_cloud())

    assert run.exports["mvab_ratio_int_min"]["units"] == ""
    assert run.exports["mvab_lambda_min"]["units"] == "nT2"
    assert run.exports["mvab_dphi_min_int"]["units"] == "deg"
    assert run.exports["mvab_dphi_min_max"]["units"] == "deg"
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
