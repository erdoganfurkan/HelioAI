from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.recipes


SHOCK_TIME = np.datetime64("2015-03-17T04:00:00", "s")


def _shock_series(*, nan_upstream: bool = False):
    t = np.datetime64("2015-03-17T03:30:00", "s") + np.arange(1201) * np.timedelta64(3, "s")
    theta = np.radians(60.0)
    b_up = 10.0 * np.array([np.cos(theta), 0.0, np.sin(theta)])
    b_dn = np.array([b_up[0], 0.0, 2.5 * b_up[2]])
    values = np.where((t < SHOCK_TIME)[:, None], b_up, b_dn).astype(float)

    values[t == np.datetime64("2015-03-17T03:53:00", "s")] = np.nan
    values[t == np.datetime64("2015-03-17T04:06:00", "s")] = np.nan
    if nan_upstream:
        u0 = np.datetime64("2015-03-17T03:50:00", "s")
        u1 = np.datetime64("2015-03-17T03:58:00", "s")
        values[(u0 <= t) & (t <= u1)] = np.nan

    return SimpleNamespace(time=t, values=values), b_up, b_dn


def _smooth_ramp_series():
    series, b_up, b_dn = _shock_series()
    ramp_start = np.datetime64("2015-03-17T04:00:00", "s")
    ramp_stop = np.datetime64("2015-03-17T04:02:00", "s")
    ramp = (ramp_start <= series.time) & (series.time <= ramp_stop)
    frac = ((series.time[ramp] - ramp_start) / (ramp_stop - ramp_start)).astype(float)
    series.values[series.time < ramp_start] = b_up
    series.values[series.time > ramp_stop] = b_dn
    series.values[ramp] = b_up + frac[:, None] * (b_dn - b_up)
    return series


def _ar1_noise(n, rng, scale=0.5, rho=0.8):
    noise = np.zeros((n, 3))
    innovation_scale = scale * np.sqrt(1.0 - rho**2)
    noise[0] = rng.normal(0.0, scale, 3)
    for i in range(1, n):
        noise[i] = rho * noise[i - 1] + rng.normal(0.0, innovation_scale, 3)
    return noise


def test_theta_bn_derives_windows_from_a_shock_time(recipe, capsys):
    series, b_up, b_dn = _shock_series()

    run = recipe("theta_bn", B=series, shock_time=SHOCK_TIME)

    assert run.value("theta_bn") == pytest.approx([60.0], abs=0.1)
    assert run.value("compression_ratio") == pytest.approx(
        [np.linalg.norm(b_dn) / np.linalg.norm(b_up)]
    )
    assert run.value("shock_normal").shape == (3,)
    assert run.value("B_up_mean_nT") == pytest.approx(b_up)
    assert run.value("B_dn_mean_nT") == pytest.approx(b_dn)
    assert run.exports["shock_normal"]["units"] == ""
    assert run.exports["B_up_mean_nT"]["units"] == "nT"
    stdout = capsys.readouterr().out
    assert "upstream window: 2015-03-17T03:50:00 to 2015-03-17T03:58:00" in stdout
    assert "downstream window: 2015-03-17T04:02:00 to 2015-03-17T04:10:00" in stdout


def test_theta_bn_refuses_a_smooth_ramp_trend_inside_a_window(recipe, capsys):
    series = _smooth_ramp_series()

    run = recipe("theta_bn", B=series, shock_time=np.datetime64("2015-03-17T04:07:00", "s"))

    assert "theta_bn" not in run.exports
    stdout = capsys.readouterr().out
    assert "window is not stationary" in stdout
    assert "it likely contains the ramp" in stdout


def test_theta_bn_refuses_a_window_that_overlaps_the_ramp(recipe, capsys):
    series, _, _ = _shock_series()

    run = recipe("theta_bn", B=series, shock_time=np.datetime64("2015-03-17T04:05:00", "s"))

    assert "theta_bn" not in run.exports
    assert "upstream window overlaps the ramp" in capsys.readouterr().out


def test_theta_bn_refuses_windows_with_too_few_finite_rows(recipe, capsys):
    series, _, _ = _shock_series(nan_upstream=True)

    run = recipe("theta_bn", B=series, shock_time=SHOCK_TIME)

    assert "theta_bn" not in run.exports
    assert "upstream window has only 0 finite rows" in capsys.readouterr().out


def test_theta_bn_exports_series_diagnostics(recipe):
    series, _, _ = _shock_series()

    run = recipe("theta_bn", B=series, shock_time=SHOCK_TIME)

    assert "theta_bn_std_deg" in run.exports
    assert "normal_spread_deg" in run.exports
    assert "Bn_std_nT" in run.exports
    assert run.exports["theta_bn_std_deg"]["units"] == "deg"
    assert run.exports["normal_spread_deg"]["units"] == "deg"
    assert run.exports["Bn_std_nT"]["units"] == "nT"
    assert run.value("theta_bn_std_deg")[0] < 0.1
    assert run.value("normal_spread_deg")[0] < 0.1
    assert run.value("Bn_std_nT")[0] < 1e-10


def test_theta_bn_bootstrap_diagnostics_are_nonzero_on_correlated_noise(recipe):
    theta = np.radians(60.0)
    b_up = 10.0 * np.array([np.cos(theta), 0.0, np.sin(theta)])
    b_dn = np.array([b_up[0], 0.0, 2.5 * b_up[2]])
    rng = np.random.default_rng(2)
    up = b_up + _ar1_noise(240, rng)
    dn = b_dn + _ar1_noise(240, rng)

    run = recipe("theta_bn", B_up=up, B_dn=dn)

    theta_std = run.value("theta_bn_std_deg")[0]
    bn_std = run.value("Bn_std_nT")[0]
    assert 0.0 < theta_std < 3.0
    assert bn_std > 0.0


def test_theta_bn_omits_series_diagnostics_for_mean_vectors(recipe):
    theta = np.radians(60.0)
    b_up = 10.0 * np.array([np.cos(theta), 0.0, np.sin(theta)])
    b_dn = np.array([b_up[0], 0.0, 2.5 * b_up[2]])

    run = recipe("theta_bn", B_up=b_up, B_dn=b_dn)

    assert run.value("theta_bn") == pytest.approx([60.0], abs=0.1)
    assert "theta_bn_std_deg" not in run.exports
    assert "normal_spread_deg" not in run.exports
    assert "Bn_std_nT" not in run.exports


def test_theta_bn_without_inputs_prints_instructions_and_exports_nothing(recipe, capsys):
    run = recipe("theta_bn")

    assert run.exports == {}
    assert "define B_up and B_dn, or B and shock_time" in capsys.readouterr().out


def test_theta_bn_placeholder_demo_runs_only_as_a_bare_script(recipe):
    run = recipe("theta_bn", __name__="__main__")

    assert run.value("theta_bn") == pytest.approx([80.24], abs=0.01)
