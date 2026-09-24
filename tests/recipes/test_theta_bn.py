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
        u0 = np.datetime64("2015-03-17T03:45:00", "s")
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
    assert "upstream window: 2015-03-17T03:45:00 to 2015-03-17T03:58:00" in stdout
    assert "downstream window: 2015-03-17T04:02:00 to 2015-03-17T04:15:00" in stdout


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

    assert "theta_bn_sampling_std_deg" in run.exports
    assert "normal_spread_deg" in run.exports
    assert "Bn_std_nT" in run.exports
    assert run.exports["theta_bn_sampling_std_deg"]["units"] == "deg"
    assert run.exports["normal_spread_deg"]["units"] == "deg"
    assert run.exports["Bn_std_nT"]["units"] == "nT"
    assert run.value("theta_bn_sampling_std_deg")[0] < 0.1
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

    theta_std = run.value("theta_bn_sampling_std_deg")[0]
    bn_std = run.value("Bn_std_nT")[0]
    assert 0.0 < theta_std < 3.0
    assert bn_std > 0.0


def test_theta_bn_omits_series_diagnostics_for_mean_vectors(recipe):
    theta = np.radians(60.0)
    b_up = 10.0 * np.array([np.cos(theta), 0.0, np.sin(theta)])
    b_dn = np.array([b_up[0], 0.0, 2.5 * b_up[2]])

    run = recipe("theta_bn", B_up=b_up, B_dn=b_dn)

    assert run.value("theta_bn") == pytest.approx([60.0], abs=0.1)
    assert "theta_bn_sampling_std_deg" not in run.exports
    assert "normal_spread_deg" not in run.exports
    assert "Bn_std_nT" not in run.exports


def test_theta_bn_without_inputs_prints_instructions_and_exports_nothing(recipe, capsys):
    run = recipe("theta_bn")

    assert run.exports == {}
    assert "define B_up and B_dn, or B and shock_time" in capsys.readouterr().out


def test_theta_bn_placeholder_demo_runs_only_as_a_bare_script(recipe):
    run = recipe("theta_bn", __name__="__main__")

    assert run.value("theta_bn") == pytest.approx([80.24], abs=0.01)


# ── finding the shock before computing its angle ───────────────────────────────


def test_find_shock_candidates_puts_the_constructed_shock_first(recipe):
    """The live run spent nine of twelve turns hunting the ramp with run_python cells;
    the recipe lists the largest |B| jumps so the analyst picks one and moves on."""
    series, _, _ = _shock_series()
    find = recipe("theta_bn").namespace["find_shock_candidates"]
    cands = find(series, n=3)
    assert cands, "the constructed shock is a 10 → 22.9 nT step"
    top = cands[0]
    assert abs(np.datetime64(top["time"]) - SHOCK_TIME) <= np.timedelta64(60, "s")
    assert top["jump_nT"] > 10 and top["ratio"] > 2
    assert top["B_before_nT"] < top["B_after_nT"]


def test_find_shock_candidates_separates_two_shocks_and_ignores_gaps(recipe):
    t = np.datetime64("2015-03-17T00:00:00", "s") + np.arange(4000) * np.timedelta64(3, "s")
    mag = np.full(t.size, 5.0)
    first = np.datetime64("2015-03-17T01:00:00", "s")
    second = np.datetime64("2015-03-17T02:30:00", "s")
    mag[t >= first] = 12.0
    mag[t >= second] = 20.0
    mag[(t > first + np.timedelta64(10, "m")) & (t < first + np.timedelta64(20, "m"))] = np.nan
    series = SimpleNamespace(time=t, values=mag)
    find = recipe("theta_bn").namespace["find_shock_candidates"]
    cands = find(series, n=5)
    times = [np.datetime64(c["time"]) for c in cands]
    assert len(cands) == 2, cands
    assert abs(times[0] - second) <= np.timedelta64(60, "s"), "the 8 nT step is larger"
    assert abs(times[1] - first) <= np.timedelta64(60, "s")
    assert all(np.isfinite(c["jump_nT"]) for c in cands), "a gap edge is not a jump"


def test_a_series_without_a_shock_time_lists_candidates_and_exports_nothing(recipe, capsys):
    series, _, _ = _shock_series()
    run = recipe("theta_bn", B=series)
    out = capsys.readouterr().out
    assert run.exports == {}, "no angle is computed until a time is chosen"
    assert "shock_time not set" in out and "run again with shock_time" in out
    assert "2015-03-17T04:00" in out, out
    assert run.namespace["shock_candidates"][0]["ratio"] > 2


# ── the window spread: the uncertainty that is actually there ──────────────────


def test_a_clean_step_has_a_negligible_window_spread(recipe, capsys):
    """On a perfect step every guard/span convention sees the same two constant vectors,
    so the spread is zero; the export exists and is named apart from the bootstrap."""
    series, _, _ = _shock_series()

    run = recipe("theta_bn", B=series, shock_time=SHOCK_TIME)

    assert run.exports["theta_bn_window_spread_deg"]["units"] == "deg"
    assert run.value("theta_bn_window_spread_deg")[0] == pytest.approx(0.0, abs=0.05)
    assert "theta_bn_sampling_std_deg" in run.exports
    out = capsys.readouterr().out
    assert "window sensitivity" in out and "conventions" in out


def test_a_rotating_downstream_field_shows_up_in_the_window_spread(recipe):
    """2004-11-07 17:59 UT: the downstream field rotates through the ICME sheath, so the
    angle depends on how far the window reaches — 41.5–68.0° over 49 conventions on the
    real data. Here the downstream field rotates 60° over twelve minutes: the 13-minute
    default window sees all of it, the shorter conventions of the ensemble see less, and
    the spread says so while the bootstrap inside the fixed windows stays near zero."""
    t = np.datetime64("2015-03-17T03:30:00", "s") + np.arange(1201) * np.timedelta64(3, "s")
    b_up = np.array([5.0, 0.0, 8.66])
    values = np.tile(b_up, (t.size, 1)).astype(float)
    after = t >= SHOCK_TIME
    minutes = ((t[after] - SHOCK_TIME) / np.timedelta64(60, "s")).astype(float)
    phi = np.radians(np.clip(minutes, 0, 12) * 5.0)
    values[after] = np.column_stack(
        [5.0 * np.cos(phi), 5.0 * np.sin(phi), np.full(phi.size, 21.65)]
    )
    series = SimpleNamespace(time=t, values=values)

    run = recipe("theta_bn", B=series, shock_time=SHOCK_TIME)

    assert "theta_bn" in run.exports
    assert run.value("theta_bn_window_spread_deg")[0] > 1.0
    assert run.value("theta_bn_sampling_std_deg")[0] < run.value("theta_bn_window_spread_deg")[0]


def test_hand_windows_carry_no_window_spread_and_say_so(recipe, capsys):
    series, _, _ = _shock_series()
    up = series.values[
        (series.time >= SHOCK_TIME - np.timedelta64(10, "m"))
        & (series.time <= SHOCK_TIME - np.timedelta64(2, "m"))
    ]
    dn = series.values[
        (series.time >= SHOCK_TIME + np.timedelta64(2, "m"))
        & (series.time <= SHOCK_TIME + np.timedelta64(10, "m"))
    ]

    run = recipe("theta_bn", B_up=up, B_dn=dn)

    assert "theta_bn" in run.exports
    assert "theta_bn_window_spread_deg" not in run.exports
    assert "windows chosen by the caller" in capsys.readouterr().out


# ── screening candidates against the plasma ────────────────────────────────────


def _day_with_two_jumps():
    """|B| steps at 01:00 (a shock: n and V step with it) and at 02:30 (a larger |B|
    step with no plasma signature — a sheath compression). Both on 3-s field data;
    the plasma is at 90 s cadence, as Wind SWE is."""
    t = np.datetime64("2004-11-07T00:00:00", "s") + np.arange(4000) * np.timedelta64(3, "s")
    shock = np.datetime64("2004-11-07T01:00:00", "s")
    sheath = np.datetime64("2004-11-07T02:30:00", "s")
    mag = np.full(t.size, 5.0)
    mag[t >= shock] = 12.0
    mag[t >= sheath] = 30.0
    tp = np.datetime64("2004-11-07T00:00:00", "s") + np.arange(134) * np.timedelta64(90, "s")
    n = np.where(tp >= shock, 20.0, 9.0)
    v = np.where(tp >= shock, 520.0, 400.0)
    return (
        SimpleNamespace(time=t, values=mag),
        SimpleNamespace(time=tp, values=n),
        SimpleNamespace(time=tp, values=v),
        shock,
        sheath,
    )


def test_candidates_screened_by_plasma_put_the_shock_before_the_bigger_sheath_jump(recipe):
    """2004-11-07: four sheath structures out-jumped both real shocks in |B|. With density
    and speed the recipe says which |B| rises are shocks, and lists them first."""
    series, density, speed, shock, sheath = _day_with_two_jumps()
    find = recipe("theta_bn").namespace["find_shock_candidates"]

    cands = find(series, n=10, density=density, speed=speed)

    assert len(cands) == 2
    assert abs(np.datetime64(cands[0]["time"]) - shock) <= np.timedelta64(3, "s")
    assert cands[0]["screen"].startswith("fast-forward")
    assert cands[0]["n_ratio"] == pytest.approx(20 / 9, abs=0.01)
    assert cands[0]["dV_km_s"] == pytest.approx(120.0)
    assert abs(np.datetime64(cands[1]["time"]) - sheath) <= np.timedelta64(3, "s")
    assert cands[1]["screen"].startswith("not a fast-forward shock")
    assert cands[1]["jump_nT"] > cands[0]["jump_nT"], (
        "the sheath jump is larger, and ranks second anyway"
    )


def test_candidates_without_plasma_are_labelled_unscreened_and_ranked_by_amplitude(recipe):
    series, _, _, shock, sheath = _day_with_two_jumps()
    find = recipe("theta_bn").namespace["find_shock_candidates"]

    cands = find(series, n=10)

    assert [c["screen"] for c in cands] == ["unscreened (|B| only)"] * 2
    assert abs(np.datetime64(cands[0]["time"]) - sheath) <= np.timedelta64(3, "s")


def test_candidate_time_is_the_ramp_not_the_box_centre(recipe):
    """The one-minute box that measures a jump is maximal anywhere within a minute of the
    ramp; the time handed back must be the ramp, since it is passed on as `shock_time`."""
    series, _, _ = _shock_series()
    find = recipe("theta_bn").namespace["find_shock_candidates"]

    cands = find(series, n=1)

    assert np.datetime64(cands[0]["time"]) == SHOCK_TIME


def test_a_series_with_plasma_prints_the_screen_verdicts(recipe, capsys):
    series, density, speed, _, _ = _day_with_two_jumps()

    run = recipe("theta_bn", B=series, density=density, speed=speed)

    out = capsys.readouterr().out
    assert run.exports == {}
    assert "checked against density and speed" in out
    assert "[fast-forward: n and V jump with |B|]" in out
    assert "[not a fast-forward shock: |B| jumps alone]" in out
    assert "why the others are not it" in out
