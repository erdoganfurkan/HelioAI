"""Every shipped recipe runs offline, and computes the right thing on inputs whose
answer is known.

Two layers. `test_recipe_runs_with_its_own_self_checks` executes each script the way the
sandbox would — placeholders, `if __name__ == "__main__"` demos and the `assert`s several
recipes carry all run. The synthetic tests below then feed each method an input built
from the textbook definition and check the number that comes out, so a regression in
the physics is caught here and not at a demo.

What these tests do NOT assert is left deliberately open for the scientific review in
the integration plan: the Walén frame (mean-subtracted, not de Hoffmann-Teller), the
`pressure_balance` reference field, the SEA's interpolation across gaps.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tests.recipes.conftest import RECIPE_NAMES

pytestmark = pytest.mark.recipes

_NEEDS_INPUT = {"superposed_epoch", "sep_onset_poisson_cusum"}


def _events(n=6, gap=False):
    """Synthetic event collection in the shape `get_events_timeseries` persists."""
    out = []
    for i in range(n):
        t = np.arange(0, 3600, 60).astype("timedelta64[s]") + np.datetime64(
            f"2015-03-{17 + i:02d}T00:00:00"
        )
        y = 5.0 + np.sin(np.linspace(0, np.pi, t.size)) * (i + 1)
        if gap:
            y[20:25] = np.nan
        out.append(SimpleNamespace(time=t, values=y, start=str(t[0]), stop=str(t[-1])))
    return out


def _sep_flux(onset_index=400, n=900):
    rng = np.random.default_rng(1)
    t = np.datetime64("2021-10-28T12:00:00") + np.arange(n).astype("timedelta64[m]")
    x = rng.poisson(20, n).astype(float)
    x[onset_index:] += np.linspace(0, 300, n - onset_index)
    return SimpleNamespace(time=t, values=x)


@pytest.mark.parametrize("name", RECIPE_NAMES)
def test_recipe_runs_with_its_own_self_checks(recipe, name):
    inputs = {}
    if name == "superposed_epoch":
        inputs["events"] = _events()
    if name == "sep_onset_poisson_cusum":
        inputs["flux"] = _sep_flux()
    if name == "solar_mach":
        pytest.importorskip("solarmach")
    recipe(name, **inputs)


# ── theta_bn ──────────────────────────────────────────────────────────────────


def test_theta_bn_recovers_a_constructed_normal(recipe):
    """Build a shock whose normal is known: choose n̂, an upstream field at 60° to it,
    and a downstream field with the same normal component (∇·B = 0) and an amplified
    tangential one. Coplanarity must give θ_Bn = 60° to a fraction of a degree."""
    run = recipe("theta_bn")
    theta_bn = run.namespace["theta_bn"]

    n_hat = np.array([1.0, 0.0, 0.0])
    t_hat = np.array([0.0, 1.0, 0.0])
    b_up = 5.0 * (np.cos(np.radians(60)) * n_hat + np.sin(np.radians(60)) * t_hat)
    b_dn = b_up * np.array([1.0, 2.5, 1.0])

    result = theta_bn(b_up, b_dn)
    assert abs(result["theta_bn_deg"] - 60.0) < 0.5
    assert result["geometry"] == "quasi-perpendicular"
    assert abs(abs(np.dot(result["shock_normal"], n_hat)) - 1.0) < 1e-6


def test_theta_bn_refuses_degenerate_inputs(recipe):
    theta_bn = recipe("theta_bn").namespace["theta_bn"]
    assert "error" in theta_bn(np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]))
    assert "error" in theta_bn(np.array([np.nan, 1.0, 0.0]), np.array([1.0, 2.0, 0.0]))


def test_theta_bn_averages_over_finite_rows(recipe):
    theta_bn = recipe("theta_bn").namespace["theta_bn"]
    up = np.tile([5.0, 0.0, 0.0], (10, 1))
    up[3] = np.nan
    dn = np.tile([5.0, 8.0, 0.0], (10, 1))
    result = theta_bn(up, dn)
    assert "error" not in result
    assert result["B_up_mean_nT"] == [5.0, 0.0, 0.0]


# ── mvab ──────────────────────────────────────────────────────────────────────


def test_mvab_finds_the_direction_of_least_variance(recipe):
    """A cloud with variances 100 : 10 : 0.1 along x, y, z has z as its minimum-variance
    direction, and the eigenvalue ratios follow the variances."""
    mvab = recipe("mvab").namespace["mvab"]
    rng = np.random.default_rng(7)
    b = np.column_stack(
        [
            rng.normal(0, 10, 4000),
            rng.normal(0, np.sqrt(10), 4000),
            rng.normal(0, np.sqrt(0.1), 4000),
        ]
    )
    result = mvab(b)
    assert abs(abs(result["normal_n_min"][2]) - 1.0) < 0.02
    assert 60 < result["ratio_int_min"] < 140
    assert result["quality"].startswith("well-determined")


# ── walen_test ────────────────────────────────────────────────────────────────


def test_walen_slope_is_one_on_a_pure_rotational_discontinuity(recipe):
    """On an RD the velocity fluctuation *is* the Alfvén velocity fluctuation, so the
    regression slope must be ±1 with R² near 1 — whatever frame convention is used."""
    ns = recipe("walen_test").namespace
    n = 400
    t = np.linspace(0, np.pi, n)
    b = np.column_stack([6 + 3 * np.cos(t), -2 + 2 * np.sin(t), 1 + np.sin(2 * t)])
    n_cm3 = np.full(n, 5.0)
    v = ns["alfven_velocity"](b, n_cm3) + np.array([300.0, -20.0, 10.0])
    result = ns["walen_test"](v, b, n_cm3)
    assert abs(abs(result["slope"]) - 1.0) < 0.02
    assert result["R2"] > 0.99


def test_walen_slope_is_zero_when_velocity_ignores_the_field(recipe):
    ns = recipe("walen_test").namespace
    rng = np.random.default_rng(3)
    n = 400
    b = np.column_stack([6 + 3 * np.cos(np.linspace(0, np.pi, n)), np.zeros(n), np.ones(n)])
    v = rng.normal(0, 1, (n, 3)) + 400.0
    result = ns["walen_test"](v, b, np.full(n, 5.0))
    assert abs(result["slope"]) < 0.1


# ── rankine_hugoniot ──────────────────────────────────────────────────────────


def test_rankine_hugoniot_reference_event_matches_published_values(recipe):
    """The recipe carries its own check against the 2015-03-17 shock; running it is the
    test. Here the same core is called directly with the reference means so the
    numbers are asserted in one place a reader can see."""
    ns = recipe("rankine_hugoniot").namespace
    out = ns["_rh_core"](
        n_u=10.0, n_d=25.9, V_u=-410.0, V_d=-520.0, B_u=7.0, B_d=18.0, T_u=10.0, T_d=40.0
    )
    assert out["r"] == pytest.approx(2.59, abs=0.01)
    # (n_d V_d − n_u V_u)/(n_d − n_u) on these means, in the recipe's anti-sunward-
    # positive convention (it flips the sign of a sunward-negative input pair).
    assert out["V_shock"] == pytest.approx((25.9 * 520 - 10 * 410) / 15.9, abs=0.01)
    assert out["r_mismatch"] >= 0


def test_rankine_hugoniot_speed_from_mass_conservation(recipe):
    """V_sh = (n_d V_d − n_u V_u)/(n_d − n_u): with n doubling and V from 400 to 300
    (anti-sunward positive), the shock frame speed is 200 km/s."""
    ns = recipe("rankine_hugoniot").namespace
    out = ns["_rh_core"](n_u=5.0, n_d=10.0, V_u=400.0, V_d=300.0, B_u=5.0, B_d=10.0)
    assert out["V_shock"] == pytest.approx(200.0, abs=1e-6)
    assert out["r"] == pytest.approx(2.0)


# ── pressure_balance ──────────────────────────────────────────────────────────


def test_magnetopause_standoff_shrinks_with_dynamic_pressure(recipe):
    ns = recipe("pressure_balance").namespace
    quiet = ns["mp_standoff"](n_sw=5.0, V_sw=400.0, B_sw=5.0)
    storm = ns["mp_standoff"](n_sw=20.0, V_sw=700.0, B_sw=15.0)
    assert 8.0 < quiet < 12.0
    assert storm < quiet
    # r ∝ P_dyn^(-1/6): a 4× density rise alone shrinks r by 4^(1/6) ≈ 1.26
    dense = ns["mp_standoff"](n_sw=20.0, V_sw=400.0, B_sw=5.0)
    assert quiet / dense == pytest.approx(4 ** (1 / 6), rel=0.03)


# ── pitch_angle_dist ──────────────────────────────────────────────────────────


def test_pitch_angles_are_ninety_degrees_for_velocities_perpendicular_to_b(recipe):
    ns = recipe("pitch_angle_dist").namespace
    v = np.column_stack([np.ones(50), np.zeros(50), np.zeros(50)])
    b = np.array([0.0, 0.0, 10.0])
    pa_deg, counts, edges = ns["compute_pad"](v, b)
    assert np.allclose(pa_deg, 90.0)
    # counts are normalised by sin(α): all the weight sits in the bin containing 90°
    assert np.argmax(counts) == np.searchsorted(edges, 90.0, side="right") - 1
    assert counts[np.argmax(counts)] > 0 and np.count_nonzero(counts) == 1


def test_pitch_angles_are_zero_and_180_along_and_against_b(recipe):
    ns = recipe("pitch_angle_dist").namespace
    v = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, -3.0]])
    pa_deg, _, _ = ns["compute_pad"](v, np.array([0.0, 0.0, 10.0]))
    assert np.allclose(pa_deg, [0.0, 180.0])


# ── superposed_epoch ──────────────────────────────────────────────────────────


def test_superposed_epoch_median_of_scaled_copies_is_the_middle_copy(recipe):
    """Six events that are the same curve scaled 1…6: the composite median at every
    epoch is the mean of the 3rd and 4th scaling, i.e. 5 + 3.5·sin(πτ)."""
    run = recipe("superposed_epoch", events=_events(6))
    median = run.value("epoch_median")
    tau = np.linspace(0, 1, median.size)
    assert np.allclose(median, 5.0 + 3.5 * np.sin(np.pi * tau), atol=0.05)
    assert (run.value("epoch_q25") <= median).all() and (median <= run.value("epoch_q75")).all()


def test_superposed_epoch_survives_gapped_events(recipe):
    run = recipe("superposed_epoch", events=_events(6, gap=True))
    assert np.isfinite(run.value("epoch_median")).all()


# ── sep_onset_poisson_cusum ───────────────────────────────────────────────────


def test_sep_onset_lands_at_the_start_of_the_rise(recipe):
    run = recipe("sep_onset_poisson_cusum", flux=_sep_flux(onset_index=400))
    onset = np.datetime64(run.namespace["onset_time"])
    minutes = (onset - np.datetime64("2021-10-28T12:00:00")).astype("timedelta64[m]").astype(int)
    assert 395 <= minutes <= 440, f"onset detected at +{minutes} min, rise starts at +400"


def test_sep_onset_is_none_on_a_flat_background(recipe):
    flux = _sep_flux(onset_index=899)
    flux.values[:] = np.random.default_rng(0).poisson(20, flux.values.size)
    run = recipe("sep_onset_poisson_cusum", flux=flux)
    assert run.namespace["onset_time"] is None


# ── shock_timing_2sc ──────────────────────────────────────────────────────────


def test_two_spacecraft_timing_recovers_a_planar_front_speed(recipe):
    """A front moving at 500 km/s along n̂ = x̂ reaches a spacecraft 1e6 km further
    along x 2000 s later; the recipe must read 500 km/s off that geometry."""
    ns = recipe("shock_timing_2sc").namespace
    t1 = np.datetime64("2015-03-17T04:00:00")
    t2 = t1 + np.timedelta64(2000, "s")
    r1 = np.array([1.5e6, 2.0e5, 0.0])
    r2 = r1 + np.array([1.0e6, 0.0, 3.0e4])
    out = ns["timing_2sc"](t1, t2, r1, r2, np.array([1.0, 0.0, 0.0]))
    assert out["shock_speed_km_s"] == pytest.approx(500.0, rel=1e-6)
    assert out["lag_s"] == pytest.approx(2000.0)


# ── fill_values ───────────────────────────────────────────────────────────────


def test_fill_values_blank_the_three_conventions(recipe):
    ns = recipe("fill_values").namespace
    blank_fill = ns["blank_fill"]
    assert np.isnan(blank_fill(np.array([420.0, 99999.8984375]), [99999.9])[1])
    assert np.isnan(blank_fill(np.array([5.0, -1e31]), None)[1])
    assert not np.isnan(blank_fill(np.array([99093.0]), [99999.9])[0])
