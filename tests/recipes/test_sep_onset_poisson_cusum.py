from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u

pytestmark = pytest.mark.recipes


def _sep_flux(onset_index=400, n=900, ramp=300.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.datetime64("2021-10-28T12:00:00") + np.arange(n).astype("timedelta64[m]")
    x = rng.poisson(20, n).astype(float)
    if onset_index < n:
        x[onset_index:] += np.linspace(0, ramp, n - onset_index)
    return SimpleNamespace(time=t, values=x)


def _minutes_after_start(run):
    if run.namespace["onset_time"] is None:
        return None
    onset = np.datetime64(run.namespace["onset_time"])
    return (onset - np.datetime64("2021-10-28T12:00:00")).astype("timedelta64[m]").astype(int)


def test_poisson_and_zscore_cusums_are_the_same_detector_at_different_scales(recipe):
    poisson = recipe("sep_onset_poisson_cusum", flux=_sep_flux(ramp=120.0), method="poisson")
    zscore = recipe("sep_onset_poisson_cusum", flux=_sep_flux(ramp=120.0), method="zscore")

    assert poisson.namespace["method_used"] == "poisson"
    assert zscore.namespace["method_used"] == "zscore"
    assert poisson.value("cusum") == pytest.approx(
        poisson.namespace["sigma"] * zscore.value("cusum")
    )
    assert poisson.value("cusum")[400:441].max() > 50.0
    assert zscore.value("cusum")[400:441].max() < 25.0


def test_poisson_cusum_starts_with_the_first_sample_in_the_detection_segment(recipe):
    ns = recipe("sep_onset_poisson_cusum", flux=_sep_flux()).namespace

    cusum = ns["cusum_poisson"](np.array([100.0, 20.0, 20.0, 30.0, 10.0]), 20.0, 30.0)

    assert cusum == pytest.approx(
        [
            75.3369653762357,
            70.67393075247139,
            66.01089612870709,
            71.34786150494278,
            56.68482688117848,
        ],
        abs=1e-9,
    )


@pytest.mark.parametrize("method", ["poisson", "zscore"])
def test_sep_onset_methods_land_at_the_start_of_the_rise(recipe, method):
    run = recipe("sep_onset_poisson_cusum", flux=_sep_flux(onset_index=400), method=method)

    minutes = _minutes_after_start(run)
    assert run.namespace["method_used"] == method
    assert 395 <= minutes <= 440, f"{method} onset detected at +{minutes} min"
    assert run.exports["onset_index"]["units"] == ""
    assert int(run.value("onset_index")) == minutes
    assert "onset_time" not in run.exports


@pytest.mark.parametrize("method", ["poisson", "zscore"])
def test_sep_onset_methods_are_none_on_flat_background(recipe, method):
    flux = _sep_flux(onset_index=899, seed=0)
    flux.values[:] = np.random.default_rng(0).poisson(20, flux.values.size)

    run = recipe("sep_onset_poisson_cusum", flux=flux, method=method)

    assert run.namespace["onset_time"] is None
    assert np.isnan(run.value("onset_index"))


def test_robust_background_ignores_spikes_that_hide_the_onset_with_mean_std(recipe):
    flux = _sep_flux(onset_index=400)
    flux.values[[10, 30, 50, 70, 90]] *= 20.0

    robust = recipe("sep_onset_poisson_cusum", flux=flux, robust=True)
    mean_std = recipe("sep_onset_poisson_cusum", flux=flux, robust=False)

    assert 395 <= _minutes_after_start(robust) <= 440
    mean_std_minutes = _minutes_after_start(mean_std)
    assert mean_std_minutes is None or mean_std_minutes > 440


def test_poisson_cusum_export_uses_bound_flux_units(recipe):
    units = "cm-2 s-1 sr-1 MeV-1"
    u.Unit(units)

    run = recipe("sep_onset_poisson_cusum", flux=_sep_flux(), units=units)

    assert run.exports["cusum"]["units"] == units


def test_poisson_cusum_export_uses_flux_units_when_units_are_not_bound(recipe):
    units = "cm-2 s-1 sr-1 MeV-1"
    u.Unit(units)
    flux = _sep_flux()
    flux.units = units

    run = recipe("sep_onset_poisson_cusum", flux=flux, method="poisson")

    assert run.exports["cusum"]["units"] == units


def test_zscore_cusum_export_is_dimensionless_even_with_flux_units(recipe):
    units = "cm-2 s-1 sr-1 MeV-1"
    u.Unit(units)

    run = recipe("sep_onset_poisson_cusum", flux=_sep_flux(), method="zscore", units=units)

    assert run.exports["cusum"]["units"] == ""


def test_missing_samples_do_not_confirm_an_onset(recipe):
    flux = _sep_flux(n=180)
    flux.values[:] = 20.0
    flux.values[:121] = np.random.default_rng(3).poisson(20, 121)
    flux.values[121] = 20.0
    flux.values[122] = 1000.0
    flux.values[123:152] = np.nan

    run = recipe("sep_onset_poisson_cusum", flux=flux)

    assert run.namespace["onset_time"] is None
    assert np.isnan(run.value("onset_index"))


def test_zero_mad_background_falls_back_to_standard_deviation_for_zscore(recipe, capsys):
    flux = _sep_flux(n=180)
    flux.values[:] = np.resize(np.array([0.0, 0.0, 1.0]), flux.values.size)

    run = recipe("sep_onset_poisson_cusum", flux=flux, method="zscore")

    bg = flux.values[:121]
    assert run.namespace["sigma"] == pytest.approx(np.std(bg))
    assert "MAD is zero" in capsys.readouterr().out


def test_poisson_cusum_refuses_non_positive_background_mean(recipe):
    flux = _sep_flux(n=180)
    flux.values[:] = np.resize(np.array([0.0, 0.0, 1.0]), flux.values.size)

    with pytest.raises(ValueError, match="positive background mean"):
        recipe("sep_onset_poisson_cusum", flux=flux, method="poisson")
