from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.recipes


def _events(n=6, *, gap_indices=(), n_samples=101, components=False):
    """Synthetic event collection in the shape `get_events_timeseries` persists."""
    out = []
    tau = np.linspace(0.0, 1.0, n_samples)
    gap = (0.4 <= tau) & (tau <= 0.5)

    for i in range(n):
        scale = i % 6 + 1
        t = np.datetime64("2015-03-17T00:00:00") + np.timedelta64(i, "D")
        t = t + np.arange(n_samples).astype("timedelta64[m]")
        y = 5.0 + scale * np.sin(np.pi * tau)
        if i in gap_indices:
            y = y.copy()
            y[gap] = np.nan
        if components:
            values = np.column_stack(
                [
                    y,
                    -100.0 - scale * np.sin(np.pi * tau),
                    -20.0 + 10.0 * scale * np.sin(np.pi * tau),
                ]
            )
        else:
            values = y
        out.append(SimpleNamespace(time=t, values=values, start=str(t[0]), stop=str(t[-1])))

    return out


def _hole_mask(n_grid):
    tau = np.linspace(0.0, 1.0, n_grid)
    return (0.4 <= tau) & (tau <= 0.5)


def test_superposed_epoch_does_not_bridge_a_hole_shared_by_all_events(recipe):
    run = recipe("superposed_epoch", events=_events(6, gap_indices=range(6)), n_grid=101, n_boot=20)
    median = run.value("epoch_median")
    tau = np.linspace(0.0, 1.0, median.size)
    hole = _hole_mask(median.size)

    assert np.isnan(median[hole]).all()
    assert median.size == 101
    assert (run.value("epoch_n")[hole] == 0).all()
    assert (run.value("epoch_n")[~hole] == 6).all()
    assert np.allclose(median[~hole], 5.0 + 3.5 * np.sin(np.pi * tau[~hole]))


def test_superposed_epoch_keeps_bins_with_enough_ungapped_events(recipe):
    run = recipe("superposed_epoch", events=_events(6, gap_indices={0, 1}), n_grid=101, n_boot=20)
    median = run.value("epoch_median")
    tau = np.linspace(0.0, 1.0, median.size)
    hole = _hole_mask(median.size)

    assert np.isfinite(median[hole]).all()
    assert (run.value("epoch_n")[hole] == 4).all()
    assert (run.value("epoch_n")[~hole] == 6).all()
    assert np.allclose(median[hole], 5.0 + 4.5 * np.sin(np.pi * tau[hole]))
    assert np.allclose(median[~hole], 5.0 + 3.5 * np.sin(np.pi * tau[~hole]))


def test_superposed_epoch_bootstrap_ci_brackets_median_and_tightens_with_more_events(recipe):
    run6 = recipe("superposed_epoch", events=_events(6), n_grid=51, n_boot=400, seed=2)
    median = run6.value("epoch_median")
    ci_low = run6.value("epoch_ci_low")
    ci_high = run6.value("epoch_ci_high")

    assert (ci_low <= median).all()
    assert (median <= ci_high).all()

    run30 = recipe("superposed_epoch", events=_events(30), n_grid=51, n_boot=400, seed=2)
    tau = np.linspace(0.0, 1.0, median.size)
    interior = np.sin(np.pi * tau) > 0.5
    width6 = ci_high - ci_low
    width30 = run30.value("epoch_ci_high") - run30.value("epoch_ci_low")

    assert np.nanmedian(width30[interior]) < np.nanmedian(width6[interior])


def test_superposed_epoch_component_binding_selects_requested_column(recipe):
    run = recipe(
        "superposed_epoch",
        events=_events(6, components=True),
        component=2,
        n_grid=101,
        n_boot=20,
    )
    tau = np.linspace(0.0, 1.0, run.value("epoch_median").size)

    assert np.allclose(run.value("epoch_median"), -20.0 + 35.0 * np.sin(np.pi * tau))


def test_superposed_epoch_exports_data_units(recipe):
    run = recipe("superposed_epoch", events=_events(6), n_grid=21, n_boot=20, units="nT")

    for name in ("epoch_median", "epoch_q25", "epoch_q75", "epoch_ci_low", "epoch_ci_high"):
        assert run.exports[name]["units"] == "nT"
    assert run.exports["epoch_n"]["units"] == ""


def test_superposed_epoch_requires_bound_events(recipe):
    with pytest.raises(ValueError, match="bind `events`"):
        recipe("superposed_epoch")
