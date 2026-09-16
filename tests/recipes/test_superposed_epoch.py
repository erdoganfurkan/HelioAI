from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.recipes


def _events(
    n=6,
    *,
    gap_indices=(),
    drop_indices=(),
    n_samples=101,
    components=False,
    units=None,
    first_sample=0,
):
    """Synthetic event collection in the shape `get_events_timeseries` persists."""
    out = []
    tau = np.linspace(0.0, 1.0, n_samples)
    gap = (0.4 <= tau) & (tau <= 0.5)
    drop = np.zeros(n_samples, dtype=bool)
    drop[41:61] = True

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
        keep = np.ones(n_samples, dtype=bool)
        keep[:first_sample] = False
        if i in drop_indices:
            keep &= ~drop
        event = SimpleNamespace(time=t[keep], values=values[keep], start=str(t[0]), stop=str(t[-1]))
        if units is not None:
            event.units = units
        out.append(event)

    return out


def _hole_mask(n_grid):
    tau = np.linspace(0.0, 1.0, n_grid)
    return (0.4 <= tau) & (tau <= 0.5)


def _constant_events(unit_values):
    """Flat event collection for testing unit conversion before compositing."""
    out = []
    for i, (value, units) in enumerate(unit_values):
        t = np.datetime64("2015-03-17T00:00:00") + np.timedelta64(i, "D")
        t = t + np.arange(11).astype("timedelta64[m]")
        out.append(
            SimpleNamespace(
                time=t,
                values=np.full(t.size, value),
                start=str(t[0]),
                stop=str(t[-1]),
                units=units,
            )
        )
    return out


def _run_superposed_epoch_with_exports(**inputs):
    src = (
        Path(__file__).resolve().parents[2] / "helioai" / "data" / "recipes" / "superposed_epoch.py"
    ).read_text(encoding="utf-8")
    exports = {}

    def export(name, data, units=""):
        exports[name] = {"value": np.asarray(data), "units": units}

    namespace = {"__name__": "recipe", "export": export, **inputs}
    return src, namespace, exports


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


def test_superposed_epoch_does_not_bridge_removed_samples(recipe):
    run = recipe(
        "superposed_epoch", events=_events(6, drop_indices=range(6)), n_grid=101, n_boot=20
    )
    median = run.value("epoch_median")
    tau = np.linspace(0.0, 1.0, median.size)
    missing = (0.41 <= tau) & (tau <= 0.60)

    assert np.isnan(median[missing]).all()
    assert (run.value("epoch_n")[missing] == 0).all()


def test_superposed_epoch_masks_ci_when_too_few_events_contribute(recipe):
    run = recipe(
        "superposed_epoch", events=_events(6, gap_indices={0, 1, 2, 3}), n_grid=101, n_boot=200
    )
    hole = _hole_mask(run.value("epoch_median").size)

    assert (run.value("epoch_n")[hole] == 2).all()
    assert np.isnan(run.value("epoch_median")[hole]).all()
    assert np.isnan(run.value("epoch_ci_low")[hole]).all()
    assert np.isnan(run.value("epoch_ci_high")[hole]).all()


def test_superposed_epoch_masks_ci_when_bootstrap_support_is_low(recipe):
    run = recipe(
        "superposed_epoch", events=_events(6, gap_indices={0, 1, 2}), n_grid=101, n_boot=200
    )
    hole = _hole_mask(run.value("epoch_median").size)

    assert (run.value("epoch_n")[hole] == 3).all()
    assert np.isfinite(run.value("epoch_median")[hole]).all()
    assert np.isnan(run.value("epoch_ci_low")[hole]).all()
    assert np.isnan(run.value("epoch_ci_high")[hole]).all()


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


def test_superposed_epoch_infers_units_from_event_collection(recipe):
    run = recipe("superposed_epoch", events=_events(6, units="nT"), n_grid=21, n_boot=20)

    for name in ("epoch_median", "epoch_q25", "epoch_q75", "epoch_ci_low", "epoch_ci_high"):
        assert run.exports[name]["units"] == "nT"


def test_superposed_epoch_converts_compatible_event_units_before_compositing(recipe):
    events = _constant_events([(1.0, "nT")] * 3 + [(0.001, "uT")] * 3)

    run = recipe("superposed_epoch", events=events, n_grid=11, n_boot=20)

    assert run.exports["epoch_median"]["units"] == "nT"
    assert np.allclose(run.value("epoch_median"), 1.0)


def test_superposed_epoch_refuses_incompatible_event_units(recipe):
    events = _constant_events([(1.0, "nT"), (400.0, "km/s"), (1.0, "nT")])
    src, namespace, exports = _run_superposed_epoch_with_exports(
        events=events, n_grid=11, n_boot=20
    )

    with pytest.raises(ValueError, match="events carry incompatible units: nT, km/s"):
        exec(compile(src, "superposed_epoch.py", "exec"), namespace)

    assert exports == {}


def test_superposed_epoch_reads_a_cadence_annotation_as_part_of_the_label_not_the_unit(recipe):
    """CDAWeb labels `WI_H0_MFI/BF1` as `nT (1min)`. A live run mixing it with plain `nT`
    events was refused three times as "incompatible units" and only ran once the caller
    dropped `units` altogether. The parenthesis is an annotation, not a unit."""
    events = _constant_events([(1.0, "nT (1min)")] * 3 + [(1.0, "nT")] * 3)

    run = recipe("superposed_epoch", events=events, units="nT", n_grid=11, n_boot=20)

    assert run.exports["epoch_median"]["units"] == "nT"
    assert np.allclose(run.value("epoch_median"), 1.0)

    inferred = recipe(
        "superposed_epoch", events=_constant_events([(1.0, "nT (1min)")] * 6), n_grid=11, n_boot=20
    )
    assert inferred.exports["epoch_median"]["units"] == "nT"


def test_superposed_epoch_converts_event_units_to_caller_unit(recipe):
    events = _constant_events([(1.0, "nT")] * 6)

    run = recipe("superposed_epoch", events=events, units="pT", n_grid=11, n_boot=20)

    assert run.exports["epoch_median"]["units"] == "pT"
    assert np.allclose(run.value("epoch_median"), 1000.0)


def test_superposed_epoch_uses_event_start_stop_for_epoch_boundaries(recipe):
    run = recipe("superposed_epoch", events=_events(6, first_sample=20), n_grid=101, n_boot=20)
    median = run.value("epoch_median")
    tau = np.linspace(0.0, 1.0, median.size)
    leading = tau < 0.20

    assert np.isnan(median[leading]).all()
    assert (run.value("epoch_n")[leading] == 0).all()
    assert np.allclose(median[~leading], 5.0 + 3.5 * np.sin(np.pi * tau[~leading]))


def test_superposed_epoch_requires_bound_events(recipe):
    with pytest.raises(ValueError, match="bind `events`"):
        recipe("superposed_epoch")
