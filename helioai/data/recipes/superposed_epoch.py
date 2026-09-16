# name: superposed_epoch
# description: Superposed epoch analysis (SEA) on a collection of events: align, normalize, and composite.
# inputs: events — list of SimpleNamespace(time, values, start, stop) from load_data("<param>_events"); units; min_events; n_boot; seed; max_gap_cadences; component; n_grid
# outputs: epoch_median, epoch_q25, epoch_q75, epoch_ci_low, epoch_ci_high, epoch_n
# reference: Superposed epoch (Chree) analysis — Chree (1913), Phil. Trans. R. Soc. A 212, 75. Bootstrap confidence intervals — Efron, B. (1979), Ann. Statist. 7, 1.

"""Superposed Epoch Analysis (SEA).

Each event is mapped to a normalized epoch tau ∈ [0, 1] where:
  tau=0 → event start (e.g., shock arrival)
  tau=1 → event stop

Values at n_grid equally-spaced tau points are interpolated per event without
bridging data gaps: if either source sample that would support the linear
estimate is missing, or if the bracketing source samples are separated by more
than max_gap_cadences times the event cadence, the epoch value is NaN. Event
start/stop metadata defines tau=0 and tau=1 when present, so missing leading or
trailing coverage is not stretched to fill the epoch. The composite is the
median + IQR (25th–75th percentile) across the events that contribute to that
epoch bin. Bins with fewer than min_events contributing events are left NaN.

A bootstrap over events estimates a pointwise 95% confidence interval on the
median composite. The default n_boot=200 is a quick-look setting, with about
five draws per 2.5% tail; raise it for publication-quality intervals.

If the event collection carries units, the recipe composites only after every
event has been converted to one common unit: either the caller's `units`, or the
first event unit when `units` is not bound. Incompatible unit collections are
refused because no warning can repair a median of unlike dimensions.

Usage inside run_python:
    events = load_data("imf_gsm_events")   # list of ns(time, values, start, stop)
    # Then run this script. Set component=0 to select a single column of multi-component data.
"""

import re

import numpy as np
import matplotlib.pyplot as plt

try:
    import astropy.units as u
except ImportError:
    u = None


_TRAILING_ANNOTATION = re.compile(r"^(?P<unit>.*\S)\s*\([^()]*\)$")


def _unit_label(unit):
    """Return the unit string attached to one event, without a trailing annotation.

    CDAWeb labels some products `nT (1min)` or `nT (3sec)`: the parenthesis is the
    cadence, not part of the unit, and left in place it made `nT (1min)` and `nT`
    "incompatible". Only a parenthesis that *follows* a unit is dropped — `(nT)` alone
    is kept as it is.
    """
    label = str(unit or "").strip()
    m = _TRAILING_ANNOTATION.match(label)
    return m.group("unit").strip() if m else label


def _unit_list_for_message(units):
    """Preserve unit encounter order for refusal messages."""
    out = []
    for unit in units:
        label = _unit_label(unit) or "<missing>"
        if label not in out:
            out.append(label)
    return ", ".join(out)


def _unit_conversion_factors(events, requested_units):
    """Choose export units and per-event scale factors before any composite.

    Values with different physical units cannot be numerically composited until
    they share a unit. If Astropy cannot prove and perform that conversion, the
    recipe refuses the composite rather than warning and exporting meaningless
    medians.
    """
    source_units = [_unit_label(getattr(ev, "units", "")) for ev in events]
    present = [unit for unit in source_units if unit]
    target = _unit_label(requested_units) if requested_units else (present[0] if present else "")

    if not present:
        return [1.0] * len(events), target

    if any(not unit for unit in source_units):
        raise ValueError(f"events carry incompatible units: {_unit_list_for_message(source_units)} — composite refused")

    message_units = source_units
    if target and target not in source_units:
        message_units = [*source_units, target]

    if all(unit == target for unit in source_units):
        return [1.0] * len(events), target

    if u is None:
        raise ValueError(f"events carry incompatible units: {_unit_list_for_message(message_units)} — composite refused")

    try:
        target_unit = u.Unit(target)
        factors = [(1.0 * u.Unit(unit)).to(target_unit).value for unit in source_units]
    except Exception as exc:
        raise ValueError(
            f"events carry incompatible units: {_unit_list_for_message(message_units)} — composite refused"
        ) from exc

    return [float(factor) for factor in factors], target


# ── parameters ────────────────────────────────────────────────────────────────
n_grid = int(globals().get("n_grid", 100))          # number of epoch bins
component = int(globals().get("component", 0))      # multi-component column, 0-based
units = globals().get("units")                      # override units label (optional)
param_label = globals().get("param_label", "")      # override parameter label (optional)
min_events = int(globals().get("min_events", 3))    # minimum contributing events per bin
n_boot = int(globals().get("n_boot", 200))          # bootstrap resamples
seed = globals().get("seed", 0)                     # bootstrap random seed
max_gap_cadences = float(globals().get("max_gap_cadences", 2.0))
events = globals().get("events")

if events is None:
    raise ValueError('bind `events` before running this recipe, for example events = load_data("<param>_events")')
events = list(events)
unit_factors, units = _unit_conversion_factors(events, units)
if n_grid < 2:
    raise ValueError("n_grid must be at least 2")
if min_events < 1:
    raise ValueError("min_events must be at least 1")
if n_boot < 1:
    raise ValueError("n_boot must be at least 1")
if max_gap_cadences <= 0:
    raise ValueError("max_gap_cadences must be positive")

# ── helpers ───────────────────────────────────────────────────────────────────

def _to_float_seconds(t_arr):
    """Convert a time array to float seconds."""
    arr = np.asarray(t_arr)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ms]").astype(np.float64) / 1000.0
    if np.issubdtype(arr.dtype, np.timedelta64):
        return arr.astype("timedelta64[ms]").astype(np.float64) / 1000.0
    return arr.astype(float)


def _time_bound_seconds(value):
    """Parse an event boundary from manifest metadata."""
    if value is None:
        return np.nan
    try:
        return np.datetime64(value).astype("datetime64[ms]").astype(np.float64) / 1000.0
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan


def _clean(arr):
    """Replace fill values (|x|≥1e30) and infinities with NaN."""
    a = np.asarray(arr, dtype=float)
    a[~np.isfinite(a)] = np.nan
    a[np.abs(a) >= 1e30] = np.nan
    return a


def _select_component(values, component):
    """Return the scalar series SEA should composite from an event.

    Multi-component plasma and field products usually arrive as N×3 arrays. SEA
    is a scalar composite, so the caller must choose the physical component
    before the events are stacked. A bad component index is a recipe input error,
    not an event to silently skip.
    """
    vals = _clean(values)
    if vals.ndim == 2:
        if vals.shape[1] == 0:
            raise ValueError("event values have zero columns")
        if vals.shape[1] == 1:
            return vals[:, 0]
        if component < 0 or component >= vals.shape[1]:
            raise ValueError(f"component={component} is outside the {vals.shape[1]} columns in event values")
        return vals[:, component]
    return vals.ravel()


def _event_bounds_seconds(ev, t_sec):
    """Choose the physical event interval used for epoch normalization."""
    finite_time = t_sec[np.isfinite(t_sec)]
    if finite_time.size < 2:
        return np.nan, np.nan

    t0 = _time_bound_seconds(getattr(ev, "start", None))
    t1 = _time_bound_seconds(getattr(ev, "stop", None))
    if not np.isfinite(t0):
        t0 = np.nanmin(finite_time)
    if not np.isfinite(t1):
        t1 = np.nanmax(finite_time)
    return t0, t1


def _interp_without_gap_bridging(t_target, t_source, values, max_gap_cadences):
    """Linearly resample one event while preserving data holes as holes.

    Dropping NaNs and calling `np.interp` draws a line from the last finite
    sample before a gap to the first finite sample after it. Removed timestamps
    are the same scientific gap with no NaN left behind, so a cadence check also
    rejects estimates supported by source samples that are too far apart.
    """
    tt = np.asarray(t_target, dtype=float)
    ts = np.asarray(t_source, dtype=float)
    v = np.asarray(values, dtype=float)

    finite_time = np.isfinite(ts)
    ts = ts[finite_time]
    v = v[finite_time]
    if ts.size < 2:
        return np.full(tt.shape, np.nan)

    order = np.argsort(ts)
    ts = ts[order]
    v = v[order]
    good = np.isfinite(v)
    if not good.any():
        return np.full(tt.shape, np.nan)

    out = np.interp(tt, ts[good], v[good], left=np.nan, right=np.nan)
    touched = np.interp(tt, ts, (~good).astype(float), left=1.0, right=1.0)
    out[touched > 0.0] = np.nan

    good_ts = ts[good]
    if good_ts.size >= 2:
        dt = np.diff(good_ts)
        finite_dt = dt[np.isfinite(dt) & (dt > 0)]
        if finite_dt.size:
            cadence = np.median(finite_dt)
            if np.isfinite(cadence) and cadence > 0:
                right = np.searchsorted(good_ts, tt, side="left")
                exact = np.zeros(tt.shape, dtype=bool)
                inside = right < good_ts.size
                tol = max(cadence, 1.0) * 1e-12
                exact[inside] = np.abs(good_ts[right[inside]] - tt[inside]) <= tol
                bracketed = (right > 0) & (right < good_ts.size) & ~exact
                span = np.full(tt.shape, np.nan)
                span[bracketed] = good_ts[right[bracketed]] - good_ts[right[bracketed] - 1]
                out[span > max_gap_cadences * cadence] = np.nan
    return out


def _nanpercentile_min_count(values, percentile, min_count):
    """Percentile by epoch bin, leaving under-sampled bins as NaN.

    A gap in one event should not destroy the whole composite, but a median
    supported by only one or two events is not a robust superposed-epoch result.
    The count threshold is applied independently in each epoch bin.
    """
    arr = np.asarray(values, dtype=float)
    counts = np.sum(np.isfinite(arr), axis=0)
    out = np.full(arr.shape[1], np.nan)
    keep = counts >= min_count
    if keep.any():
        out[keep] = np.nanpercentile(arr[:, keep], percentile, axis=0)
    return out


def _bootstrap_median_ci(matrix, min_events, n_boot, seed):
    """Bootstrap the event median to estimate a pointwise 95% confidence band.

    Events, not individual epoch bins, are the resampling unit because the SEA
    curve from one event is a coherent trajectory. The same minimum-contributor
    rule is applied to each bootstrap composite before the pointwise percentile
    interval is formed.
    """
    rng = np.random.default_rng(seed)
    n_events = matrix.shape[0]
    boot = np.full((n_boot, matrix.shape[1]), np.nan)
    for i in range(n_boot):
        sample = matrix[rng.integers(0, n_events, n_events)]
        boot[i] = _nanpercentile_min_count(sample, 50, min_events)
    finite_fraction = np.sum(np.isfinite(boot), axis=0) / n_boot
    return _nanpercentile_min_count(boot, 2.5, 1), _nanpercentile_min_count(boot, 97.5, 1), finite_fraction


# ── analysis ──────────────────────────────────────────────────────────────────

tau_grid = np.linspace(0.0, 1.0, n_grid)
matrix = []   # one row per event

for ev, unit_factor in zip(events, unit_factors):
    t_sec = _to_float_seconds(ev.time)
    y = _select_component(ev.values, component) * unit_factor

    if len(t_sec) < 2:
        continue

    t0, t1 = _event_bounds_seconds(ev, t_sec)
    if t1 <= t0:
        continue

    tau = (t_sec - t0) / (t1 - t0)

    y_interp = _interp_without_gap_bridging(tau_grid, tau, y, max_gap_cadences)
    if np.isfinite(y_interp).sum() < 2:
        continue
    matrix.append(y_interp)

if not matrix:
    raise RuntimeError("No valid events to composite — check that events have at least 2 finite points")

mat = np.array(matrix)   # shape (n_events, n_grid)
epoch_n = np.sum(np.isfinite(mat), axis=0)
epoch_median = _nanpercentile_min_count(mat, 50, min_events)
epoch_q25 = _nanpercentile_min_count(mat, 25, min_events)
epoch_q75 = _nanpercentile_min_count(mat, 75, min_events)
epoch_ci_low, epoch_ci_high, bootstrap_finite_fraction = _bootstrap_median_ci(mat, min_events, n_boot, seed)
ci_mask = (epoch_n < min_events) | (bootstrap_finite_fraction < 0.8)
epoch_ci_low[ci_mask] = np.nan
epoch_ci_high[ci_mask] = np.nan
n_events = mat.shape[0]

# ── figure ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))

for row in mat:
    ax.plot(tau_grid, row, color="steelblue", alpha=0.2, linewidth=0.7)

ax.fill_between(tau_grid, epoch_q25, epoch_q75,
                color="steelblue", alpha=0.35, label="IQR (25–75%)")
ax.fill_between(tau_grid, epoch_ci_low, epoch_ci_high,
                color="lightsteelblue", alpha=0.25, label="95% bootstrap CI")
ax.plot(tau_grid, epoch_median, color="navy", linewidth=2.0, label="Median")

ax.set_xlabel("Normalized epoch τ  (0 = start, 1 = stop)")
ylabel = param_label or f"Component {component}"
if units:
    ylabel += f"  [{units}]"
ax.set_ylabel(ylabel)
ax.set_title(f"Superposed epoch analysis — {n_events} events")
ax.legend(loc="best")
ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="--")
ax.axvline(1.0, color="gray", linewidth=0.8, linestyle="--")
plt.tight_layout()
plt.show()

# ── export ────────────────────────────────────────────────────────────────────
export("epoch_median", epoch_median, units)
export("epoch_q25", epoch_q25, units)
export("epoch_q75", epoch_q75, units)
export("epoch_ci_low", epoch_ci_low, units)
export("epoch_ci_high", epoch_ci_high, units)
export("epoch_n", epoch_n, "")

print(f"SEA complete: {n_events} events, {n_grid} epoch bins")
finite_median = epoch_median[np.isfinite(epoch_median)]
if finite_median.size:
    print(f"Median range: [{np.nanmin(finite_median):.3g}, {np.nanmax(finite_median):.3g}]")
else:
    print("Median range: all epoch bins are under-sampled")


# ── standalone demo (synthetic) ───────────────────────────────────────────────
# Uncomment to test without real data:
#
# import types
# rng = np.random.default_rng(42)
# events = []
# for _ in range(20):
#     n = rng.integers(30, 80)
#     t = np.arange(n, dtype="datetime64[s]") + np.datetime64("2005-01-17T00:00:00")
#     v = np.sin(np.linspace(0, np.pi, n)) + rng.normal(0, 0.2, n)
#     ev = types.SimpleNamespace(time=t, values=v, start=str(t[0]), stop=str(t[-1]))
#     events.append(ev)
#
# Real data example:
#
# events = load_data("imf_gsm_events")   # persisted by get_events_timeseries
# component = 2   # Bz
# units = "nT"
# param_label = "Bz GSM"
