# name: superposed_epoch
# description: Superposed epoch analysis (SEA) on a collection of events: align, normalize, and composite.
# inputs: events — list of SimpleNamespace(time, values, start, stop) from load_data("<param>_events")
# outputs: epoch_median, epoch_q25, epoch_q75, epoch_ci_low, epoch_ci_high, epoch_n
# reference: Superposed epoch (Chree) analysis — Chree (1913), Phil. Trans. R. Soc. A 212, 75. Bootstrap confidence intervals — Efron, B. (1979), Ann. Statist. 7, 1.

"""Superposed Epoch Analysis (SEA).

Each event is mapped to a normalized epoch tau ∈ [0, 1] where:
  tau=0 → event start (e.g., shock arrival)
  tau=1 → event stop

Values at n_grid equally-spaced tau points are interpolated per event without
bridging data gaps: if either source sample that would support the linear
estimate is missing, the epoch value is NaN. The composite is the median + IQR
(25th–75th percentile) across the events that contribute to that epoch bin.
Bins with fewer than min_events contributing events are left NaN. A bootstrap
over events estimates a 95% confidence interval on the median composite.

Usage inside run_python:
    events = load_data("imf_gsm_events")   # list of ns(time, values, start, stop)
    # Then run this script. Set component=0 to select a single column of multi-component data.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── parameters ────────────────────────────────────────────────────────────────
n_grid = int(globals().get("n_grid", 100))          # number of epoch bins
component = int(globals().get("component", 0))      # multi-component column, 0-based
units = globals().get("units", "")                  # override units label (optional)
param_label = globals().get("param_label", "")      # override parameter label (optional)
min_events = int(globals().get("min_events", 3))    # minimum contributing events per bin
n_boot = int(globals().get("n_boot", 200))          # bootstrap resamples
seed = globals().get("seed", 0)                     # bootstrap random seed
events = globals().get("events")

if events is None:
    raise ValueError('bind `events` before running this recipe, for example events = load_data("<param>_events")')
if n_grid < 2:
    raise ValueError("n_grid must be at least 2")
if min_events < 1:
    raise ValueError("min_events must be at least 1")
if n_boot < 1:
    raise ValueError("n_boot must be at least 1")

# ── helpers ───────────────────────────────────────────────────────────────────

def _to_float_seconds(t_arr):
    """Convert datetime64 array to float seconds (relative to first point)."""
    t = np.asarray(t_arr).astype("datetime64[ms]").astype(np.float64) / 1000.0
    return t


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


def _interp_without_gap_bridging(t_target, t_source, values):
    """Linearly resample one event while preserving data holes as holes.

    Dropping NaNs and calling `np.interp` draws a line from the last finite
    sample before a gap to the first finite sample after it. In SEA that invented
    line participates in the cross-event median and can become a scientific
    feature. The companion interpolation of the missing-sample mask rejects any
    target point whose estimate would put non-zero weight on a missing sample.
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
    return _nanpercentile_min_count(boot, 2.5, 1), _nanpercentile_min_count(boot, 97.5, 1)


# ── analysis ──────────────────────────────────────────────────────────────────

tau_grid = np.linspace(0.0, 1.0, n_grid)
matrix = []   # one row per event

for ev in events:
    t_sec = _to_float_seconds(ev.time)
    y = _select_component(ev.values, component)

    if len(t_sec) < 2:
        continue

    t0, t1 = t_sec[0], t_sec[-1]
    if t1 <= t0:
        continue

    tau = (t_sec - t0) / (t1 - t0)

    y_interp = _interp_without_gap_bridging(tau_grid, tau, y)
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
epoch_ci_low, epoch_ci_high = _bootstrap_median_ci(mat, min_events, n_boot, seed)
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
