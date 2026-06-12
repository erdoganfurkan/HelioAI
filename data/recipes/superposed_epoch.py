# name: superposed_epoch
# description: Superposed epoch analysis (SEA) on a collection of events: align, normalize, and composite.
# inputs: events — list of SimpleNamespace(time, values, start, stop) from load_data("<param>_events")
# outputs: epoch_median (array of shape (n_grid,) or (n_grid, k)), epoch_q25, epoch_q75, figure

"""Superposed Epoch Analysis (SEA).

Each event is mapped to a normalized epoch tau ∈ [0, 1] where:
  tau=0 → event start (e.g., shock arrival)
  tau=1 → event stop

Values at n_grid equally-spaced tau points are interpolated (np.interp per event),
then composited across events: median + IQR (25th–75th percentile).

Usage inside run_python:
    events = load_data("imf_gsm_events")   # list of ns(time, values, start, stop)
    # Then run this script. Set component=0 to select a single column of multi-component data.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── parameters ────────────────────────────────────────────────────────────────
n_grid = 100        # number of epoch bins
component = 0       # which column to analyse for multi-component data (0-based)
units = ""          # override units label (optional)
param_label = ""    # override parameter label (optional)

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


# ── analysis ──────────────────────────────────────────────────────────────────

tau_grid = np.linspace(0.0, 1.0, n_grid)
matrix = []   # one row per event

for ev in events:
    t_sec = _to_float_seconds(ev.time)
    vals = _clean(ev.values)

    # Select component for multi-dimensional data
    if vals.ndim == 2 and vals.shape[1] > 1:
        y = vals[:, component]
    elif vals.ndim == 2 and vals.shape[1] == 1:
        y = vals[:, 0]
    else:
        y = vals.ravel()

    if len(t_sec) < 2:
        continue

    t0, t1 = t_sec[0], t_sec[-1]
    if t1 <= t0:
        continue

    tau = (t_sec - t0) / (t1 - t0)

    # Interpolate only over finite points
    finite = np.isfinite(y)
    if finite.sum() < 2:
        continue

    y_interp = np.interp(tau_grid, tau[finite], y[finite])
    matrix.append(y_interp)

if not matrix:
    raise RuntimeError("No valid events to composite — check that events have at least 2 finite points")

mat = np.array(matrix)   # shape (n_events, n_grid)
epoch_median = np.nanmedian(mat, axis=0)
epoch_q25 = np.nanpercentile(mat, 25, axis=0)
epoch_q75 = np.nanpercentile(mat, 75, axis=0)
n_events = mat.shape[0]

# ── figure ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))

for row in mat:
    ax.plot(tau_grid, row, color="steelblue", alpha=0.2, linewidth=0.7)

ax.fill_between(tau_grid, epoch_q25, epoch_q75,
                color="steelblue", alpha=0.35, label="IQR (25–75%)")
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
export("epoch_median", epoch_median)
export("epoch_q25", epoch_q25)
export("epoch_q75", epoch_q75)

print(f"SEA complete: {n_events} events, {n_grid} epoch bins")
print(f"Median range: [{np.nanmin(epoch_median):.3g}, {np.nanmax(epoch_median):.3g}]")


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
