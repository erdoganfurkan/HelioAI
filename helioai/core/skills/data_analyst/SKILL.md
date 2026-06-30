---
name: data_analyst
description: Download and analyze heliophysics time series — statistics, FFT, multi-mission comparison, event detection, plotting.
when_to_use: The user wants to retrieve data, compute statistics, plot a time series, compare multiple missions, detect plasma events (shocks, reconnection, CME, SIR), or run any numerical analysis on speasy parameters.
allowed_tools: [search_parameters, get_timeseries, get_events_timeseries, load_recipe, run_python]
---

# Procedure — analyze a time series

## RULE ZERO — run_python is the only tool that makes figures and numbers
A text description of a plot is not a figure. To produce a figure, call run_python with `plt.show()`.

## RULE ONE — download outside the sandbox, always
Call `get_timeseries` (or `get_events_timeseries`) BEFORE `run_python` — the sandbox has a 60 s
timeout that speasy downloads blow past. The result carries a `dataset` key; read it inside
run_python with `load_data("name")`. Never call `spz.get_data()` in the sandbox for data you can
fetch first; use it only for data the loop has not already downloaded.

## RULE TWO — recipe before custom code
For a standard, named computation, `load_recipe(name)`, adapt it to your loaded data, and paste it
into run_python. Write custom code only when no recipe matches — recipes carry their scientific
reference, so this also gives you provenance.

| Task | Recipe |
|---|---|
| Shock normal angle θ_Bn | `theta_bn` |
| Discontinuity / current-sheet normal (minimum variance) | `mvab` |
| Shock jump conditions, compression ratio, shock speed | `rankine_hugoniot` |
| Rotational vs tangential discontinuity | `walen_test` |
| Magnetopause standoff distance (pressure balance) | `pressure_balance` |
| Particle pitch-angle distribution | `pitch_angle_dist` |
| Superposed epoch analysis over a catalog | `superposed_epoch` |

When you write custom code for a computation with no recipe, call
`document_method("<name>", reference="<paper/formula>", method="<one line>")` in run_python so the
method is recorded with its source.

## Sandbox helpers (provided, no import needed)
- `load_data("name")` → `ns(time, values, unit)` for a dataset from get_timeseries.
- `param_card(var, param_id)` → metadata card in the UI. Call once per downloaded parameter.
- `clean(var.values)` → masks CDF fill values (`-1e31`, `9.96e36`) to NaN. Always wrap `.values`.
- `export(name, value)` → surfaces a key number in the reply. Call for every result that matters.
- `plt.show()` → REQUIRED; it is what saves the figure to disk.

## Resolve the id first (if needed)
If the id is missing, vague, or malformed (extra path segments like `cda/ACE/MAG/AC_H0_MFI/...`),
`search_parameters` with a plain-English query and pick the shortest matching id.

## Canonical template — download then plot
```python
import numpy as np
import matplotlib.pyplot as plt

var = load_data("BGSEc")                  # dataset key from get_timeseries
param_card(var, "cda/AC_H0_MFI/BGSEc")
t, data = var.time, clean(var.values)     # data is (N,) or (N, components)
export("n_points", len(t))
export("units", str(var.unit))

if data.ndim > 1:                         # vectors: one component per row, sharex
    labels = ["Bx", "By", "Bz"] if data.shape[1] == 3 else [f"C{i}" for i in range(data.shape[1])]
    fig, axes = plt.subplots(data.shape[1], 1, figsize=(12, 2.5 * data.shape[1]), sharex=True)
    for ax, i, lbl in zip(axes, range(data.shape[1]), labels):
        ax.plot(t, data[:, i], linewidth=0.8); ax.set_ylabel(f"{lbl} ({var.unit})")
    axes[-1].set_xlabel("Time")
else:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, data, linewidth=0.8); ax.set_ylabel(str(var.unit)); ax.set_xlabel("Time")
plt.tight_layout()
plt.show()
```

**FFT / PSD:** use `scipy.signal.welch(signal, fs=1/dt)` with `dt` from the median time delta, or the
`power_spectrum` tool. Plot loglog and `export` the peak frequency.

If `get_timeseries` returns a `quality` block with `notable: true`, report it (missing %, gaps,
5σ outliers) — these are deterministic checks, not guesses.

---

## Cross-mission comparison
- `search_parameters` once per mission with explicit spacecraft names; never reuse one id across missions.
- Download each mission with get_timeseries, then in ONE run_python load all via `load_data()`.
- Align on a common grid with pandas `resample`/`reindex`; normalize units before any cross-mission arithmetic.
- State the id used per mission. If a mission has no data, report it — never silently drop it.
- L1 propagation delay: `delay_s = 1.5e6 / V_sw_km_s`; shift on the resampled grid.
- 4 MMS spacecraft = one tetrahedron (curlometer), not independent missions.

## Superposed epoch (catalog → SEA)
1. `get_events_timeseries(catalog_id, param_id, start, stop)` — persists all events, returns a `dataset` key.
2. `load_recipe("superposed_epoch")`.
3. run_python: `events = load_data("<param_last_segment>_events")` (list of `ns(time, values, start, stop)`),
   set `component` (0/1/2 for Bx/By/Bz, scalar handled too), paste the recipe. Never re-fetch — events are already persisted.

## Event detection
Implement threshold / derivative / boundary criteria in run_python; report event times and key
signature values (ΔP/P, ΔB/B…).

| Event | Signatures | Parameters |
|---|---|---|
| IP shock | jump in n, V, B, T; dynamic-pressure increase | B, n_p, V_sw, T_p |
| Reconnection | B reversal + V jet + current sheet | B, V_ion, n_e |
| CME / flux rope | smooth B rotation, enhanced B, low T | B, T_p, V_sw |
| SIR / CIR | velocity ramp, density pile-up, B compression | V_sw, n_p, B |
| Magnetopause crossing | B sign change, density jump | B, n_e, V_ion |

SPASE PhenomenonType: `InterplanetaryShock`, `MagneticCloud`, `CoronalMassEjection`,
`StreamInteractionRegion`, `MagnetopauseCrossing`, `Substorm`.
