---
name: data_analyst
description: Download and analyze heliophysics time series — statistics, FFT, multi-mission comparison, event detection, plotting.
when_to_use: The user wants to retrieve data, compute statistics, plot a time series, compare multiple missions, detect plasma events (shocks, reconnection, CME, SIR), or run any numerical analysis on speasy parameters.
allowed_tools: [search_parameters, get_timeseries, get_events_timeseries, load_recipe, run_recipe, run_python]
---

# Procedure — analyze a time series

## RULE ZERO — run_python is the only tool that makes figures and numbers
A text description of a plot is not a figure. To produce a figure, call run_python with `plt.show()`.

## RULE ZERO-BIS — a computation with a recipe is NEVER hand-written, even when you already know the formula
Knowing the physics is not the point — for any task in the table below, run the recipe as shipped
with `run_recipe(name, inputs)`, unconditionally. `inputs` binds what the recipe reads, each value
a Python expression evaluated in the sandbox: `run_recipe("theta_bn", inputs={"B": "load_data('b')",
"shock_time": "np.datetime64('2015-03-17T04:00:00')"})` — the recipe derives its own averaging
windows, computes, and exports; its exports are the numbers you report. For a recipe that is a
library of functions (`rankine_hugoniot`, `shock_timing_2sc`, `pressure_balance`) add `call`, one
expression applying its function to the inputs. `load_recipe(name)` is for *reading* a recipe when
you need to know what it expects; do not paste its source into `run_python`. A recipe carries
calibrated parameters (averaging windows, physical constants) and a self-test that code written
from memory does not have. Getting the formula right from memory and still being wrong is exactly
how this table earned its entries: a hand-written Rankine-Hugoniot on this same event guessed an
eV→K conversion instead of using the constant, picked averaging windows the recipe's own
calibration table flags as the worst combination, and landed 10% off a compression ratio the recipe
gets exactly; three runs of θ_Bn on one shock, windows chosen by hand, gave 54.85°, 59.95° and
64.27°.

## RULE ONE — download outside the sandbox, always
Call `get_timeseries` (or `get_events_timeseries`) BEFORE `run_python` — the sandbox has a 60 s
timeout that speasy downloads blow past. The result carries a `dataset` key; read it inside
run_python with `load_data("name")`. Never call `spz.get_data()` in the sandbox for data you can
fetch first; use it only for data the loop has not already downloaded.

## RULE TWO — which recipe for which task (see RULE ZERO-BIS: never optional)
Write custom code only when no recipe matches — recipes carry their scientific reference, so
using one also gives you provenance.

| Task | Recipe |
|---|---|
| Shock normal angle θ_Bn | `theta_bn` — bind `B` (the downloaded series) and `shock_time`; the recipe derives 8-minute windows clear of the ramp and refuses a window that contains it. Bind `B_up`/`B_dn` yourself only when the user specifies the windows. |
| Discontinuity / current-sheet normal (minimum variance) | `mvab` |
| Shock jump conditions, compression ratio, shock speed | `rankine_hugoniot` — **also picks the upstream/downstream averaging windows**; call `upstream_downstream(t, values, shock_time)` per quantity and never pass averages you computed yourself. Choosing those windows by hand is where this analysis goes wrong: a generous guard band with a long window sounds careful, lands in the decaying sheath, and returns a compression of 1.89 instead of 2.59 with every downstream number wrong. |
| Rotational vs tangential discontinuity | `walen_test` |
| Shock timing between TWO spacecraft | `shock_timing_2sc` — two spacecraft do **not** determine a shock normal (that needs four). Pass the normal from `theta_bn` or `mvab`; deriving it from the separation and the lag makes the transverse separation come out as exactly 0 km for any input, and that tautology has already been published as a geometrical result. |
| Magnetopause standoff distance (pressure balance) | `pressure_balance` |
| Particle pitch-angle distribution | `pitch_angle_dist` |
| Superposed epoch analysis over a catalog | `superposed_epoch` |

When you write custom code for a computation with no recipe, call
`document_method("<name>", reference="<paper/formula>", method="<one line>")` in run_python so the
method is recorded with its source.

## Sandbox helpers (provided, no import needed)
- `load_data("name")` → `ns(time, values, columns, units, param_id, missing_pct)` for a dataset from get_timeseries. `units` is plural — there is no `.unit`. `time` is a numpy datetime64 array, so it has no `.isoformat()`: use `str(t[i])`.
- `param_card(var, param_id)` → metadata card in the UI. Call once per downloaded parameter.
- `clean(var.values)` → masks CDF fill values (`-1e31`, `9.96e36`) to NaN. Always wrap `.values`.
- `magnitude(vectors)` → |B| or |V| of an N×3 array. Use it. `np.sqrt(np.nansum(v**2, axis=1))`
  turns a data gap into a magnitude of exactly 0, and `np.diff` then reads the recovery out of
  that hole as the biggest jump in the interval — that is a shock detector locking onto a data
  gap, and it has already published a shock time 3.5 minutes early.
- `export(name, value, units="nT")` → surfaces a key number in the reply and records it in the
  session provenance ledger. Call for every result that matters, with its unit when it has one:
  a number that is never exported has no traceable origin.
- `save_path("name.ext")` → the only correct way to build a path for a file you write yourself
  (a standalone script, a CSV). NEVER hardcode a path or guess where "the workspace" is — only
  this exact directory is writable; a path one level off can accept the write and still lose the
  file, because everything else is a read-only overlay that looks writable and is not.
- `plt.show()` → REQUIRED; it is what saves the figure to disk.

## Resolve the id first (if needed) — two searches, then download
If the id is missing, vague, or malformed (extra path segments like `cda/ACE/MAG/AC_H0_MFI/...`),
`search_parameters` with a plain-English query — one batched call for all the parameters you
need — and pick from what comes back. The top hit of each query carries `dataset_variables`:
every variable its dataset holds. **Read that list before searching again**: a variable that is
not in it does not exist under that dataset (Wind SWE has no `Proton_Temp`; its temperature is
`Proton_W_nonlin`, a thermal speed; its velocity is `Proton_VX/VY/VZ_nonlin`, not a vector). A
hit's `also_in` lists the same variable in other cadences and field models.

Budget: **two searches**. If the exact product is still not there, take the best dataset you
were shown and use its variables as they are named. **Never assemble an id by hand** — the
guessed `MMS1_MEC_SRVY_L2/mms1_mec_r_gsm` dropped the `_EPHT89D` the real dataset carries and
failed; the correct id was in the results of the first search. Past the budget with nothing
downloaded, you receive a correction listing the ids you already have: use them.

Spacecraft position: `ssc/<spacecraft>` (e.g. `ssc/mms1`, `ssc/wind`) is the SSCWeb trajectory
in km — GSE by default. Prefer it to an instrument's own ephemeris variable.

## Canonical template — download then plot
```python
import numpy as np
import matplotlib.pyplot as plt

var = load_data("BGSEc")                  # dataset key from get_timeseries
param_card(var, "cda/AC_H0_MFI/BGSEc")
t, data = var.time, clean(var.values)     # data is (N,) or (N, components)
export("n_points", len(t))
export("units", str(var.units))

if data.ndim > 1:                         # vectors: one component per row, sharex
    labels = ["Bx", "By", "Bz"] if data.shape[1] == 3 else [f"C{i}" for i in range(data.shape[1])]
    fig, axes = plt.subplots(data.shape[1], 1, figsize=(12, 2.5 * data.shape[1]), sharex=True)
    for ax, i, lbl in zip(axes, range(data.shape[1]), labels):
        ax.plot(t, data[:, i], linewidth=0.8); ax.set_ylabel(f"{lbl} ({var.units})")
    axes[-1].set_xlabel("Time")
else:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, data, linewidth=0.8); ax.set_ylabel(str(var.units)); ax.set_xlabel("Time")
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
- Relative geometry between spacecraft — which is upstream, sunward, closer, hit first — is
  a FACT YOU FETCHED, never one you recall from what a mission is usually for. State the
  coordinates next to the claim; if they disagree with it, the claim is wrong. Two probes
  sharing an orbit region swap their ordering over a mission, so no pair has a fixed answer.
- Frames: GSE/GSM are geocentric, +X toward the Sun — larger X is sunward, hit first by a
  radial front. HEE/HCI are heliocentric — distance from the Sun orders them. RTN is
  observer-centred, so R is radial from the Sun through that spacecraft, not a common axis.

## Superposed epoch (catalog → SEA)
1. `get_events_timeseries(catalog_id, param_id, start, stop)` — persists all events, returns a `dataset` key.
2. `run_recipe("superposed_epoch", inputs={"events": "load_data('<param_last_segment>_events')", "component": 2})`
   — `component` 0/1/2 for Bx/By/Bz (a scalar is handled). The recipe keeps a data gap a gap and
   exports the median, quartiles, a bootstrap CI and the count of contributing events per epoch.
   Never re-fetch — events are already persisted.

## Event detection
For an interplanetary shock whose time you do not know yet: download B, then
`run_recipe("theta_bn", inputs={"B": "load_data('<b>')"})` **with no `shock_time`** — the recipe
lists the largest |B| jumps of the interval (time, jump, ratio) and stops. Look at them against
the plot, name the one you take **and the others you rejected** in your report (the question
"around 2004-11-07" has several jumps that day; say which is the shock and why: n, V, T jump
with it), then run the recipe again with `shock_time` set. Do not hunt the ramp with a series
of hand-written `run_python` cells: one run spent nine turns doing that.

For other events, implement threshold / derivative / boundary criteria in run_python; report
event times and key signature values (ΔP/P, ΔB/B…).

| Event | Signatures | Parameters |
|---|---|---|
| IP shock | jump in n, V, B, T; dynamic-pressure increase | B, n_p, V_sw, T_p |
| Reconnection | B reversal + V jet + current sheet | B, V_ion, n_e |
| CME / flux rope | smooth B rotation, enhanced B, low T | B, T_p, V_sw |
| SIR / CIR | velocity ramp, density pile-up, B compression | V_sw, n_p, B |
| Magnetopause crossing | B sign change, density jump | B, n_e, V_ion |

SPASE PhenomenonType: `InterplanetaryShock`, `MagneticCloud`, `CoronalMassEjection`,
`StreamInteractionRegion`, `MagnetopauseCrossing`, `Substorm`.
