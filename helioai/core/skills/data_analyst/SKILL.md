---
name: data_analyst
description: Download and analyze heliophysics time series — statistics, FFT, multi-mission comparison, event detection, plotting.
when_to_use: The user wants to retrieve data, compute statistics, plot a time series, compare multiple missions, detect plasma events (shocks, reconnection, CME, SIR), or run any numerical analysis on speasy parameters.
allowed_tools: [search_parameters, run_python]
---

# Procedure — analyze a time series

## RULE ZERO

**run_python is the only tool that produces figures and numerical results.**
A text description of a plot is not an actual figure. To produce a figure you must call run_python with plt.show().
You have no get_timeseries tool. All data access happens inside run_python via spz.get_data().

**RULE: never re-download a persisted dataset.** If the agent loop returned a `dataset` key (e.g. from `get_timeseries` or `get_events_timeseries`), access it in run_python via `load_data("name")` — never call `spz.get_data()` again for the same data. The sandbox already has `load_data` available; no import needed.

## 1. Resolve the parameter id (if needed)

If the id is missing, vague, or looks malformed (e.g. extra path segments like `cda/ACE/MAG/AC_H0_MFI/...`),
call `search_parameters` first with a plain-English query (e.g. `"ACE magnetic field GSE components"`).
Pick the shortest matching id from the results.

## 2. Call run_python immediately

Do not add intermediate steps. Call run_python with the full analysis code.
The sandbox has speasy (spz), numpy (np), matplotlib (plt), scipy, astropy.

### Template — time series plot

```python
import speasy as spz
import numpy as np
import matplotlib.pyplot as plt

var = spz.get_data("cda/AC_H0_MFI/BGSEc", "2008-01-01T00:00:00", "2008-01-02T00:00:00")
param_card(var, "cda/AC_H0_MFI/BGSEc")  # displays metadata card in UI
t = var.time
data = clean(var.values)  # shape (N,) or (N, components) — masks CDF fill values

export("n_points", len(t))
export("units", str(var.unit))

if data.ndim > 1:
    labels = ["Bx", "By", "Bz"] if data.shape[1] == 3 else [f"C{i}" for i in range(data.shape[1])]
    fig, axes = plt.subplots(data.shape[1], 1, figsize=(12, 2.5 * data.shape[1]), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t, data[:, i], linewidth=0.8)
        ax.set_ylabel(f"{label} ({var.unit})")
    axes[-1].set_xlabel("Time")
else:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, data, linewidth=0.8)
    ax.set_ylabel(str(var.unit))
    ax.set_xlabel("Time")

plt.suptitle("ACE IMF — 2008-01-01")
plt.tight_layout()
plt.show()  # REQUIRED — saves the figure to disk
```

### Template — FFT / Power spectral density

```python
import speasy as spz
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

var = spz.get_data("PARAM_ID", "START", "STOP")
param_card(var, "PARAM_ID")
t = var.time
data = clean(var.values)
signal = np.nanmean(data, axis=1) if data.ndim > 1 else data
dt = float(np.median(np.diff(t.astype("int64"))) * 1e-9)

f, Pxx = welch(signal[~np.isnan(signal)], fs=1.0/dt, nperseg=256)
export("peak_freq_Hz", float(f[np.argmax(Pxx)]))
export("sampling_dt_s", dt)

fig, ax = plt.subplots(figsize=(10, 4))
ax.loglog(f[1:], Pxx[1:])
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD")
plt.tight_layout()
plt.show()
```

## 3. Rules

- **plt.show() is mandatory** — it is the only way to save a figure to disk.
- **param_card(var, param_id)** — call immediately after `spz.get_data()` to display a metadata card. Always call it for every downloaded parameter.
- **clean(var.values)** — always wrap `var.values` in `clean()`. speasy/CDAWeb return CDF fill values (`-1e31`, `9.96e36`) for data gaps that destroy the Y-axis scale. `clean()` converts them to NaN, which matplotlib renders as gaps.
- **export(name, value)** for every key number so the LLM sees it in the response.
- If spz.get_data() raises an error, try a corrected id (strip extra path segments, e.g. `cda/AC_H0_MFI/BGSEc` instead of `cda/ACE/MAG/AC_H0_MFI/BGSEc`).
- For vectors: plot components separately (Bx, By, Bz) with sharex=True.
- Units come from var.unit. State them in the reply.

---

## Cross-mission comparison

### Resolve one parameter id per mission

Use `search_parameters` once per mission with explicit spacecraft names in the query. Do not reuse the same id for different missions.

### Download and align on a common grid

```python
import speasy as spz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

t_start, t_stop = "2005-01-17T00:00:00", "2005-01-18T00:00:00"

param_ids = {"ACE": "cda/AC_H0_MFI/BGSEc", "Wind": "cda/WI_H0_MFI/BGSEc"}
datasets = {name: spz.get_data(pid, t_start, t_stop) for name, pid in param_ids.items()}
for name, var in datasets.items():
    if var is not None:
        param_card(var, param_ids[name])

t_common = pd.date_range(t_start, t_stop, freq="1min")
aligned = {}
for name, var in datasets.items():
    if var is None or len(var.time) == 0:
        export(f"{name}_ok", False)
        continue
    df = pd.DataFrame(clean(var.values), index=pd.to_datetime(var.time))
    aligned[name] = df.resample("1min").mean().reindex(t_common)
    export(f"{name}_ok", True)

n = len(aligned)
fig, axes = plt.subplots(n, 1, sharex=True, figsize=(12, 3 * n))
if n == 1:
    axes = [axes]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
for ax, (name, df), color in zip(axes, aligned.items(), colors):
    if df.shape[1] == 3:
        for col, lbl in zip(df.columns, ["x", "y", "z"]):
            ax.plot(df.index, df[col], label=lbl, alpha=0.8)
        ax.legend(fontsize=8)
    else:
        ax.plot(df.index, df.iloc[:, 0], color=color)
    ax.set_ylabel(name)
axes[0].set_title("Cross-mission comparison")
plt.tight_layout()
plt.show()
```

### Propagation delay (L1 spacecraft)

```python
V_sw_km_s = 400.0
d_L1_km = 1.5e6
delay_s = d_L1_km / V_sw_km_s
export("propagation_delay_min", round(delay_s / 60, 1))
# Shift dataset: df_shifted = df.shift(periods=int(delay_s / 60))  # 1-min grid
```

### Rules for multi-mission work

- Always state which param id was used for each mission.
- If one mission has no data, report it — never silently drop it.
- Normalize units before cross-mission arithmetic.
- For 4 MMS spacecraft: treat as tetrahedral configuration (curlometer), not independent missions.

---

## Superposed epoch analysis (catalog → SEA)

**Full workflow:**
1. `get_events_timeseries(catalog_id, param_id, start, stop)` — downloads + persists all events, returns `dataset` key.
2. `load_recipe("superposed_epoch")` — retrieve the SEA recipe source.
3. In run_python:
```python
events = load_data("<param_id_last_segment>_events")  # list of ns(time, values, start, stop)
component = 2   # e.g. Bz for a 3-component field
units = "nT"
param_label = "Bz GSM"
# ... paste recipe source below (epoch_median, IQR figure, export calls) ...
```

**Never re-fetch with spz.get_data()** — the events are already persisted from step 1.
Multi-component data: set `component=0/1/2` for Bx/By/Bz; recipe handles both scalar and vector inputs.

---

## Event detection

### Event signatures

| Event | Signatures | Parameters needed |
|---|---|---|
| **IP shock** | Jump in n, V, B, T; pressure increase | B vector, n_p, V_sw, T_p |
| **Magnetic reconnection** | B reversal + V jet + current sheet | B vector, V_ion, n_e |
| **CME / flux rope** | Smooth B rotation, enhanced B, low T | B vector, T_p, V_sw |
| **SIR / CIR** | Velocity ramp, density pile-up, B compression | V_sw, n_p, B |
| **Magnetopause crossing** | B sign change, density jump | B, n_e, V_ion |

### Detection template (IP shock)

```python
n = n_var.values[:, 0]
V = np.abs(V_var.values[:, 0])
Pdyn = n * V**2
window = 10
ratio = np.array([
    Pdyn[i+window:i+2*window].mean() / (Pdyn[i:i+window].mean() + 1e-10)
    for i in range(len(Pdyn) - 2*window)
])
shock_candidates = np.where(ratio > 2.0)[0]
export("n_shock_candidates", len(shock_candidates))
if len(shock_candidates) > 0:
    export("first_shock_time", str(n_var.time[shock_candidates[0] + window]))
```

### Report format

- Number of candidates found, time of most prominent event.
- Key signature values at event time (ΔP/P, ΔB/B…).
- SPASE PhenomenonType: `InterplanetaryShock`, `MagneticCloud`, `CoronalMassEjection`, `StreamInteractionRegion`, `MagnetopauseCrossing`, `Substorm`.
