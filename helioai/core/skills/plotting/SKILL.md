---
name: plotting
description: Produce publication-quality heliophysics matplotlib plots inside run_python.
when_to_use: The user asks for a plot, figure, time series visualization, spectrum, hodogram, or any graphical output.
allowed_tools: [run_python]
---

# Procedure — produce a heliophysics matplotlib figure

## 1. Always call clean(), export(), and plt.show()

Before plotting, always pass `var.values` through `clean()` to mask CDF fill values
(`-1e31`, `9.96e36`) and infinities as NaN. matplotlib renders NaN as gaps in the line,
preserving the correct Y-axis scale.

```python
data = clean(var.values)           # → masks fill values / infinities as NaN
export("B_stats", data)            # → min/max/mean/std surfaced to the LLM
plt.show()                         # → captures the PNG to disk
```

## 2. Dark-style time-series template

Standard template for a multi-panel time series (most common heliophysics case):

```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

plt.style.use("dark_background")
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8), sharex=True)
fig.subplots_adjust(hspace=0.05)

# Panel 0 — magnetic field components
ax = axes[0]
ax.plot(times, Bx, label="Bx", color="#4fc3f7", lw=0.8)
ax.plot(times, By, label="By", color="#ef9a9a", lw=0.8)
ax.plot(times, Bz, label="Bz", color="#a5d6a7", lw=0.8)
ax.axhline(0, color="white", lw=0.4, ls="--")
ax.set_ylabel("|B| (nT)", fontsize=9)
ax.legend(fontsize=7, loc="upper right")
ax.grid(alpha=0.2)

# Panel 1 — scalar quantity
ax = axes[1]
ax.plot(times, density, color="#ffcc80", lw=0.8)
ax.set_ylabel("n (cm⁻³)", fontsize=9)
ax.grid(alpha=0.2)

# Panel 2 — velocity
ax = axes[2]
ax.plot(times, speed, color="#ce93d8", lw=0.8)
ax.set_ylabel("V (km/s)", fontsize=9)
ax.grid(alpha=0.2)

# Time axis formatting
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
fig.autofmt_xdate(rotation=0, ha="center")
fig.suptitle("Solar wind — ACE 2005-01-17", fontsize=11, y=0.98)

plt.tight_layout()
plt.show()
```

## 3. speasy datetime axis

speasy returns numpy datetime64 arrays — convert before plotting:

```python
import matplotlib.dates as mdates
import pandas as pd

times = pd.to_datetime(var.time)     # SpezVariable → DatetimeIndex
# or: matplotlib.dates.num2date(mdates.date2num(var.time))
```

## 4. PSD / power spectrum template

```python
from scipy import signal

f, psd = signal.welch(values, fs=1.0 / dt_s, nperseg=256)
f_nonzero = f[1:]
psd_nonzero = psd[1:]

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8, 4))
ax.loglog(f_nonzero, psd_nonzero, color="#4fc3f7", lw=0.9)
ax.set_xlabel("Frequency (Hz)", fontsize=9)
ax.set_ylabel("PSD (nT²/Hz)", fontsize=9)
ax.set_title("Magnetic field PSD", fontsize=10)
ax.grid(alpha=0.2, which="both")
plt.tight_layout()
plt.show()

export("psd_peak_hz", np.array([f_nonzero[np.argmax(psd_nonzero)]]))
```

## 5. Hodogram template

Used to determine polarisation / min-variance direction:

```python
plt.style.use("dark_background")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(Bx, By, color="#4fc3f7", lw=0.6, alpha=0.8)
axes[0].set_xlabel("Bx (nT)"); axes[0].set_ylabel("By (nT)")
axes[0].set_title("Bx–By plane"); axes[0].set_aspect("equal"); axes[0].grid(alpha=0.2)

axes[1].plot(By, Bz, color="#ef9a9a", lw=0.6, alpha=0.8)
axes[1].set_xlabel("By (nT)"); axes[1].set_ylabel("Bz (nT)")
axes[1].set_title("By–Bz plane"); axes[1].set_aspect("equal"); axes[1].grid(alpha=0.2)

plt.suptitle("Hodogram — MMS1 FGM", fontsize=10)
plt.tight_layout()
plt.show()
```

## 6. Style rules

| Rule | Why |
|---|---|
| `plt.style.use("dark_background")` | Consistent with the Web UI dark theme |
| `lw=0.8` or `lw=0.9` for time series | Avoid fat lines on dense data |
| `grid(alpha=0.2)` | Subtle grid, readable |
| `fontsize=9` for axis labels | Dense multi-panel figures |
| `tight_layout()` before `show()` | Prevent label clipping |
| `sharex=True` on multi-panel | Linked zoom in interactive viewers |

## Output format

After running, the sandbox returns:
- `figure_paths` — list of saved PNG paths (displayed in the UI automatically)
- `exports` — dict of numerical summaries from `export()` calls

Report the key numerical findings from `exports` in your reply. Never claim to "see" the figure content — describe what the numbers say.
