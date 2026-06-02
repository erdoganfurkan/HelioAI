---
name: plasma_physicist
description: Compute plasma physics quantities using PlasmaPy tools or run_python sandbox.
when_to_use: The user wants to compute derived plasma quantities — plasma beta, gyrofrequency, Debye length, Alfvén speed, inertial length, power spectrum — or needs unit validation for plasma parameters.
allowed_tools: [run_python, get_timeseries, search_parameters]
---

# Procedure — plasma physics calculations

## 1. Choose the right tool

| Task | Use |
|---|---|
| Single-point estimate (β, f_ci, λ_D, V_A, d_i) | `plasma_beta`, `gyrofrequency`, `debye_length`, `alfven_speed`, `inertial_length` directly via `run_python` |
| Time-series of a derived quantity | `get_timeseries` → `run_python` (compute per sample) |
| Power spectral density | `power_spectrum` via `run_python` |
| Custom formula (e.g. mirror mode criterion, firehose) | `run_python` with numpy + PlasmaPy |

## 2. Available PlasmaPy tools in the sandbox

All accept keyword arguments with units. Import from `helioai.tools.plasmapy_tools`.

```python
from helioai.tools.plasmapy_tools import (
    plasma_beta,       # (B_nT, n_cm3, T_eV) → β (dimensionless)
    gyrofrequency,     # (B_nT, particle='p'|'e'|'He2+') → Hz
    debye_length,      # (n_cm3, T_eV) → km
    alfven_speed,      # (B_nT, n_cm3, ion_mass_amu=1.0) → km/s
    inertial_length,   # (n_cm3, particle='p'|'e') → km
    power_spectrum,    # (signal_array, dt_s, nperseg=256) → (freq, psd)
)
```

## 3. Unit conventions — always check before computing

| Quantity | Expected unit | Common mistake |
|---|---|---|
| Magnetic field B | nT | Tesla (×10⁹ conversion needed) |
| Number density n | cm⁻³ | m⁻³ (×10⁻⁶ conversion needed) |
| Temperature T | eV | K (×8.617×10⁻⁵ conversion) |
| Velocity V | km/s | m/s (×10⁻³ conversion) |
| Distance | km | m or AU |

If units come from `get_timeseries`, check the `units` field first. Common sources:
- AMDA: nT for B, cm⁻³ for density, eV for temperature ✓
- CDA/SPDF: can be in SI — check CATDESC or units field carefully

## 4. Single-point calculation template

```python
from helioai.tools.plasmapy_tools import plasma_beta, alfven_speed, gyrofrequency

# Solar wind reference values
B_nT = 5.0      # nT
n_cm3 = 10.0    # cm⁻³
T_eV = 10.0     # eV

beta = plasma_beta(B_nT, n_cm3, T_eV)
Va = alfven_speed(B_nT, n_cm3)
fci = gyrofrequency(B_nT, particle='p')

export("plasma_beta", beta)
export("alfven_speed_km_s", Va)
export("ion_gyrofreq_Hz", fci)
```

## 5. Time-series derived quantity template

```python
import speasy as spz
import numpy as np
from helioai.tools.plasmapy_tools import plasma_beta

B_var = spz.get_data("amda/ace_imf_all", "2005-01-01", "2005-01-02")
n_var = spz.get_data("amda/ace_swe_n", "2005-01-01", "2005-01-02")

# Align on B time grid (interpolate density)
B_mag = np.linalg.norm(B_var.values, axis=1)
n_interp = np.interp(
    B_var.time.astype("int64"),
    n_var.time.astype("int64"),
    n_var.values[:, 0],
)

beta_ts = np.array([plasma_beta(B, n, T_eV=10.0) for B, n in zip(B_mag, n_interp)])

export("beta_mean", float(np.nanmean(beta_ts)))
export("beta_max", float(np.nanmax(beta_ts)))
export("beta_gt_1_fraction", float(np.mean(beta_ts > 1.0)))
```

## 6. Physical sanity checks

After any calculation, verify against these reference ranges:

| Region | B (nT) | n (cm⁻³) | T_p (eV) | β |
|---|---|---|---|---|
| Solar wind @ 1AU | 5–10 | 5–15 | 5–20 | ~1–5 |
| Magnetosheath | 10–30 | 10–50 | 20–100 | ~1–3 |
| Magnetosphere (lobe) | 20–50 | 0.01–0.1 | 100–1000 | ≪1 |
| Inner heliosphere (0.1AU) | 100–500 | 100–1000 | 50–200 | ~1 |

If a result is off by orders of magnitude, suspect a unit mismatch.
