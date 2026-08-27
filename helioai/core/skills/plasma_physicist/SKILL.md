---
name: plasma_physicist
description: Compute plasma physics quantities using PlasmaPy tools or run_python sandbox.
when_to_use: The user wants to compute derived plasma quantities — plasma beta, gyrofrequency, Debye length, Alfvén speed, inertial length, power spectrum — or needs unit validation for plasma parameters.
allowed_tools: [run_python, search_parameters, list_recipes, load_recipe]
---

# Procedure — plasma physics calculations

## RULE ZERO — recipe before your own derivation

For a standard, named computation, call `load_recipe(name)`, adapt it to your numbers,
and paste it into `run_python`. Recipes carry their scientific reference, so the result
is attributable rather than improvised — and they have already settled the frame and
sign conventions that are easy to get wrong from memory.

| Task | Recipe |
|---|---|
| Shock jump conditions, compression ratio, shock speed | `rankine_hugoniot` |
| Shock normal angle θ_Bn | `theta_bn` |
| Rotational vs tangential discontinuity | `walen_test` |
| Discontinuity / current-sheet normal (minimum variance) | `mvab` |
| Magnetopause standoff from solar wind pressure | `pressure_balance` |
| Pitch angle distribution | `pitch_angle_dist` |

`load_recipe` returns the **source code** — it is not a function you can call inside the
sandbox. Read it, adapt the variable names to your data, and include it in your
`run_python` code.

Use `list_recipes()` when unsure what exists. Write your own derivation only when no
recipe matches, and say so explicitly in the answer.

⚠️ **Shock speed is the classic trap.** `V_shock = V_d · r/(r−1)` assumes the upstream
plasma is at rest. In the spacecraft frame the solar wind is flowing at ~400 km/s, and
that form inflates the answer by ~45 %. The `rankine_hugoniot` recipe derives it from
mass-flux conservation instead. The Alfvénic Mach number then needs the shock speed **in
the upstream frame** — `(V_shock − V_upstream)/V_A`, not `V_shock/V_A`.

## 1. Choose the right tool

| Task | Use |
|---|---|
| Standard named computation | `load_recipe` → `run_python` (see RULE ZERO) |
| Single-point estimate (β, f_ci, λ_D, V_A, d_i) | `plasma_beta`, `gyrofrequency`, `debye_length`, `alfven_speed`, `inertial_length` directly via `run_python` |
| Time-series of a derived quantity | `load_data("name")` in `run_python`, computing per sample |
| Power spectral density | `power_spectrum` via `run_python` |
| Custom formula (e.g. mirror mode criterion, firehose) | `run_python` with numpy + PlasmaPy |

This role cannot download. The lead agent fetches the data before delegating; reach it
with `load_data("name")`, whose fill values are already NaN — use `np.nanmean` and
friends rather than filtering on magnitude.

## 2. Available PlasmaPy tools in the sandbox

`plasmapy.formulary` is pre-imported as `pf`, and `astropy.units` is pre-imported as `u`. Use `pf` directly:

```python
# Quantities accept astropy units attached via * u.<unit>
beta = pf.beta(T=T_eV * u.eV, n=n_cm3 * u.cm**-3, B=B_nT * u.nT)  # dimensionless
fci = pf.gyrofrequency(B=B_nT * u.nT, particle="p+").to(u.Hz)       # Hz
lambda_D = pf.Debye_length(T=T_eV * u.eV, n_e=n_cm3 * u.cm**-3).to(u.km)  # km
Va = pf.Alfven_speed(B=B_nT * u.nT, density=n_cm3 * u.cm**-3).to(u.km / u.s)  # km/s
di = pf.inertial_length(n=n_cm3 * u.cm**-3, particle="p+").to(u.km)  # km
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

Multi-species note: $\beta_{total} = \beta_i + \beta_e$. In the solar wind $T_i/T_e \sim 2\text{--}5$, and in the magnetosheath $T_i/T_e \sim 7$. A single temperature scalar underestimates $\beta_{total}$.

## 4. Single-point calculation template

```python
# pf (plasmapy.formulary) and u (astropy.units) are already imported

# Solar wind reference values
B_nT = 5.0      # nT
n_cm3 = 10.0    # cm⁻³
T_eV = 10.0     # eV

beta = float(pf.beta(T=T_eV * u.eV, n=n_cm3 * u.cm**-3, B=B_nT * u.nT))
Va = float(pf.Alfven_speed(B=B_nT * u.nT, density=n_cm3 * u.cm**-3).to_value(u.km / u.s))
fci = float(pf.gyrofrequency(B=B_nT * u.nT, particle="p+").to_value(u.Hz))

export("plasma_beta", beta)
export("alfven_speed_km_s", Va)
export("ion_gyrofreq_Hz", fci)
```

## 5. Time-series derived quantity template

```python
import numpy as np

# The lead agent downloaded these; load them, never spz.get_data here.
B_var = load_data("imf_all")
n_var = load_data("swe_n")

# Align on B time grid using interp_to (handles gaps and arrays cleanly)
B_mag = magnitude(B_var.values)
n_interp = interp_to(B_var.time, n_var.time, n_var.values[:, 0])

# Compute beta vectorized (Astropy unit instantiation in a loop is too slow)
valid = np.isfinite(B_mag) & np.isfinite(n_interp) & (B_mag > 0) & (n_interp > 0)
beta_ts = np.full_like(B_mag, np.nan)
beta_ts[valid] = pf.beta(T=10.0 * u.eV, n=n_interp[valid] * u.cm**-3, B=B_mag[valid] * u.nT).value

# nan-aware throughout: fill values arrive as NaN, so plain mean/max return NaN
export("beta_mean", float(np.nanmean(beta_ts)))
export("beta_max", float(np.nanmax(beta_ts)))

# A fraction has to select the finite samples BEFORE comparing. `np.nanmean(beta_ts > 1.0)`
# reads as nan-aware and is not: `NaN > 1.0` is False, so every gap has already been filed
# under "does not exceed" and nanmean has no NaN left to skip. On a series half missing it
# returns 0.25 where the measured fraction is 0.5 — an answer that is wrong in the safe
# direction, which is the kind nobody checks.
finite_beta = beta_ts[np.isfinite(beta_ts)]
export("beta_gt_1_fraction", float(np.mean(finite_beta > 1.0)))
export("beta_n_samples", float(finite_beta.size))
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

## 7. Derive it twice when you can

A single number has nothing to contradict it. Where two quantities are physically
linked, compute both and report the comparison — that is what catches an error a
plausible-looking value will not reveal on its own:

| Pair | Relation |
|---|---|
| Density compression r_n vs magnetic compression r_B | Independent measurements of the same jump; they should agree |
| Compression ratio vs Alfvénic Mach number | MHD ties them: r = (γ+1)M²/((γ−1)M²+2) |
| β from (B, n, T) vs β from V_A and c_s | Same quantity by two routes |
| Shock speed from jump conditions vs from multi-spacecraft timing | Agree only for a front normal to the separation |

State the disagreement when there is one and say which number you trust. An
unchecked number presented confidently is worse than a checked number with a caveat.
