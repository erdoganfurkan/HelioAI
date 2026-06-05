---
name: helioai_helper
description: Guide the user on HelioAI capabilities, available missions, typical analyses, and how to formulate effective queries.
when_to_use: The user asks "help", "what can you do", "what missions", "how do I…", or any meta-question about HelioAI usage.
---

# Procedure — answer a meta-question about HelioAI

## 1. Available data (70+ missions via speasy)

Three main providers — call `list_missions()` for the live catalogue.

| Provider | Key missions | Coverage |
|---|---|---|
| **AMDA** (CDPP) | Cluster (C1-C4), MMS (1-4), Solar Orbiter (SolO), WIND, ACE, Cassini, MEX, VEX, STEREO A/B, Helios | 1970–present, EU focus |
| **CDAWeb** (NASA) | MMS, THEMIS (TH-A to TH-E), Van Allen Probes (RBSP), Parker Solar Probe (PSP), ACE, WIND, Ulysses, Voyager 1/2, Juno, MAVEN | 1970–present, NASA focus |
| **CSA** (ESA) | Cluster (C1-C4), Double Star, Solar Orbiter, Mars Express | ESA missions |

## 2. Common parameters by region

| Region | Typical parameters | Key missions |
|---|---|---|
| Solar wind (L1) | Bulk velocity, density, temperature, IMF (Bx/By/Bz/|B|) | ACE, WIND, PSP |
| Magnetosheath | |B|, density (high), temperature, flow speed | MMS, Cluster |
| Magnetopause | Reconnection jets, Bz reversal, electron jets | MMS (< 150 RE orbit) |
| Magnetosphere | Energetic particles, wave activity (chorus/hiss), ring current | Van Allen Probes, THEMIS |
| Ionosphere | Field-aligned currents, ion outflow | SWARM, FAST |
| Outer heliosphere | Pickup ions, termination shock | Voyager 1/2 |

## 3. Typical analysis workflows

### 3.1 Find and download a parameter
```
User: "solar wind density from ACE in January 2005"
→ parameter_hunter: search_parameters("solar wind ion number density ACE")
→ data_analyst: get_timeseries(param_id, "2005-01-01T00:00:00", "2005-01-31T00:00:00")
```

### 3.2 Interplanetary shock detection
```
User: "find IP shocks in WIND data 2000"
→ data_analyst (event_detector mode):
  1. Download B, Vp, Np, Tp for the interval
  2. Compute |B| jump, density compression ratio, velocity step
  3. sliding-window: ΔN/N > 1.5, ΔV > 50 km/s, ΔB > 3 nT → candidate shock
  4. Compute theta_Bn via recipe theta_bn (load_recipe + run_python)
```

### 3.3 Magnetic reconnection identification
```
User: "reconnection events at MMS 2017-07-11"
→ data_analyst:
  1. Download MMS1 FGM (B), FPI (Ve, Vi, Ne) at burst cadence
  2. Look for: Bz reversal, electron jet |Ve| > 1000 km/s, density depletion
  3. Walén test: run walen_test recipe — if ΔV ~ ΔV_A (|slope| ~ 1), rotational discontinuity
  4. Hodogram (load_skill plotting) to identify minimum variance direction
```

### 3.4 Plasma parameter study
```
User: "what is the plasma beta in the magnetosheath?"
→ plasma_physicist: plasma_beta(B_nT=20, n_cm3=20, T_eV=200)  → β ~ 1.5 (high-β)
→ Also: gyrofrequency, Debye length, Alfvén speed for context
```

### 3.5 Cross-mission comparison
```
User: "compare solar wind at L1 and at Saturn"
→ parameter_hunter: resolve IMF/density for ACE + Cassini
→ data_analyst: get both time series, resample to common cadence, plot side-by-side
```

### 3.6 Statistical study with an event catalog
```
User: "IMF Bz during all ICMEs in 2005 — Richardson & Cane catalog"
→ list_catalogs(region="ICME") → find id for Richardson & Cane list
→ get_catalog(id, start="2005-01-01", stop="2005-12-31") → inspect events
→ parameter_hunter: resolve IMF Bz param id for ACE or WIND
→ get_events_timeseries(catalog_id, param_id, "2005-01-01", "2005-12-31")
  → returns per-event stats (mean/std/min/max) + saves data
→ run_python: stack plot, epoch-aligned superposition, histogram
```

### 3.7 Superposed epoch analysis
```
User: "proton density at MMS bow-shock crossings in 2017"
→ list_catalogs(region="bow shock") → MMS_Lalti_BScrossings (2797 events)
→ get_events_timeseries(id, "amda/mms1_dis_ni_fast", "2017-01-01", "2017-12-31", max_events=50)
→ run_python: epoch-align on crossing time, stack-plot, compute mean profile
```

## 4. Formulating effective queries

| Instead of | Try |
|---|---|
| "get B" | "download magnetic field vector from MMS1 FGM for 2017-07-11 12:00–13:00" |
| "show solar wind" | "solar wind bulk velocity and proton density from WIND for January 2005" |
| "reconnection at MMS" | "MMS1 FPI electron jet and FGM B reversal 2017-07-11T22:30–22:40" |
| "spectrum" | "power spectral density of Bz from Cluster C1 2003-11-20 00:00–02:00" |

**Rule**: always give mission, parameter type, and time range (ISO 8601). The more specific, the fewer disambiguation rounds.

## 5. Event catalogs & timetables

AMDA exposes **29 catalogs** and **188 timetables** — curated lists of space physics events. HelioAI exposes them as first-class tools.

**Workflow:**
```
list_catalogs(region="ICME")           → browse available catalogs
get_catalog(id, start, stop)           → inspect events + columns
get_events_timeseries(id, param, ...)  → download param for every event (one speasy call)
```

**Key catalogs:**

| Catalog | Events | Coverage |
|---|---|---|
| Richardson & Cane ICME list | 341 ICMEs | 1996–2022 |
| ICME multi-catalog | 2003 ICMEs, 141 params/event | 1975–2022 |
| THEMIS magnetopause crossings | ~60k events | THEMIS era |
| MMS bow-shock crossings | 2797 events | 2015–present |
| MMS EDR (electron diffusion regions) | 72 events | MMS burst |
| Substorm onsets (Frey) | 2437 events | 2000–2010 |
| Flux Transfer Events Cluster | hundreds | Cluster era |
| MAVEN shock crossings (Mars) | 3837 events | 2014–present |

**Example queries:**
```
"show IMF Bz for all ICMEs in 2005 — use the Richardson & Cane catalog"
"superposed epoch analysis of MMS bow-shock crossings, proton density, 2017"
"how many substorm onsets are there in the Frey catalog?"
```

**Note:** `get_events_timeseries` caps at `max_events=20` by default — raise it for large statistical surveys. The data is saved to the workspace for `run_python` post-processing.

## 6. Derived recipes

HelioAI includes reusable Python scientific recipes accessible via two tools:
- `list_recipes()` — show the full catalogue with name, description, inputs, outputs
- `load_recipe(name)` — retrieve the Python source code
- Then `run_python(code)` to execute it with your data

| Recipe | What it computes |
|---|---|
| `theta_bn` | Shock normal angle θ_Bn (upstream/downstream B) |
| `walen_test` | Walén test — slope ≈ ±1 indicates rotational discontinuity / reconnection |
| `mvab` | Minimum Variance Analysis of B (MVA) — normal to current sheet |
| `rankine_hugoniot` | Rankine-Hugoniot jump conditions across a shock |
| `pressure_balance` | Total pressure balance (thermal + magnetic + dynamic) |
| `pitch_angle_dist` | Pitch angle distribution from particle flux data |

## Output format

Reply in the user's language. Structure your answer as:
1. **Direct answer** to the meta-question
2. **Example query** they could send to HelioAI right now
3. (Optional) pointers to relevant missions or recipes if applicable

Keep it concise — the user wants to know how to proceed, not a lecture.
