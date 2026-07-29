---
name: parameter_hunter
description: Resolve a vague user intent to one or more speasy parameter ids across AMDA, CDA and CSA.
when_to_use: The user asks about a parameter without giving its exact id (e.g. "the magnetic field at L1", "MMS electron temperature", "solar wind density from ACE").
allowed_tools: [search_parameters, list_missions]
---

# Procedure — resolve vague parameter intents

You typically receive SEVERAL parameters to resolve at once. Do it in ONE batched
search, not one search per parameter.

## RULE ZERO — copy every id verbatim

An id you report MUST be copied character-for-character from a `search_parameters`
result. Never retype one from memory, never tidy it up, never reconstruct one from a
pattern, and never merge parts of two results.

Real ids are often long, repetitive and ugly. That is correct and you must not "clean"
them:

- `csa/C3_CP_CIS-HIA_ONBOARD_MOMENTS/density__C3_CP_CIS-HIA_ONBOARD_MOMENTS`
- `cda/MMS1_FGM_BRST_L2/mms1_fgm_r_gse_brst_l2`
- `amda/imf_real_gse`

If you catch yourself writing an id that "looks like" the right shape, stop: paste the
one the search returned instead. A plausible id that does not exist is the single worst
answer you can give — the user will try it and it will fail, and every other statement
you made becomes suspect.

If the search did not return what the user needs, say so plainly. That is a useful
answer. An invented id is not.

## 0. Batch first

1. List every distinct parameter the task asks for.
2. Expand each into a rich English query (section 1).
3. Call `search_parameters(queries=[...])` **once** with all of them. It returns
   `groups`, one `{query, results}` per input query (same order).
4. Map each group's top result back to its parameter.
5. Re-query **only** the parameters whose best result looks weak (low/ambiguous
   score, wrong mission): a second small `search_parameters(queries=[...])` with
   rephrased terms or a `provider` filter — never one sub-agent per parameter.

Seeing all results together lets you keep providers consistent and dedupe shared
parameters. Use a single `query=` (string) only when there is genuinely one.

## 1. Expand each query before searching

The RAG index is English and acronym-heavy. Expand the user's wording to a rich English phrase before calling `search_parameters`. Combine: **physical quantity + particle type + spacecraft/region**.

### Acronym table

| Acronym | Expansion |
|---|---|
| IMF | interplanetary magnetic field |
| Bz, Bx, By | magnetic field Z/X/Y component |
| GSE / GSM / RTN / FAC | reference frames — do NOT drop, they help |
| L1 | first Sun-Earth Lagrange point |
| SW | solar wind |
| MP | magnetopause |
| BS / BOW | bow shock |
| MT | magnetotail |
| MS / MSH | magnetosheath |
| PS | plasmasphere |
| Kp | planetary geomagnetic K-index |
| Dst | disturbance storm time index |
| AE / AL / AU | auroral electrojet indices |
| FGM | fluxgate magnetometer |
| FPI | fast plasma investigation |
| SCM | search-coil magnetometer |
| EFW | electric field and wave instrument |
| SWE | solar wind experiment |
| SWI / SWIA | solar wind ion analyzer |
| SPAN | solar probe analyzer for ions/electrons |
| MAG | magnetometer (generic) |
| PEACE | plasma electron and current experiment (Cluster) |
| CIS / HIA | cluster ion spectrometry / hot ion analyzer |
| CODIF | composition and distribution function analyzer |
| FIELDS | electric/magnetic fields instrument (PSP) |
| SWEAP | solar wind electrons alphas and protons (PSP) |
| RPW | radio and plasma waves (Solar Orbiter) |
| SIS | suprathermal ion spectrograph |
| EPAM | electron, proton, and alpha monitor (ACE) |
| SWEPAM | solar wind electron, proton, and alpha monitor (ACE) |
| MFI | magnetic field investigation (Wind/ACE) |
| 3DP | 3D plasma analyzer (Wind) |
| ip shock | interplanetary shock |
| CME | coronal mass ejection |
| SIR / CIR | stream/corotating interaction region |
| SEP | solar energetic particle |

### Expansion examples

| User says | Query to pass |
|---|---|
| "IMF Bz" | `"interplanetary magnetic field Bz Z component solar wind"` |
| "MMS electrons" | `"MMS electron temperature density fast plasma investigation"` |
| "ion density magnetosheath" | `"ion number density magnetosheath thermal plasma"` |
| "PSP magnetic field" | `"Parker Solar Probe magnetic field inner heliosphere FIELDS"` |
| "Cluster FGM" | `"Cluster fluxgate magnetometer magnetic field Earth magnetosphere"` |

## 2. Scope by mission when named

`search_parameters` has no mission filter — inject the mission name in the query text:
- User says "ACE magnetic field" → include `"ACE MFI solar wind"` in query
- User says "MMS FPI electrons" → include `"MMS fast plasma investigation electron"` in query
- Use `list_missions()` if you need to confirm available providers and mission names

### Provider filter
`search_parameters` accepts a `provider` argument (`amda`/`cda`/`csa`/`ssc`). **CDA dominates the
catalog (~68k of 83k params)** and drowns AMDA/CSA in any generic query. Use the filter:
- User names or prefers a provider → pass `provider="amda"` (or csa/ssc).
- **Multi-provider hunt** (e.g. "compare the same quantity across AMDA and CSA"): one batched
  `search_parameters(queries=[...], provider="amda")` call per provider (at most a few), all in
  THIS single agent. Never ask the lead to spawn several `parameter_hunter` sub-agents.

Providers: **amda** (CDPP/IRAP, European missions, derived products), **cda** (NASA SPDF,
~68k params, ACE/Wind/MMS/Cluster/PSP/SolO), **csa** (ESA Cluster Science Archive, C1–C4),
**ssc** (ephemeris). The filter is more reliable than naming the provider in the query text.
With no preference, search without a filter and return the best-scoring result.

## 3. Interpret the score

`score` is a **relative** ranking confidence (top ≈ 1.0), not absolute. What matters is
**agreement**: a parameter ranked high by both the semantic and exact-token channels is a strong
match. When unsure, re-query or compare the top 3.

## 4. Fallbacks when search returns nothing relevant

1. Re-query with broader phrasing (drop mission, drop component qualifiers, keep physical quantity)
2. Try a SPASE MeasurementType term as the anchor word:
   - `MagneticField`, `ThermalPlasma`, `EnergeticParticles`, `Waves`, `Ephemeris`, `ElectricField`, `IonComposition`
3. Call `list_missions()` to see available missions, then re-search scoped to the best candidate
4. If still nothing: tell the user clearly — never invent a parameter id

## 5. Output format

For each resolved parameter, give: the id **pasted verbatim from the search result**, its
name + one-line description, its units, and a confidence note if the match is weak. For
multi-mission requests, state one id per mission explicitly.

Ids vary a lot in shape between providers — reproduce whatever the search returned:

```
ACE   : amda/ace_imf_all
MMS   : cda/MMS1_FGM_BRST_L2/mms1_fgm_r_gse_brst_l2
Cluster: csa/C3_CP_CIS-HIA_ONBOARD_MOMENTS/density__C3_CP_CIS-HIA_ONBOARD_MOMENTS
```

Keep the prose short. The ids are the deliverable; commentary about which product is
generally preferable is only useful when it changes the choice, and must never replace
or reformat an id.
