# Recipes and provenance

A number without a method is not a result. HelioAI's answer to "how did you get that?" is
recipes: reviewed scientific scripts that each carry the paper they come from, and that
follow the value all the way into the exported notebook.

## What a recipe is

A plain Python module in `helioai/data/recipes/`, with a comment header:

```python
# name: mvab
# description: Minimum variance analysis of the magnetic field (MVAB).
# inputs: B (nT, N×3)
# outputs: n_min, n_int, n_max, eigenvalue ratios
# reference: Sonnerup & Scheible (1998), ISSI SR-001, Ch. 8
```

The agent calls `load_recipe("mvab")`, gets the source *and* the metadata, and runs it in
the sandbox. Because the recipe is a real script rather than a prompt instruction, the
model cannot quietly reimplement the method its own way.

## The shipped recipes

| Recipe | Method |
|---|---|
| `theta_bn` | Shock normal angle via the coplanarity theorem |
| `mvab` | Minimum variance analysis of **B** |
| `walen_test` | Walén test for rotational discontinuities |
| `rankine_hugoniot` | MHD jump conditions across a shock |
| `shock_timing_2sc` | Two-spacecraft shock timing against a normal from `theta_bn`/`mvab` |
| `pressure_balance` | Magnetopause pressure balance |
| `pitch_angle_dist` | Pitch angle distribution |
| `superposed_epoch` | Superposed epoch (Chree) analysis |
| `sep_onset_poisson_cusum` | SEP onset via Poisson-CUSUM |
| `solar_mach` | Parker spiral connectivity (needs the `solarmach` extra) |

## Why "recipe-first" is enforced

Left alone, a capable model will happily recompute θ_Bn from first principles inline. The
arithmetic is often right and the provenance is always gone. So the `data_analyst` skill
carries a hard rule and a task→recipe table: if a recipe exists for the task, load it.

This came directly from user feedback. During a demo, θ_Bn was reported "via recipes" while
an intermediate MVAB had in fact been done inline, with no trace of the method or its
source. The fix was not a better prompt — it was making provenance a tracked artifact.

## Methods you write inline

Not everything has a recipe. For a one-off method, the sandbox provides:

```python
document_method(
    name="Shue et al. magnetopause model",
    reference="Shue et al. (1997), JGR 102, 9497",
    method="r = r0 * (2 / (1 + cos(theta)))**alpha",
)
```

This produces the same provenance card as a recipe, and lands in the same export section.
The point is that *every* derived quantity can be traced, not only the pre-packaged ones.

## Where provenance surfaces

1. **Live** — a 📐 chip in the web UI and Jupyter, naming the recipe or method.
2. **In prose** — the agent adds a line to its answer saying how the value was obtained.
3. **In the export** — a *Methods & data acknowledgements* cell listing every recipe and
   reference used in the session, assembled by scanning the tool calls.

The double signal is deliberate: the artifact chip is precise, and the prose line is what a
reader actually notices.

## Where each number came from

Every `export()` call is recorded in `<session>/data/provenance.json`: the value, its unit,
the script that produced it, and which agent ran it. When the agent finishes an answer, the
numbers it states are checked against that ledger and a `📐 provenance` line reports how many
were traced, contradicted, derived or unsourced — with the flagged ones listed underneath.

Pass the unit when you export, or a compression ratio of 2.53 and a field of 2.53 nT are the
same number to the checker:

```python
export("B_downstream", Bd, units="nT")
```

Three things it deliberately does not do:

- **It annotates, it never blocks.** A number that is absent from the ledger is flagged, not
  removed, and the answer is delivered either way. An approximate numeric match is not a
  good enough judge to overrule a scientist.
- **It certifies provenance, not correctness.** A compression ratio computed over the wrong
  averaging window is recorded, traceable, and wrong. Nothing here will say so.
- **It only sees numbers.** "ACE is sunward of Wind" was published four times against the
  positions that had just been downloaded; a claim with no number in it is out of reach.

## Adding your own

Drop a `.py` file in `helioai/data/recipes/` with the header above. The `# reference:` line
is required — a recipe without a citation defeats the purpose, and the test suite enforces
that every shipped recipe has one.

Point `HELIOAI_RECIPES_DIR` at your own directory to use a private set instead.

!!! note "Recipes are not linted"
    They are executed in the sandbox against an injected namespace (`export`, the loaded
    parameters), so they legitimately reference names that do not exist at import time and
    are excluded from ruff.

## Citing the data too

Recipes cover the method. The data has its own obligations — AMDA, CDAWeb and CSA each ask
to be acknowledged, and catalogs like HELIO4CAST/ICMECAT require citing Möstl et al.
HelioAI's export collects these alongside the method references so the acknowledgements
section is complete rather than half-remembered.
