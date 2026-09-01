# HelioAI

**AI agent for heliophysics and space plasma data analysis.**

Ask a question in natural language. HelioAI finds the right parameter across 70+ missions,
downloads it, runs the analysis in a sandbox, and hands back a plot, a number, and a
notebook you can re-run.

```
You:      "IP shock in WIND data, January 2005 — compute θ_Bn"

HelioAI:  → resolves parameter IDs for B, Vp, Np across 83k speasy products
          → downloads the time series via speasy (AMDA / CDAWeb / CSA)
          → runs shock detection + the coplanarity theorem in a sandboxed Python env
          → returns a plot, the θ_Bn value, and a reproducible .ipynb notebook
```

No API key is needed for data access, and no manual parameter hunting.

![Wind observations of the 17 March 2015 shock and storm: magnetic field magnitude with the
northward/southward Bz turning, proton density, bulk speed and
temperature](assets/shock-analysis.png)

<sub>A HelioAI answer: Wind data for the St Patrick's Day 2015 storm, shock arrival marked,
exported as a runnable notebook.</sub>

## Why it exists

Finding the right parameter is often harder than the analysis itself. A researcher who
knows exactly what they want — *the magnetic field magnitude from Cluster 3 during this
substorm* — still has to know that it lives under `c3_b`, in which dataset, from which
provider, in which coordinate system. HelioAI closes that gap with a hybrid semantic +
lexical search over the whole speasy catalogue, then carries on into the analysis instead
of stopping at the download.

## What makes it different

- **Automation-first.** The unit of work is a scientific question, not a plot panel. Point
  it at an event catalog and it will run the same analysis across every event.
- **Provenance by default.** Derived quantities come from [recipes](guide/recipes.md) that
  each carry a citation, and every session exports a
  [*Methods & data acknowledgements*](guide/export.md) section listing what was used.
- **Reproducible output.** Any session becomes a self-contained `.ipynb` whose cells run in
  a plain Jupyter kernel, with sandbox helpers rewritten to direct `speasy` calls.
- **No separate AI subscription.** Bring any provider — Azure OpenAI, Groq, Gemini, or a
  fully local Ollama model.

## Where it sits in the ecosystem

HelioAI is a layer *on top of* the existing Python heliophysics stack, not a replacement
for any of it:

| It uses | For |
|---|---|
| [speasy](https://github.com/SciQLop/speasy) | data access across AMDA, CDAWeb and CSA |
| [PlasmaPy](https://github.com/PlasmaPy/PlasmaPy) | plasma physics formulary |
| [geopack](https://github.com/tsssss/geopack) | coordinate transforms and boundary models |
| [Astropy](https://www.astropy.org/) · [SunPy](https://sunpy.org/) | units, time, solar context |

Catalogs are written in the standard `speasy` format, so an event set detected here opens
directly in [SciQLop](https://github.com/SciQLop/SciQLop) for visual inspection.

## Next steps

- [Install it](installation.md) — `pip install helioai-agent`, then build the parameter index once.
- [Quickstart](quickstart.md) — your first question, end to end.
- [Interfaces](guide/interfaces.md) — CLI, Jupyter, web UI, or MCP server.

## License and citation

MIT. If HelioAI contributes to published work, please cite it — see
[CITATION.cff](https://github.com/erdoganfurkan/HelioAI/blob/main/CITATION.cff) — and cite
the underlying data providers and any recipe references the export lists for you.
