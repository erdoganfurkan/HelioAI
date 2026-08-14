# Quickstart

This walks through one real question end to end. It assumes you have
[installed HelioAI](installation.md) and built the index.

## Ask a question

```bash
helioai "solar wind proton density from ACE, 16-17 January 2005"
```

HelioAI works in visible steps:

```
→ search_parameters(queries=['solar wind proton density ACE'])
← search_parameters: 5 matches, best cda/AC_H0_SWE/Np (score 0.89)
→ get_timeseries(param_id='cda/AC_H0_SWE/Np', start='2005-01-16', stop='2005-01-18')
← get_timeseries: 1440 points, 64 s cadence, units cm^-3
📊 1 figure(s)
  → /home/you/.local/share/helioai/workspace/run_3/figure_1.png
```

Three things are worth noticing.

**It resolved the parameter itself.** You wrote "proton density from ACE"; it found
`cda/AC_H0_SWE/Np`. The search is hybrid — semantic embeddings for descriptions, BM25 for
exact tokens — so `BGSEc` and "magnetic field in GSE coordinates" both work.

**It reported the data quality.** Every download is scanned for fill values, gaps and 5σ
outliers. If something matters, the agent tells you *before* the analysis, because a
plasma beta computed across a data gap is a wrong number that looks right.

**It saved the script.** Everything the agent ran is on disk in the session workspace.
Nothing is a black box.

## Go further in the same session

```
helioai> now compute the plasma beta over that interval
helioai> compare with WIND over the same window
helioai> export this session
```

The session keeps its history and its downloaded data, so follow-ups do not re-download.

## Run an analysis over a whole event catalog

This is where HelioAI stops being a convenience and starts being a different tool:

```bash
helioai "superposed epoch analysis of IMF Bz across the Richardson & Cane ICME catalog, 2003-2005"
```

It resolves the catalog, filters it to the requested window, downloads the parameter across
**every** event in one call, and runs the superposed epoch
[recipe](guide/recipes.md) — which carries its own citation (Chree 1913) into the export.

217 AMDA catalogs and timetables are available this way: ICMEs, bow-shock crossings,
reconnection events, substorm onsets, MAVEN shock crossings, and monthly MMS burst-mode
timetables. Ask `helioai "what event catalogs are available"` to browse them.

## Get a reproducible notebook

```bash
helioai export
```

You get a self-contained `.ipynb`. The sandbox helper `load_data()` is rewritten into
direct `spz.get_data(...)` calls, agent-only helpers are stripped, and a final
*Methods & data acknowledgements* cell lists every recipe and reference used. It runs in a
plain Jupyter kernel with no HelioAI installed — that is the point.

See [Reproducible export](guide/export.md) for what is rewritten and why.

## Runnable notebooks

Two example notebooks live in
[`examples/`](https://github.com/erdoganfurkan/HelioAI/tree/main/examples):
a guided Jupyter tour, and the 2015 St. Patrick's Day storm worked end to end. They ship without
outputs on purpose — run them and produce your own.

## Where to next

- [Interfaces](guide/interfaces.md) — the same agent in Jupyter, a web UI, or as an MCP
  server inside Claude Desktop.
- [Agent tools](guide/tools.md) — what the agent can actually do.
- [Recipes and provenance](guide/recipes.md) — how derived quantities stay citable.
