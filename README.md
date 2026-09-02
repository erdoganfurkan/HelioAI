# HelioAI

**AI agent for heliophysics and space plasma data analysis.**

Ask questions in natural language — HelioAI finds the right parameter across 70+ missions, 
downloads it, runs the analysis, and produces reproducible plots and notebooks.

[![CI](https://github.com/erdoganfurkan/HelioAI/actions/workflows/ci.yml/badge.svg)](https://github.com/erdoganfurkan/HelioAI/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/erdoganfurkan/HelioAI/branch/main/graph/badge.svg)](https://codecov.io/gh/erdoganfurkan/HelioAI)
[![PyPI](https://img.shields.io/pypi/v/helioai-agent)](https://pypi.org/project/helioai-agent/)
[![PyHC](https://img.shields.io/badge/PyHC-listed-5a4fcf)](https://heliopython.org/projects/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](pyproject.toml)

📖 **[Documentation](https://erdoganfurkan.github.io/HelioAI/)** · listed in the
[PyHC Project List](https://heliopython.org/projects/)

---

## What it does

```
You:      "IP shock in WIND data, January 2005 — compute θ_Bn"

HelioAI:  → resolves param IDs for B, Vp, Np across 83k speasy products
          → downloads the time series via speasy (AMDA / CDAWeb / CSA)
          → runs shock detection + coplanarity theorem in a sandboxed Python env
          → returns a plot, the θ_Bn value, and a reproducible .ipynb notebook
```

Data access needs no API key. Parameter hunting is the agent's job, not yours.

![The HelioAI web UI running a real session: the question, the resolved parameter card for cda/WI_H0_MFI/B3F1, the plotted magnetic field magnitude with the shock arrival marked, the activity log of every tool call, and the generated Python in the code panel](https://raw.githubusercontent.com/erdoganfurkan/HelioAI/main/docs/assets/web-demo.gif)

<sub>A real session, unedited — only the waiting between turns is compressed. The activity
log lists every tool call and ends with a provenance verdict; the code panel holds the exact
Python that produced the figure, which is also what the session exports as a runnable
notebook.</sub>

---

## Why it's different

- **It finds the parameter.** Hybrid RAG — semantic (MiniLM) + lexical (BM25), fused by
  Reciprocal Rank Fusion — over **83 000 speasy products**. Handles both vague descriptions
  and exact codes (`BGSEc`, `FGM`, `igrf_8sec_gse`).
- **It works on events, not just intervals.** 217 curated AMDA catalogs (ICMEs, bow-shock
  crossings, substorms, reconnection events) are first-class tools — download a parameter
  across *every* event in one call.
- **The result is reproducible.** Every session exports to a self-contained `.ipynb` that
  re-runs in a plain Jupyter kernel, with a *Methods & data acknowledgements* cell listing
  the recipes and references used.
- **It shows its work.** A provenance ledger checks the numbers in the answer against what
  was actually computed, and 11 vetted recipes (θ_Bn, Walén, MVAB, Rankine-Hugoniot, …)
  each carry a citation.
- **It runs inside the agent you already use.** HelioAI is also an MCP server, so Claude
  Code, Claude Desktop or Codex can call its tools with no LLM key of its own.

Full feature list in the [documentation](https://erdoganfurkan.github.io/HelioAI/).

---

## Install

```bash
pip install helioai-agent
helioai index          # one-time, ~10 min — indexes 83k products into a local ChromaDB
```

Then set one LLM provider key (`opencode`, `groq`, `gemini`, `azure` or `ollama` — the
OpenCode Zen gateway serves reasoning models behind a single key, and the model name is
required since there is no sensible default):

```bash
export HELIOAI_LLM_PROVIDER=opencode
export OPENCODE_API_KEY=...
export HELIOAI_OPENCODE_MODEL=deepseek-v4-pro
```

→ [Full installation and configuration guide](https://erdoganfurkan.github.io/HelioAI/installation/)

---

## Use it

```bash
helioai                       # interactive CLI
helioai "θ_Bn for the 2005-01-16 WIND shock"      # one-shot
helioai serve --web           # web UI on http://localhost:7890
helioai mcp-install           # wire it into Claude Code, Claude Desktop or Codex
```

In Jupyter:

```python
%load_ext helioai.interfaces.jupyter_magic
%%helioai
Download Bz from ACE for the 2003 Halloween storm and plot the sudden commencement.
```

→ [All four interfaces, in detail](https://erdoganfurkan.github.io/HelioAI/guide/interfaces/)

---

## Data coverage

| Provider | Missions (examples) | Parameters |
|---|---|---|
| **AMDA** (CDPP) | Cluster, MMS, Solar Orbiter, WIND, ACE, Cassini, Helios, STEREO | ~12k |
| **CDAWeb** (NASA) | MMS, THEMIS, Van Allen Probes, Parker Solar Probe, Ulysses, Voyager | ~68k |
| **CSA** (ESA) | Cluster, Double Star, Solar Orbiter, Mars Express | ~1.9k |

Plus 217 AMDA event catalogs and timetables. Ask `helioai "what missions are available"` or
`helioai "what event catalogs are available"` for the live list.

---

## Documentation

| | |
|---|---|
| [Quickstart](https://erdoganfurkan.github.io/HelioAI/quickstart/) | First session, end to end |
| [Interfaces](https://erdoganfurkan.github.io/HelioAI/guide/interfaces/) | CLI · Jupyter · web UI · MCP |
| [Agent tools](https://erdoganfurkan.github.io/HelioAI/guide/tools/) | The 17 tools and 4 sub-agents |
| [Recipes and provenance](https://erdoganfurkan.github.io/HelioAI/guide/recipes/) | The 11 vetted scientific scripts |
| [Reproducible export](https://erdoganfurkan.github.io/HelioAI/guide/export/) | How a session becomes a notebook |
| [Architecture](https://erdoganfurkan.github.io/HelioAI/dev/architecture/) | For contributors |

---

## Contributing

Issues and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and
[SECURITY.md](SECURITY.md) (the sandbox model matters if you touch `run_python`).

```bash
uv sync --extra dev
.venv/bin/python -m pytest      # 865 tests, 80% coverage, no exclusions
.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .
```

---

## License

MIT — see [LICENSE](LICENSE).

## Related projects

- [speasy](https://github.com/SciQLop/speasy) — the data access layer powering HelioAI
- [PlasmaPy](https://github.com/PlasmaPy/PlasmaPy) — plasma physics calculations
- [PyHC](https://heliopython.org) — Python in Heliophysics Community
