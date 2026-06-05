# HelioAI

**AI agent for heliophysics and space plasma data analysis.**

Ask questions in natural language — HelioAI finds the right parameter across 70+ missions, downloads it, runs the analysis, and produces reproducible plots and notebooks.

[![CI](https://github.com/erdoganfurkan/HelioAI/actions/workflows/ci.yml/badge.svg)](https://github.com/erdoganfurkan/HelioAI/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/erdoganfurkan/HelioAI/branch/main/graph/badge.svg)](https://codecov.io/gh/erdoganfurkan/HelioAI)
[![PyPI](https://img.shields.io/pypi/v/helioai)](https://pypi.org/project/helioai)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![PyHC](https://img.shields.io/badge/PyHC-compatible-orange)](https://heliopython.org)

---

## What it does

```
You:      "IP shock in WIND data, January 2005 — compute θ_Bn"

HelioAI:  → resolves param IDs for B, Vp, Np across 65k speasy products
          → downloads the time series via speasy (AMDA / CDAWeb / CSA)
          → runs shock detection + coplanarity theorem in a sandboxed Python env
          → returns a plot, the θ_Bn value, and a reproducible .ipynb notebook
```

No API key required for data access. No manual parameter hunting.

---

## Features

- **Hybrid RAG** — semantic (MiniLM) + lexical (BM25) search over 83k parameters, fused by Reciprocal Rank Fusion. Finds both vague descriptions *and* exact codes (`BGSEc`, `FGM`, `igrf_8sec_gse`).
- **Event catalogs & timetables** — access 217 curated AMDA catalogs (ICMEs, bow-shock crossings, reconnection events, substorms, …). Download a parameter across every event in one call — the foundation for superposed epoch analysis and statistical surveys.
- **PlasmaPy tools** — plasma β, gyrofrequency, Debye length, Alfvén speed, inertial length, power spectrum — ready-made for the agent.
- **Sandboxed Python** — the agent writes and runs analysis code safely (subprocess, timeout). All scripts are saved for reproducibility.
- **5 specialised skills** — `parameter_hunter`, `data_analyst`, `plasma_physicist`, `plotting`, `helioai_helper` — loaded as markdown, zero coupling to the agent loop.
- **Derived recipes** — 6 reusable scientific scripts: θ_Bn, Walén test, MVAB, Rankine-Hugoniot jump conditions, pressure balance, pitch angle distribution.
- **Fill value masking** — `clean()` helper in the sandbox automatically masks CDF fill values (`|x| ≥ 1e30`, `±inf`) before any plot or analysis.
- **Export to notebook** — any session exports as a self-contained `.ipynb` with provenance cell, setup shims, and all generated code.
- **Multiple interfaces** — interactive CLI, Jupyter magic, Web UI (FastAPI + SSE + activity dock), MCP server (Claude Desktop / `claude` CLI).
- **User profile** — inject your preferred missions, domain, and plot style once; the agent adapts to you.
- **Heliophysics scope guardrail** — the agent stays on-topic; a dev token unlocks unrestricted mode for development.

---

## Installation

```bash
pip install helioai
```

Or from source with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/erdoganfurkan/HelioAI.git
cd HelioAI
uv sync
```

**First run — build the parameter index** (one-time, ~10 min, 83k products):

```bash
helioai index
```

This downloads the speasy catalogue and indexes it into a local ChromaDB (`data/chroma/`).

---

## Configuration

Copy `.env.example` to `.env` and set at least one LLM provider key:

```ini
# LLM provider (groq | gemini | azure | ollama)
HELIOAI_LLM_PROVIDER=groq

GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here

# Azure OpenAI (if using azure)
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

| Provider | Model | Notes |
|---|---|---|
| `groq` | `llama-3.3-70b-versatile` | Free tier, fast — **recommended to start** |
| `gemini` | `gemini-2.5-flash` | Better reasoning, generous free quota |
| `azure` | configurable | Enterprise / CNES deployments |
| `ollama` | `qwen2.5:14b-instruct` | Fully local, no API key |

---

## Usage

### Interactive CLI

```bash
helioai
```

```
helioai> solar wind density from ACE in January 2005
helioai> compare MMS and Cluster magnetic field during 2017-07-11 reconnection event
helioai> compute plasma beta in the magnetosheath — B=20nT, n=20cm-3, T=200eV
helioai> show me the IMF Bz for all ICMEs in the Richardson & Cane catalog between 2003 and 2005
helioai> superposed epoch analysis of MMS bow-shock crossings — proton density, 2017
```

One-shot mode:

```bash
helioai "IP shock detection in WIND/MFI data, 2005-01-16 to 2005-01-17"
```

### Jupyter

```python
%load_ext helioai.interfaces.jupyter_magic
```

```python
%%helioai
Download Bz from ACE for the 2003 Halloween storm and plot the storm sudden commencement.
```

Figures render inline. Export the session as a notebook:

```python
%helioai_export
```

### Web UI

```bash
helioai serve --web
# → http://localhost:7890
```

Three-panel layout: conversation · artifact viewer (plots, parameter cards) · code panel (generated scripts).

### Claude Desktop / MCP

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "helioai": {
      "command": "helioai-mcp"
    }
  }
}
```

Or run the HTTP MCP server:

```bash
helioai-mcp --http --port 8080
```

### Docker

```bash
docker compose -f docker/docker-compose.yml up -d
# → http://localhost:7890
```

Mount `./data` for persistent index and sessions. Set your LLM keys in `.env`.

---

## Data coverage

| Provider | Missions (examples) | Parameters |
|---|---|---|
| **AMDA** (CDPP) | Cluster, MMS, Solar Orbiter, WIND, ACE, Cassini, Helios, STEREO | ~12k |
| **CDAWeb** (NASA) | MMS, THEMIS, Van Allen Probes, Parker Solar Probe, Ulysses, Voyager | ~68k |
| **CSA** (ESA) | Cluster, Double Star, Solar Orbiter, Mars Express | ~1.9k |

In addition, **217 AMDA event catalogs and timetables** are accessible as first-class tools: ICMEs (Richardson & Cane — 341 events, ICME multi-catalog — 2003 events), bow-shock crossings (MMS 2797, THEMIS ~60k), magnetic reconnection EDR events (72), substorm onsets (2437), flux transfer events, MAVEN shock crossings (3837), and monthly MMS burst-mode timetables (2015–present).

Full parameter catalogue via `list_missions()` or `helioai "what missions are available"`.
Full catalog catalogue via `list_catalogs()` or `helioai "what event catalogs are available"`.

---

## Agent tools

### Data access

| Tool | Description |
|---|---|
| `search_parameters` | Hybrid RAG search — single query or batch `queries=[...]` |
| `get_timeseries` | Download a parameter via speasy (returns cadence, mission, components) |
| `list_missions` | Live catalogue of providers and missions |

### Event catalogs

| Tool | Description |
|---|---|
| `list_catalogs` | Browse 217 AMDA catalogs/timetables — filter by type and region keyword |
| `get_catalog` | Download and inspect a catalog: event count, columns, time-filtered sample |
| `get_events_timeseries` | Download a parameter across **every event** in a catalog (one speasy call) — core tool for superposed epoch analysis |

### Analysis

| Tool | Description |
|---|---|
| `run_python` | Sandboxed Python — speasy + plasmapy + numpy + matplotlib available |
| `plasma_beta` | β = nkT / (B²/2μ₀) |
| `gyrofrequency` | Ion/electron gyrofrequency for a given B |
| `debye_length` | Debye screening length |
| `alfven_speed` | Alfvén speed V_A = B / √(μ₀ρ) |
| `inertial_length` | Ion/electron inertial length |
| `power_spectrum` | Welch PSD on a time series |
| `list_recipes` | Catalogue of scientific Python recipes |
| `load_recipe` | Load a recipe source (θ_Bn, Walén, MVAB, Rankine-Hugoniot, pressure balance, pitch angle dist) |
| `task` | Delegate to a specialised sub-agent |

---

## Sub-agents

| Role | Purpose | Max turns |
|---|---|---|
| `parameter_hunter` | Resolve vague descriptions → speasy parameter IDs | 4 |
| `data_analyst` | Download, analyse, plot, multi-mission, event detection | 8 |
| `plasma_physicist` | PlasmaPy calculations, sanity checks by region | 4 |

---

## Architecture

```
helioai/
├── config.py                   env vars, provider settings, RAG config
├── core/
│   ├── agent_loop.py           async streaming agent (stream_chat)
│   ├── sub_agents.py           specialised sub-agents (parameter_hunter, …)
│   ├── tool_exec.py            shared tool execution logic
│   ├── session.py              SQLite conversation history
│   ├── skills_loader.py        markdown skill loader
│   ├── skills/                 5 × SKILL.md
│   └── llm/                    groq · gemini · azure · ollama + factory
├── tools/
│   ├── rag.py                  hybrid BM25+dense RAG (RRF fusion)
│   ├── speasy_tools.py         search_parameters, get_timeseries, list_missions
│   ├── plasmapy_tools.py       6 plasma physics functions
│   ├── sandbox.py              sandboxed Python execution
│   └── recipes.py              scientific recipe loader
├── interfaces/
│   ├── cli.py                  readline CLI
│   ├── jupyter_magic.py        IPython magic
│   └── web/                    FastAPI + SSE + vanilla JS UI
├── mcp_server.py               MCP stdio + HTTP streamable
├── export.py                   session → reproducible .ipynb
├── indexer.py                  speasy catalogue → ChromaDB
└── docker/                     Dockerfile + docker-compose.yml
```

---

## Development

```bash
uv sync --extra dev
uv run pytest                          # 254 tests, 65% coverage
uv run ruff check helioai/ tests/      # lint
uv run ruff format helioai/ tests/     # format
```

Pre-commit hooks (ruff + trailing-whitespace):

```bash
pre-commit install
```

---

## Roadmap

- [x] CI/CD — GitHub Actions (lint + test matrix Python 3.11/3.12)
- [x] Docker — `helioai serve --web` in a container (`docker/`)
- [ ] JOSS paper
- [ ] PyPI release

---

## License

MIT — see [LICENSE](LICENSE).

---

## Related projects

- [speasy](https://github.com/SciQLop/speasy) — the data access layer powering HelioAI
- [PlasmaPy](https://github.com/PlasmaPy/PlasmaPy) — plasma physics calculations
- [PyHC](https://heliopython.org) — Python in Heliophysics Community
