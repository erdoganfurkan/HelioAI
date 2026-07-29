# Installation

## Requirements

- **Python 3.12, 3.13 or 3.14.** HelioAI follows
  [PHEP 3](https://doi.org/10.5281/zenodo.17794207) — see the
  [dependency policy](dev/dependencies.md).
- **Linux is recommended.** The sandbox that runs agent-written code uses
  [bubblewrap](https://github.com/containers/bubblewrap) for real isolation; on macOS and
  Windows it degrades to a plain subprocess. Read
  [SECURITY.md](https://github.com/erdoganfurkan/HelioAI/blob/main/SECURITY.md) before
  running HelioAI anywhere it is reachable from a network.
- **~1 GB of disk** for the parameter index, plus whatever your sessions download.

## Install

=== "From PyPI"

    ```bash
    pip install helioai
    ```

=== "From source"

    ```bash
    git clone https://github.com/erdoganfurkan/HelioAI.git
    cd HelioAI
    uv sync
    ```

    !!! warning "uv purges extras you do not list"
        `uv sync --extra docs` removes the `dev` extra. Always combine the ones you
        want: `uv sync --extra dev --extra solarmach`.

=== "Docker"

    ```bash
    docker compose -f docker/docker-compose.yml up -d
    # → http://localhost:7890
    ```

    The image ships with `bubblewrap`, so the sandbox is fully isolated. Mount `./data`
    to persist the index and sessions.

### Optional extras

| Extra | Brings | For |
|---|---|---|
| `solarmach` | `solarmach` | Parker spiral connectivity figures |
| `dev` | pytest, ruff | contributing |
| `docs` | mkdocs-material, mkdocstrings | building this site |

## Configure a model provider

HelioAI needs one LLM provider. Copy `.env.example` to `.env` and set **one** of:

```ini
HELIOAI_LLM_PROVIDER=groq        # groq | gemini | azure | ollama
GROQ_API_KEY=your_key_here
```

| Provider | Model | Notes |
|---|---|---|
| `groq` | `llama-3.3-70b-versatile` | free tier, fast — good place to start |
| `gemini` | `gemini-2.5-flash` | stronger reasoning, generous free quota |
| `azure` | your deployment | enterprise deployments |
| `ollama` | `qwen2.5:14b-instruct` | fully local, no API key |

Any other OpenAI-compatible endpoint works too: a provider is a `base_url` entry in
`helioai/core/llm/factory.py`, not a class. See [Extending HelioAI](dev/extending.md).

!!! note "Data access needs no key"
    The LLM key is for the agent's reasoning. Downloading data through speasy from AMDA,
    CDAWeb and CSA requires no credentials.

## Build the parameter index

One time, roughly ten minutes, ~83 000 products:

```bash
helioai index
```

This downloads the speasy catalogue and indexes it into a local ChromaDB. It lands in
`<repo>/data/` when you are running from a clone, and in `~/.local/share/helioai/` when
installed from PyPI. Override with `HELIOAI_DATA_DIR`.

Rebuild from scratch with `helioai index --rebuild` — worth doing when speasy ships a
significant catalogue update.

## Check it works

```bash
helioai "what missions are available"
```

You should get a list of providers and missions without any data being downloaded. If you
see `AZURE_OPENAI_API_KEY is not set`, `HELIOAI_LLM_PROVIDER` is still on its `azure`
default — set it to the provider you configured.
