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
    pip install helioai-agent
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

### Check the install

```bash
helioai doctor            # offline: Python, .env found where, provider key, index, sandbox, disk
helioai doctor --online   # plus one request to the provider's model list
helioai doctor --json     # the same report for a bug report or a CI smoke test
```

Every line is a check with a status; `✗` lines block HelioAI and name the fix (`helioai
index`, `helioai migrate-storage`, the missing key). The exit code is 1 when any check
fails, so the command doubles as a health probe.

### Optional extras

| Extra | Brings | For |
|---|---|---|
| `solarmach` | `solarmach` | Parker spiral connectivity figures |
| `dev` | pytest, ruff | contributing |
| `docs` | mkdocs-material, mkdocstrings | building this site |

## Configure a model provider

HelioAI needs one LLM provider. Copy `.env.example` to `.env` and set **one** of:

```ini
HELIOAI_LLM_PROVIDER=groq        # groq | gemini | azure | opencode | ollama
GROQ_API_KEY=your_key_here
```

| Provider | Model | Notes |
|---|---|---|
| `groq` | `llama-3.3-70b-versatile` | free tier, fast — good place to start |
| `gemini` | `gemini-2.5-flash` | stronger reasoning, generous free quota |
| `azure` | your deployment | enterprise deployments |
| `opencode` | set `HELIOAI_OPENCODE_MODEL` | OpenCode's Zen gateway, flat-rate access to hosted reasoning models |
| `ollama` | `qwen2.5:14b-instruct` | fully local, no API key |

Every variable is listed in [Configuration](configuration.md). Any other OpenAI-compatible endpoint works too: a provider is a `base_url` entry in
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
installed from PyPI. Override with `HELIOAI_DATA_DIR`: the index, the session store,
the per-user workspaces, the saved catalogues and the profile all live under it.

!!! note "Upgrading an install that already set `HELIOAI_DATA_DIR`"
    Earlier versions kept the index, the catalogues and the profile under the
    *default* data directory whatever the variable said. Run `helioai migrate-storage`
    once to move them; the `search_parameters` error also tells you when this applies.

Rebuild from scratch with `helioai index --rebuild` — worth doing when speasy ships a
significant catalogue update.

!!! warning "Upgrading HelioAI does not reindex"

    `helioai index` is incremental: it skips every product already in the index, so a
    release that changes *how* products are described leaves your existing index
    untouched and the improvement invisible. After upgrading, run `helioai index
    --rebuild` to pick those up.

## Check it works

```bash
helioai "what missions are available"
```

You should get a list of providers and missions without any data being downloaded. If you
see `AZURE_OPENAI_API_KEY is not set`, `HELIOAI_LLM_PROVIDER` is still on its `azure`
default — set it to the provider you configured.
