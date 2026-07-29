# Architecture

```
helioai/
├── config.py               settings singleton, loaded once from .env
├── logging_config.py       structlog setup (console or JSON)
├── datastore.py            npz + manifest per session; the key to reproducible export
├── workspace.py            per-user, per-session directories
├── export.py               session → standalone .ipynb
├── indexer.py              speasy catalogue → ChromaDB
├── mcp_server.py           MCP stdio + streamable HTTP
├── core/
│   ├── agent_loop.py       stream_chat — the lead agent
│   ├── sub_agents.py       stream_subagent — delegation with tool whitelists
│   ├── tool_exec.py        shared execution logic between the two loops
│   ├── session.py          SQLite history per (user_id, session_id)
│   ├── skills_loader.py    markdown skills
│   ├── vision.py           stateless figure review side-call
│   ├── skills/             SKILL.md prompt assets
│   └── llm/
│       ├── base.py         Message, ToolCall, ToolDef, LLMClient, call_with_retry
│       ├── openai_compat.py  one client for every OpenAI-wire provider
│       ├── azure_openai.py   thin subclass (deployment routing, developer role)
│       ├── gemini.py         native google-genai client
│       └── factory.py        build_llm_client + the provider table
├── tools/
│   ├── registry.py         ToolRegistry — JSON dispatch to async functions
│   ├── setup.py            registers all 17 tools at import
│   ├── rag.py              hybrid BM25 + dense retrieval, fused by RRF
│   ├── speasy_tools.py     search, download, data-quality scan
│   ├── catalog_tools.py    AMDA catalogs, event timeseries
│   ├── plasmapy_tools.py   formulary wrappers
│   ├── sandbox.py          bubblewrap-isolated Python execution
│   ├── sandbox_helpers.py  coordinate transforms, boundary models
│   ├── recipes.py          recipe loading
│   ├── literature.py       NASA ADS
│   └── mcp_client.py       mounts remote MCP servers into the registry
├── data/recipes/           shipped scientific recipes (inside the package)
└── interfaces/
    ├── cli.py              readline CLI
    ├── jupyter_magic.py    %%helioai
    └── web/                FastAPI + SSE + vanilla JS
```

## The agent loop

`stream_chat` is an async generator. Each turn: send history plus tool definitions to the
model, dispatch any tool calls through the registry, append results, repeat until the model
answers with text or the iteration cap is hit. Every step yields an event, which is what
lets all four interfaces render progress live from the same source.

`sub_agents.stream_subagent` runs the same shape with a restricted tool set and its own
turn cap. The two loops share `tool_exec.py` rather than duplicating dispatch — they were
duplicated once, and a signature change updated one and not the other.

## Registry

Tools are async functions registered with a JSON Schema:

```python
@registry.register(name="plasma_beta", description="...", parameters={...})
async def plasma_beta(B_nT: float, n_cm3: float, T_eV: float) -> dict: ...
```

`call_tool` always awaits, so every tool must be `async`. Arguments starting with `_` are
rejected from model-supplied input — framework-injected parameters travel through a
separate `trusted` channel so generated code cannot spoof them.

## Storage

Everything is namespaced per user, then per session:

```
<data_dir>/users/<user_id>/workspace/<session>/   figures, scripts, npz, manifest.json
<data_dir>/users/<user_id>/catalogs/              saved catalogs
<data_dir>/chroma/                                the shared parameter index
<data_dir>/sessions.db                            SQLite history
```

`<data_dir>` is `<repo>/data` from a clone and `~/.local/share/helioai` when installed —
see `config._default_data_dir`. The agent's hot path resolves paths from a contextvar; every
other entry point passes `user_id` explicitly, because a contextvar set inside `stream_chat`
is not visible to a CLI subcommand.

## Sandbox

`run_python` spawns a subprocess. On Linux with `bubblewrap` available: read-only root, an
environment allowlist rather than inherited `os.environ` (so API keys are unreadable),
dropped privileges, and resource limits. Elsewhere it degrades to a plain subprocess with
only the timeout and process-group kill — documented in
[SECURITY.md](https://github.com/erdoganfurkan/HelioAI/blob/main/SECURITY.md).

The subprocess always gets `start_new_session=True`. Without it, killing the process group
on timeout kills the HelioAI server itself.

## LLM providers

Groq, Ollama and Azure all speak the OpenAI chat-completions format, so they share
`OpenAICompatClient`; a provider is a `base_url` entry in `factory.OPENAI_COMPAT`. Azure
subclasses it for `AsyncAzureOpenAI` and two dialect quirks. Gemini keeps a native client
because its wire format genuinely differs — notably it has no tool-call ids, so the client
synthesises `name::hex` and parses the name back out.

## Where to look first

| To change | Start at |
|---|---|
| how the agent decides | `core/agent_loop.py` + `core/skills/` |
| what the agent can do | `tools/setup.py` |
| how parameters are found | `tools/rag.py` |
| what a notebook looks like | `export.py` |
| how a provider is added | `core/llm/factory.py` |
