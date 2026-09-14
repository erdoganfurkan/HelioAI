# Configuration

Every setting is an environment variable, read once when `helioai.config` is imported.
Put them in a `.env` file at the repository root (from a clone) or in the directory you
run `helioai` from (installed from PyPI) — `.env.example` is a commented template — or
export them in the shell. Nothing here is required to `import helioai`; the selected
provider's key is checked when a client is built.

`helioai doctor` reports which `.env` was read and which of these are in effect.

## Model provider

| Variable | Default | Effect |
|---|---|---|
| `HELIOAI_LLM_PROVIDER` | `azure` | `azure`, `groq`, `gemini`, `opencode` or `ollama`. |
| `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` | — | Azure OpenAI credentials; both required for `azure`. |
| `AZURE_OPENAI_DEPLOYMENT` | `models-gpt-53-chat` | Deployment name (Azure routes by deployment, not model). |
| `AZURE_OPENAI_API_VERSION` | `2024-12-01-preview` | Azure API version. |
| `GROQ_API_KEY` | — | Required for `groq`. |
| `GEMINI_API_KEY` | — | Required for `gemini`. |
| `OPENCODE_API_KEY`, `HELIOAI_OPENCODE_MODEL` | — | Required for `opencode`; the model id from your dashboard, no default on purpose. |
| `HELIOAI_OPENCODE_URL` | `https://opencode.ai/zen/go` | Zen gateway endpoint (the Go-plan catalogue; override for plain Zen). |
| `HELIOAI_OLLAMA_URL`, `HELIOAI_OLLAMA_MODEL` | `http://localhost:11434`, `qwen2.5:14b-instruct` | Local inference; no key. |
| `HELIOAI_GROQ_HEADERS`, `HELIOAI_OPENCODE_HEADERS`, `HELIOAI_OLLAMA_HEADERS` | — | Extra HTTP headers, `name=value,name=value`; `{uuid}` becomes a fresh id per client. |
| `HELIOAI_MAX_OUTPUT_TOKENS` | per provider (azure 8192, opencode 16384, others 4096) | One output budget for every provider. Reasoning models spend it on thinking first; too low truncates tool arguments. |

## Agent

| Variable | Default | Effect |
|---|---|---|
| `HELIOAI_MAX_ITERATIONS` | `10` | Tool-calling rounds one question may take before the loop gives up. |
| `HELIOAI_RAG_HYBRID` | `1` | `0` for dense-only parameter search instead of BM25 + dense fused by RRF. |
| `HELIOAI_VISION_ENABLED` | `0` | Review generated figures with a multimodal side-call (text verdict only enters the history). |
| `HELIOAI_VISION_PROVIDER`, `HELIOAI_VISION_MODEL` | `azure`, — | Provider and model for that review. |
| `HELIOAI_DEV_TOKEN` | — | Shared secret that lifts the heliophysics scope guardrail (`--dev`, `X-Helio-Dev-Token`). Empty means nothing unlocks it. |
| `ADS_API_TOKEN` | — | NASA ADS token for `find_papers`. |
| `HELIOAI_MCP_SERVERS` | — | JSON describing remote MCP servers to mount into the tool registry. |

## Storage

| Variable | Default | Effect |
|---|---|---|
| `HELIOAI_DATA_DIR` | `<repo>/data` from a clone, `$XDG_DATA_HOME/helioai` installed | Root of everything HelioAI writes: index, session store, per-user workspaces, catalogues, profile. Set it and run `helioai migrate-storage` once when upgrading an install that already set it. |
| `HELIOAI_SESSION_DB` | `<data_dir>/sessions.db` | The SQLite session store. |
| `HELIOAI_PROFILE` | `<data_dir>/profile.md` | Default user profile injected into the system prompt. |
| `HELIOAI_CATALOGS_DIR` | `<data_dir>/catalogs` | Saved event catalogues (speasy JSON). |
| `HELIOAI_RECIPES_DIR` | the copy inside the package | Your own recipe set instead of the shipped one. |
| `HELIOAI_WORKSPACE_TTL_S` | `604800` (7 days) | Age after which a session's workspace directory is deleted at startup. |

## Serving

| Variable | Default | Effect |
|---|---|---|
| `HELIOAI_USERS` | — | `token:user,token:user` — nominative tokens for the web UI; each user gets its own storage. Empty means one local user and no authentication. |
| `HELIOAI_ALLOW_UNAUTHENTICATED_PUBLIC` | `0` | Lets `serve --web` bind a non-loopback address with no users. Only for a container whose port the host publishes on loopback (`docker-compose.yml` sets it). |
| `HELIOAI_MCP_TOKEN` | — | Bearer token for `helioai-mcp --http`; a non-loopback bind without it is refused. |
| `HELIOAI_LOG_FORMAT` | `console` | `console` or `json` (structlog). |
| `HELIOAI_LOG_LEVEL` | per entry point | Overrides the level the CLI, the web server or the MCP server set. |

Also read: `XDG_DATA_HOME` (the installed-package data root and speasy's own inventory
location), `EDITOR` (`helioai profile`), and the sandbox's environment allow-list, which is
not a setting — see `helioai/tools/sandbox.py`.
