# Troubleshooting

Start with the one command that checks everything at once:

```bash
helioai doctor            # the provider key, the index, the sandbox, the .env it read
helioai doctor --online   # also sends one request to the provider
```

Each line is `✓` fine, `!` worth knowing, or `✗` blocking. What follows is what each
message you are likely to meet means, and what to do about it.

## The model provider

`OPENCODE_API_KEY is not set` (or `GROQ_API_KEY`, `GEMINI_API_KEY`, `AZURE_OPENAI_API_KEY`)
:   The provider named by `HELIOAI_LLM_PROVIDER` — `opencode` unless you changed it — has
    no key. Set the key in your shell or in a `.env` file, or switch to the provider you did
    configure. [Installation](installation.md#configure-a-model-provider) lists them.

`HELIOAI_OPENCODE_MODEL is not set`
:   OpenCode has no default model: which ones you can reach depends on your plan. Set the
    exact id from your OpenCode dashboard, e.g. `HELIOAI_OPENCODE_MODEL=deepseek-v4-pro`.

`HelioAI could not reach the ollama server at http://localhost:11434/v1`
:   Ollama is not running, or runs elsewhere. Start it with `ollama serve`, pull the model
    once (`ollama pull qwen2.5:14b-instruct`), or point `HELIOAI_OLLAMA_URL` at the right
    host. For a hosted provider the same message means a network problem or a wrong URL.

`The … server refused the credentials (HTTP 401)`
:   The key is wrong, revoked, or was pasted with a stray space or quote. Copy it again.

`The … server does not know the requested model (HTTP 404)`
:   The model id is misspelt, or your plan does not include it: check
    `HELIOAI_OPENCODE_MODEL`, `HELIOAI_OLLAMA_MODEL` or `AZURE_OPENAI_DEPLOYMENT`.

`… (HTTP 429)`
:   Rate limit or exhausted quota. HelioAI already waits and retries when the provider
    says for how long; if it still fails, wait, or switch provider.

`Unexpected error — …`
:   Something HelioAI does not recognise. Run the same question again with
    `HELIOAI_LOG_LEVEL=DEBUG` to get the traceback, and
    [open an issue](https://github.com/erdoganfurkan/HelioAI/issues) with it.

## The parameter index

`ChromaDB index not found … Run helioai index first`
:   The search index is installed once per release: `helioai index`. On an empty index it
    downloads the prebuilt snapshot from the
    [Hugging Face Hub](https://huggingface.co/datasets/erdoganfurkan/helioai-speasy-index)
    (~125 MB, about a minute). If the snapshot is unavailable or you disabled downloads
    with `HELIOAI_INDEX_REPO=`, it builds the same index locally instead, which takes 7 to
    10 minutes on a recent machine and longer on a modest one. Until then the agent cannot
    find any parameter.

Searches miss parameters you know exist, after an upgrade
:   `helioai index` only adds products it does not have yet, so an improvement to how
    products are described does not reach an existing index. Run `helioai index --download`
    to replace it with the snapshot for the installed release, or `helioai index --rebuild`
    to rebuild it locally.

The index seems to have vanished after setting `HELIOAI_DATA_DIR`
:   Versions before 0.4 kept the index under the default data directory whatever that
    variable said. Run `helioai migrate-storage` once.

## The sandbox

`! sandbox  bubblewrap unavailable or refused by the kernel; run_python falls back to a plain subprocess`
:   The code the agent writes runs without filesystem isolation. On Debian or Ubuntu,
    `sudo apt install bubblewrap`. If it is installed and still refused, the kernel or a
    container profile denies unprivileged user namespaces. On macOS and Windows,
    `run_python` always runs this way. [SECURITY.md](https://github.com/erdoganfurkan/HelioAI/blob/main/SECURITY.md)
    explains what is and is not protected either way.

## The web UI

`address already in use`
:   Port 7890 is taken, often by another `helioai serve --web`. Pick another port:
    `helioai serve --web --port 7891`.

It refuses to start with `--host 0.0.0.0`
:   On purpose: the agent executes code, so a network-reachable server needs users.
    Set `HELIOAI_USERS=token:name,...` and give each person their token, which they
    paste into the sidebar's *Access token* field.

A provider is greyed out as *not configured*
:   The server has no key for it (or, for OpenCode, no model). Keys are read by the
    server, not the browser: set them where `helioai serve --web` runs, then restart it.

*This figure could not be loaded — its file is gone or moved*
:   Session workspaces are deleted after `HELIOAI_WORKSPACE_TTL_S` (seven days by
    default); the conversation stays, its files do not. Export the sessions you want to
    keep as notebooks, or raise the TTL.

## The MCP server

The client says it cannot start `helioai-mcp`
:   MCP clients launch the server from their own working directory, where a bare
    `helioai-mcp` from a virtual environment is not on the path. `helioai mcp-install`
    prints the configuration with the absolute path for your install.

## Reporting a problem

Attach the output of `helioai doctor --json` and, if a question failed, the same run with
`HELIOAI_LOG_LEVEL=DEBUG`. Check that neither contains a key before you post it: the
doctor reports whether a key is set, never its value.
