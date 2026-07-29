# Interfaces

The same agent, the same tools and the same session store are exposed through four
surfaces. Pick whichever fits how you work; sessions are shared across them.

## Interactive CLI

```bash
helioai
```

A readline prompt with history (persisted to `~/.helioai_history`), editing and
tab-friendly recall. Each event is rendered as it happens, so you watch the agent resolve
parameters, call tools and produce figures rather than waiting on a spinner.

```
helioai> solar wind density from ACE in January 2005
helioai> compare MMS and Cluster magnetic field during the 2017-07-11 reconnection event
helioai> superposed epoch analysis of MMS bow-shock crossings — proton density, 2017
```

Figures open in your OS viewer automatically. Ctrl+D or `exit` to leave.

### One-shot

```bash
helioai "IP shock detection in WIND/MFI data, 2005-01-16 to 2005-01-17"
```

### Session management

```bash
helioai history              # list past sessions
helioai --resume             # pick one to continue
helioai --session <id>       # continue a specific one
helioai delete <prefix>      # drop a session and its workspace
helioai export [prefix]      # export a session as a notebook
helioai profile              # edit your profile in $EDITOR
```

## Jupyter

```python
%load_ext helioai.interfaces.jupyter_magic
```

```python
%%helioai
Download Bz from ACE for the 2003 Halloween storm and plot the storm sudden commencement.
```

Figures render inline; parameter cards and catalog previews render as styled HTML. This is
the surface where the [reproducible export](export.md) matters most — you can keep working
on the generated code in the same notebook.

```python
%helioai_export
```

## Web UI

```bash
helioai serve --web
# → http://localhost:7890
```

A three-panel layout: conversation, artifact viewer (figures with a PDF download and
lightbox, parameter cards, catalog previews), and a code panel showing the scripts the
agent generated. An activity dock streams tool calls, sub-agent spawns and figure reviews
live over SSE.

!!! danger "Do not expose this without authentication"
    `run_python` executes model-written code. The open-source build binds to localhost and
    ships no authentication. Read
    [SECURITY.md](https://github.com/erdoganfurkan/HelioAI/blob/main/SECURITY.md) before
    putting it on any network.

## MCP server

HelioAI exposes its tools over the [Model Context Protocol](https://modelcontextprotocol.io),
so any MCP client can drive the heliophysics tooling with its own model.

=== "Claude Desktop"

    ```json
    {
      "mcpServers": {
        "helioai": {
          "command": "helioai-mcp"
        }
      }
    }
    ```

=== "HTTP"

    ```bash
    helioai-mcp --http --port 8080
    ```

### Mounting other MCP servers

The reverse also works: HelioAI is an MCP *client*, so remote tools join its own registry
under an `<alias>_` prefix.

```ini
HELIOAI_MCP_SERVERS={"alphaxiv": {"command": "npx", "args": ["-y", "mcp-remote", "https://api.alphaxiv.org/mcp/v1"]}}
```

Both stdio and streamable-HTTP servers are supported; HTTP servers accept a `headers`
object for authentication. Unreachable servers are logged and skipped rather than fatal.

## Choosing between them

| If you want | Use |
|---|---|
| a quick answer, or scripting | CLI one-shot |
| to iterate on an analysis | Jupyter |
| to show someone the reasoning | Web UI |
| HelioAI's tools with your own agent | MCP server |
