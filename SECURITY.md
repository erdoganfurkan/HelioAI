# Security Policy

## Reporting a vulnerability

Please report security issues privately to **furkan.erdogan.pro@gmail.com**
rather than opening a public issue. Include what you did, what happened, and the
HelioAI version. You can expect an acknowledgement within a week.

## Threat model you should know about

HelioAI is an agent: a language model writes Python and HelioAI executes it.
That is the product, not a bug — but it means **the sandbox is a real security
boundary**, and how strong that boundary is depends on your platform.

### Sandbox isolation is Linux-only

`run_python` executes generated code in a subprocess. On Linux, when
[bubblewrap](https://github.com/containers/bubblewrap) (`bwrap`) is installed and
functional, that subprocess runs in a namespace with a read-only root, a
restricted environment, dropped privileges and resource limits.

**On macOS and Windows, and on Linux without `bwrap`, HelioAI falls back to a
plain subprocess.** The timeout and the process-group kill still apply, but there
is no filesystem isolation, no privilege drop, and no namespace separation. Code
the model writes can read and write anything your user account can.

Check what you are getting:

```bash
which bwrap   # empty means you are on the fallback path
```

Install it on Debian/Ubuntu with `apt install bubblewrap`, or run HelioAI in the
provided Docker image, which has it.

### What the sandbox does protect against, on every platform

* API keys are not readable by generated code — the subprocess receives an
  explicit environment allowlist, not `os.environ`.
* Tool arguments beginning with `_` are rejected, so generated code cannot inject
  internal parameters into tool calls.
* Runaway code is killed at the timeout, along with its whole process group.

### Deployment

The web UI (`helioai serve --web`) binds to localhost by default. **Do not expose
it to a network without putting authentication in front of it**, and do not run
it on a host where the fallback sandbox path is in use. A reachable
`run_python` endpoint without isolation is remote code execution.

A loopback bind is not a security boundary on its own: any web page can point a
hostname at `127.0.0.1` and reach a local server (DNS rebinding). `serve --web`
therefore pins the `Host` header when it binds to loopback.

### The MCP server over HTTP

`helioai-mcp` exposes **every registered tool, `run_python` included, with no
authentication**. Over stdio that is the normal MCP contract — the client owns the
process. Over `--http` it is not: anything that can reach the port executes Python
on that host, outside the web UI's scope guardrail.

Keep it on loopback (the default). `helioai-mcp --http --host 0.0.0.0` logs a
warning for this reason; treat that warning as a deployment error unless you have
put authentication in front of the port yourself.

## Supported versions

HelioAI is pre-1.0. Security fixes land on `main` and in the next release; older
releases are not backported.
