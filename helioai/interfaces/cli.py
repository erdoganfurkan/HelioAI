"""Interactive CLI for HelioAI.

Usage:
    helioai                  # interactive readline session
    helioai "your query"     # one-shot query
    helioai index            # rebuild speasy catalog index
    helioai index --rebuild  # force full reindex
    helioai export [id]      # export a session as a reproducible .ipynb
    helioai mcp-install      # print MCP client config pointing at this install
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_SESSION_ID = str(uuid.uuid4())
_USER_ID = "cli"


def _delete_session(prefix: str) -> None:
    import shutil

    from helioai.core.session import store
    from helioai.workspace import _root as _ws_root

    all_ids = store.all_sessions(_USER_ID)
    matches = [s for s in all_ids if s.startswith(prefix)]
    if not matches:
        print(f"No session matching {prefix!r}.")
        return
    sid = matches[0]
    wdir = store.get_workspace_dir(_USER_ID, sid)
    store.reset(_USER_ID, sid)
    if wdir:
        ws_path = _ws_root() / wdir
        if ws_path.exists():
            shutil.rmtree(ws_path, ignore_errors=True)
    print(f"Session {sid[:8]} deleted.")


def _show_history() -> None:
    from helioai.core.session import store

    summaries = store.list_summaries(_USER_ID)
    if not summaries:
        print("No sessions found.")
        return
    print(f"{'Session':<10}  {'Updated':<16}  {'Msgs':>4}  First message")
    print("-" * 72)
    for s in summaries:
        sid = s["session_id"][:8]
        ts = s["updated_at"][:16].replace("T", " ")
        print(f"{sid:<10}  {ts:<16}  {s['n_messages']:>4}  {s['first_message']}")


def _pick_session() -> str | None:
    from helioai.core.session import store

    summaries = store.list_summaries(_USER_ID, limit=10)
    if not summaries:
        print("No previous sessions found.")
        return None
    print("\nRecent sessions:")
    for i, s in enumerate(summaries, 1):
        ts = s["updated_at"][:16].replace("T", " ")
        wdir = s.get("workspace_dir") or ""
        winfo = f"  📂 {wdir}" if wdir else ""
        print(f"  [{i}] {ts}  ({s['n_messages']} msgs)  {s['first_message']}{winfo}")
    try:
        choice = input(f"\nResume [1-{len(summaries)} or session id, Enter to skip]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not choice:
        return None
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(summaries):
            return summaries[idx]["session_id"]
    all_ids = store.all_sessions(_USER_ID)
    matches = [s for s in all_ids if s.startswith(choice)]
    return matches[0] if matches else None


def _build_llm_client():
    from helioai.core.llm.factory import build_llm_client

    return build_llm_client()


def _open_file(path: str) -> None:
    """Open a file in the OS default viewer, cross-platform."""
    import subprocess
    import sys

    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == "win32":
            import os

            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(
                ["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    except Exception:
        pass


def _tilde(path) -> str:
    """Shorten a path under the user's home to `~/...`.

    Purely cosmetic, and the reason is not tidiness: these lines end up in screenshots,
    screen recordings and pasted bug reports, where a full home directory is somebody's
    username on display for no benefit. `~` is just as clickable in a terminal.
    """
    from pathlib import Path as _Path

    text = str(path)
    home = str(_Path.home()).rstrip("/")
    return text.replace(home, "~") if home and home != "/" else text


def _capped_output(text: str, pad: str, max_lines: int = 6) -> str:
    """Keep the head of a tool's stdout and say how much was left out.

    A run that prints one line per sample pushes the answer off the screen; the full
    text is still in the workspace script and the exported notebook.
    """
    lines = str(text).rstrip().splitlines()
    body = "\n".join(f"{pad}{line}" for line in lines[:max_lines])
    hidden = len(lines) - max_lines
    if hidden > 0:
        body += f"\n{pad}\033[90m… +{hidden} more lines\033[0m"
    return body


def _render_event(ev: dict) -> None:
    name, data = ev["event"], ev["data"]
    nested = "sub_agent_ctx" in data
    pad = "    " if nested else "  "

    if name == "reply":
        print(f"\n\033[92m{data['text']}\033[0m\n")

    elif name == "tool_call":
        tool = data["name"]
        detail = data.get("display")
        if detail is None:
            args = data.get("arguments") or {}
            detail = ", ".join(f"{k}={repr(v)[:60]}" for k, v in args.items())
        print(f"{pad}\033[90m→ {tool}{' ' + detail if detail else ''}\033[0m")

    elif name == "tool_result":
        detail = data.get("display") or data.get("summary", "")
        print(f"{pad}\033[90m← {data['name']}: {detail}\033[0m")

    elif name == "sub_agent_start":
        print(f"  \033[94m⚡ spawning {data['role']}...\033[0m")

    elif name == "sub_agent_end":
        role = data.get("role", "")
        summary = (data.get("summary") or "")[:80]
        icon = "✗" if data.get("error") else "✓"
        print(f"  \033[94m{icon} {role}: {data.get('error') or summary}\033[0m")

    elif name == "skill_loaded":
        print(f"{pad}\033[95m📖 skill: {data['name']}\033[0m")

    elif name == "artifact":
        kind = data.get("kind", "")
        if kind == "image":
            paths = data.get("figure_paths", [])
            print(f"{pad}\033[93m📊 {len(paths)} figure(s)\033[0m")
            if data.get("stdout"):
                # Printed in the default colour, not the dim grey used for tool traffic:
                # this is the science the reader came for, and it was previously as faint
                # as the plumbing around it.
                print(_capped_output(data["stdout"], pad))
            for path in paths:
                print(f"{pad}\033[93m  → {_tilde(path)}\033[0m")
                _open_file(path)
        elif kind == "data_preview":
            param = data.get("param_id", "")
            n = data.get("n_points", 0)
            print(f"{pad}\033[93m📈 {param} — {n} points\033[0m")
            if data.get("preview"):
                for line in (data["preview"] or "").split("\n")[:5]:
                    print(f"{pad}\033[90m  {line}\033[0m")

    elif name == "plan":
        print(f"\n{pad}\033[96m📋 {data.get('title', 'Plan')}\033[0m")
        for n, step in enumerate(data.get("steps") or [], 1):
            tool = step.get("tool")
            suffix = f"  \033[90m[{tool}]\033[0m" if tool else ""
            print(f"{pad}  \033[96m{n}.\033[0m {step.get('description', '')}{suffix}")
        print()

    elif name == "figure_review":
        print(f"{pad}\033[95m🔍 figure review: {data.get('text', '')}\033[0m")

    elif name == "provenance":
        counts = (
            f"{data.get('matched', 0)} traced, {data.get('contradicted', 0)} contradicted, "
            f"{data.get('unsourced', 0)} unsourced, {data.get('derived', 0)} derived"
        )
        colour = "91" if data.get("contradicted") or data.get("unsourced") else "90"
        print(f"{pad}\033[{colour}m📐 provenance — {counts}\033[0m")
        for d in data.get("details") or []:
            origin = f" (session computed {d['name']})" if d.get("name") else ""
            print(f"{pad}  \033[{colour}m{d['status']}: {d['text']}{origin}\033[0m")

    elif name == "invalid_ids":
        print(f"\n{pad}\033[91m⚠ ids not in the catalogue — do not use:\033[0m")
        for pid in data.get("ids") or []:
            print(f"{pad}  \033[91m✗ {pid}\033[0m")
        print()

    elif name == "recipe_bypassed":
        print(f"\n{pad}\033[93m⚠ recipe check:\033[0m")
        for r in data.get("recipes") or []:
            reason = (
                "never loaded" if r.get("reason") == "not_loaded" else "loaded but outputs missing"
            )
            print(f"{pad}  \033[93m→ {r.get('recipe')} ({reason})\033[0m")
        print()

    elif name == "error":
        print(f"\n\033[91m✗ {data['message']}\033[0m\n")

    elif name == "done":
        n = data.get("n_iterations", 0)
        print(f"  \033[90m({n} iteration(s))\033[0m")


async def _run_query(query: str, *, restricted: bool = True) -> None:
    import helioai.tools.setup  # noqa: F401  registers all tools
    from helioai.core.agent_loop import stream_chat
    from helioai.logging_config import setup_logging
    from helioai.tools.mcp_client import discover_and_register

    setup_logging("WARNING")
    await discover_and_register()

    llm = _build_llm_client()
    try:
        async for ev in stream_chat(llm, _USER_ID, _SESSION_ID, query, restricted=restricted):
            _render_event(ev)
            if ev["event"] == "done":
                from helioai.workspace import get_session_dir

                print(f"  \033[90m📂 workspace: {_tilde(get_session_dir())}\033[0m")
    finally:
        # The interactive loop runs one asyncio.run per query, so the pool must be
        # released here rather than left for the garbage collector.
        await llm.aclose()


def _run_index(rebuild: bool = False) -> None:
    from helioai.indexer import build_index  # helioai/indexer.py

    build_index(rebuild=rebuild)


def _run_export(prefix: str | None = None) -> None:
    from helioai.core.session import store
    from helioai.export import export_session_notebook

    if prefix:
        matches = [s for s in store.all_sessions(_USER_ID) if s.startswith(prefix)]
        if not matches:
            print(f"No session matching {prefix!r}.")
            return
        session_id = matches[0]
    else:
        sessions = store.all_sessions(_USER_ID)
        if not sessions:
            print("No sessions to export.")
            return
        session_id = sessions[0]
    path = export_session_notebook(_USER_ID, session_id)
    print(f"Exported session {session_id[:8]} → {path}")


def _run_profile() -> None:
    import os
    import subprocess

    from helioai.workspace import user_home

    # Where the agent actually reads it (agent_loop._load_user_profile). Editing
    # settings.profile.profile_path wrote a file nothing has injected since storage
    # was namespaced per user — the command looked like it worked, every time.
    p = user_home(_USER_ID) / "profile.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.touch()
    editor = os.environ.get("EDITOR", "vi")
    subprocess.run([editor, str(p)])


def _interactive(*, restricted: bool = True) -> None:
    import readline  # enables history & editing
    from pathlib import Path

    hist = Path.home() / ".helioai_history"
    try:
        readline.read_history_file(hist)
    except OSError:
        pass

    mode = "" if restricted else " \033[33m[dev mode]\033[0m"
    print(f"\033[1mHelioAI\033[0m{mode} — type your query, Ctrl+D to exit\n")
    try:
        while True:
            try:
                query = input("\033[1m> \033[0m").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                break
            asyncio.run(_run_query(query, restricted=restricted))
    finally:
        try:
            readline.write_history_file(hist)
        except OSError:
            pass


def _run_migrate_storage() -> None:
    """One-shot, idempotent migration of legacy flat storage into per-user homes."""
    import shutil
    from pathlib import Path

    from helioai.config import settings
    from helioai.workspace import DEFAULT_USER, user_home

    moved = 0

    legacy_catalogs = Path(settings.catalogs.catalogs_dir)
    if legacy_catalogs.is_dir():
        dest = user_home(DEFAULT_USER) / "catalogs"
        dest.mkdir(parents=True, exist_ok=True)
        for src in legacy_catalogs.glob("*.json"):
            tgt = dest / src.name
            if not tgt.exists():
                shutil.move(str(src), str(tgt))
                moved += 1

    legacy_profiles = Path(settings.profile.profile_path).parent / "profiles"
    if legacy_profiles.is_dir():
        for src in legacy_profiles.glob("*.md"):
            tgt = user_home(src.stem) / "profile.md"
            if not tgt.exists():
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(tgt))
                moved += 1

    legacy_default_profile = Path(settings.profile.profile_path)
    if legacy_default_profile.is_file():
        tgt = user_home(DEFAULT_USER) / "profile.md"
        if not tgt.exists():
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(legacy_default_profile), str(tgt))
            moved += 1

    print(f"migrate-storage: moved {moved} file(s) into data/users/")


_MCP_CLIENTS = ("claude-code", "claude-code-project", "claude-desktop", "codex")


def _mcp_server_command() -> str:
    """Absolute path to this install's `helioai-mcp` executable.

    An MCP client launches the server from its own working directory, so a bare
    name only works if the install happens to be on the client's PATH — which it
    is not for a venv, and not reliably for pipx. `sysconfig` is asked before
    `which` because a pyenv shim resolves to the shim, not to the script the
    running interpreter would actually use.
    """
    import shutil
    import sys as _sys
    import sysconfig
    from pathlib import Path

    scripts = Path(sysconfig.get_path("scripts"))
    # Windows names the console script helioai-mcp.exe. Without the suffix the probe
    # never matched there, so every Windows install silently fell through to `which` —
    # the very lookup this function exists to avoid, since it can resolve to a different
    # install than the running interpreter's.
    names = ("helioai-mcp.exe", "helioai-mcp") if _sys.platform == "win32" else ("helioai-mcp",)
    for name in names:
        candidate = scripts / name
        if candidate.exists():
            return str(candidate)
    return shutil.which("helioai-mcp") or "helioai-mcp"


def _mcp_config_path(client: str):
    """Where `client` keeps its MCP config, or None when it has no config file.

    Claude Code is the None case on purpose: it ships `claude mcp add`, and
    writing its user config behind its back would be a worse version of a command
    the user already has.
    """
    import sys as _sys
    from pathlib import Path

    if client == "claude-code":
        return None
    if client == "claude-code-project":
        return Path(".mcp.json")
    if client == "codex":
        return Path.home() / ".codex" / "config.toml"
    if client == "claude-desktop":
        if _sys.platform == "darwin":
            return (
                Path.home()
                / "Library"
                / "Application Support"
                / "Claude"
                / ("claude_desktop_config.json")
            )
        if _sys.platform == "win32":
            import os

            base = os.environ.get("APPDATA", str(Path.home()))
            return Path(base) / "Claude" / "claude_desktop_config.json"
        return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"
    raise ValueError(f"unknown MCP client: {client!r}")


def _mcp_payload() -> dict:
    return {"mcpServers": {"helioai": {"command": _mcp_server_command()}}}


def _mcp_snippet(client: str) -> str:
    """The config or command to give `client` so it can reach this install."""
    import json as _json

    command = _mcp_server_command()
    if client == "claude-code":
        return f"claude mcp add helioai -- {command}"
    if client == "codex":
        return f'[mcp_servers.helioai]\ncommand = "{command}"\nargs = []'
    if client in ("claude-desktop", "claude-code-project"):
        return _json.dumps(_mcp_payload(), indent=2)
    raise ValueError(f"unknown MCP client: {client!r}")


def _write_json_config(path) -> None:
    """Merge the helioai entry into a JSON MCP config, creating it if absent.

    Raises:
        ValueError: If the file exists but does not parse. These files hold the
            user's other servers; overwriting one we failed to read would delete
            working configuration to fix nothing.
    """
    import json as _json

    existing: dict = {}
    if path.exists():
        try:
            existing = _json.loads(path.read_text() or "{}")
        except _json.JSONDecodeError as e:
            raise ValueError(f"{path} could not be parsed as JSON ({e}); left untouched") from e
        if not isinstance(existing, dict):
            raise ValueError(f"{path} could not be parsed as an object; left untouched")

    servers = existing.setdefault("mcpServers", {})
    servers["helioai"] = {"command": _mcp_server_command()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps(existing, indent=2) + "\n")


def _run_mcp_install(args: list[str]) -> None:
    """Print (or write) the config that points an MCP client at this install."""
    client = None
    if "--client" in args:
        idx = args.index("--client")
        if idx + 1 < len(args):
            client = args[idx + 1]
    write = "--write" in args

    if client is not None and client not in _MCP_CLIENTS:
        print(f"Unknown client {client!r}. Valid clients: {', '.join(_MCP_CLIENTS)}")
        return

    targets = [client] if client else list(_MCP_CLIENTS)
    for name in targets:
        path = _mcp_config_path(name)
        print(f"\n=== {name} ===")
        if path is not None:
            print(f"config: {path}")
        print(_mcp_snippet(name))

        if not write:
            continue
        if name == "codex":
            print("\n(cannot write TOML safely — paste the block above into that file)")
            continue
        if path is None:
            print("\n(run the command above; Claude Code owns its own config)")
            continue
        try:
            _write_json_config(path)
        except ValueError as e:
            print(f"\nNOT written: {e}")
        else:
            print(f"\nwritten to {path}")


def main() -> None:
    """Entry point for the `helioai` command.

    Routes subcommands (index, export, history, delete, profile, serve, ...) and
    otherwise runs either a one-shot query or the interactive prompt.
    """
    global _SESSION_ID
    from helioai.config import dev_unlock, settings
    from helioai.workspace import cleanup_old_runs, set_user

    set_user(_USER_ID)
    cleanup_old_runs()
    args = sys.argv[1:]

    # --dev: supply the configured dev token to bypass the scope guardrail
    dev_flag = "--dev" in args
    if dev_flag:
        args = [a for a in args if a != "--dev"]
    restricted = not dev_unlock(settings.dev.token if dev_flag else None)

    if "--session" in args:
        idx = args.index("--session")
        if idx + 1 < len(args):
            _SESSION_ID = args[idx + 1]
            args = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]

    if not args:
        _interactive(restricted=restricted)
        return

    if args[0] == "history":
        if len(args) >= 3 and args[1] == "delete":
            _delete_session(args[2])
        else:
            _show_history()
        return

    if args[0] == "index":
        _run_index(rebuild="--rebuild" in args)
        return

    if args[0] == "profile":
        _run_profile()
        return

    if args[0] == "export":
        _run_export(args[1] if len(args) > 1 else None)
        return

    if args[0] == "migrate-storage":
        _run_migrate_storage()
        return

    if args[0] == "mcp-install":
        _run_mcp_install(args[1:])
        return

    if args[0] == "serve":
        if "--web" in args:
            serve_args = args[1:]
            host = "127.0.0.1"
            port = 7890
            if "--host" in serve_args:
                idx = serve_args.index("--host")
                host = serve_args[idx + 1]
            if "--port" in serve_args:
                idx = serve_args.index("--port")
                port = int(serve_args[idx + 1])
            from helioai.interfaces.web.app import serve_web

            serve_web(host=host, port=port)
        else:
            from helioai.mcp_server import main as mcp_main

            sys.argv = [sys.argv[0]] + args[1:]
            mcp_main()
        return

    if "--resume" in args:
        session_id = _pick_session()
        if session_id:
            _SESSION_ID = session_id
        _interactive(restricted=restricted)
        return

    query = " ".join(args)
    asyncio.run(_run_query(query, restricted=restricted))


if __name__ == "__main__":
    main()
