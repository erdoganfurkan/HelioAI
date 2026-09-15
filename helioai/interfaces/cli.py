"""Interactive CLI for HelioAI.

Usage:
    helioai                       # interactive readline session
    helioai "your query"          # one-shot query
    helioai --resume              # pick a past session and continue it
    helioai history               # list sessions
    helioai history delete <id>   # delete a session and its workspace
    helioai index [--rebuild]     # (re)index the speasy catalog
    helioai export [id]           # export a session as a reproducible .ipynb
    helioai profile               # edit the user profile
    helioai mcp-install [--write] # MCP client config pointing at this install
    helioai serve --web           # web UI on :7890 (--host, --port)
    helioai serve                 # MCP server on stdio
    helioai migrate-storage       # move legacy data into the per-user layout
    helioai doctor [--online]     # check this install: index, sandbox, keys, .env (--json)

Options:
    --session <id>                # continue a specific session
    --dev                         # supply the dev token, lifting the scope guard
    -h, --help                    # print this and exit
"""

from __future__ import annotations

import argparse
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
    print(f"{'Session':<10}  {'Updated':<16}  {'Msgs':>4}  {'Tokens':>7}  First message")
    print("-" * 80)
    for s in summaries:
        sid = s["session_id"][:8]
        ts = s["updated_at"][:16].replace("T", " ")
        tokens = _fmt_tokens(s.get("tokens", 0))
        print(f"{sid:<10}  {ts:<16}  {s['n_messages']:>4}  {tokens:>7}  {s['first_message']}")


def _fmt_tokens(n: int) -> str:
    """`12.3k` rather than `12345`: a column, not a bill."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n) if n else "-"


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


# The text of the reply printed so far from `reply_delta` events, so the final `reply`
# prints only what the stream did not — an appended correction — instead of the whole
# answer a second time.
_streamed: list[str] = []


def _render_event(ev: dict) -> None:
    from helioai.core.event_display import describe_findings

    name, data = ev["event"], ev["data"]
    nested = "sub_agent_ctx" in data
    pad = "    " if nested else "  "

    if name == "user":
        pass  # the person who typed the question is looking at it; journaled for replay

    elif name == "reply_delta":
        if not _streamed:
            print("\n\033[92m", end="")
        print(data["text"], end="", flush=True)
        _streamed.append(data["text"])

    elif name == "reply":
        streamed = "".join(_streamed)
        _streamed.clear()
        text = data["text"]
        if streamed and text.startswith(streamed):
            print(f"{text[len(streamed) :]}\033[0m\n")
        elif streamed:
            print(f"\033[0m\n\n\033[92m{text}\033[0m\n")
        else:
            print(f"\n\033[92m{text}\033[0m\n")

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
        from helioai.core.event_display import describe_sub_agent_end

        text, tone = describe_sub_agent_end(data)
        icon, colour = {"ok": ("✓", "94"), "capped": ("◔", "93"), "error": ("✗", "91")}[tone]
        print(f"  \033[{colour}m{icon} {text}\033[0m")
        for line in describe_findings(data.get("findings")):
            print(f"      \033[94m{line}\033[0m")

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

    elif name == "plan":
        print(f"\n{pad}\033[96m📋 {data.get('title', 'Plan')}\033[0m")
        for n, step in enumerate(data.get("steps") or [], 1):
            tool = step.get("tool")
            suffix = f"  \033[90m[{tool}]\033[0m" if tool else ""
            print(f"{pad}  \033[96m{n}.\033[0m {step.get('description', '')}{suffix}")
        print()

    elif name == "plan_report":
        from helioai.core.event_display import describe_plan_report

        capped = any(d.get("capped") for d in data.get("delegations") or [])
        flagged = data.get("missed_tools") or data.get("unplanned_tools") or capped
        colour = "93" if flagged else "90"
        print(f"{pad}\033[{colour}m📋 {describe_plan_report(data)}\033[0m")

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

    elif name == "verdict":
        from helioai.core.event_display import describe_verdict

        summary, lines = describe_verdict(data)
        colour = "91" if data.get("contradicted") else "90"
        print(f"{pad}\033[{colour}m⚖ {summary}\033[0m")
        for line in lines:
            print(f"{pad}  \033[{colour}m{line}\033[0m")

    elif name == "correction":
        ids = ", ".join(data.get("ids") or [])
        print(
            f"{pad}\033[90m↩ correction sent to the model — ids not in the catalogue: {ids}\033[0m"
        )

    elif name == "invalid_ids":
        print(f"\n{pad}\033[91m⚠ ids not in the catalogue — do not use:\033[0m")
        for pid in data.get("ids") or []:
            print(f"{pad}  \033[91m✗ {pid}\033[0m")
        print()

    elif name == "recipe_bypassed":
        print(f"\n{pad}\033[93m⚠ recipe check:\033[0m")
        for r in data.get("recipes") or []:
            reason = {
                "not_loaded": "never loaded",
                "not_called": "loaded but never called",
            }.get(r.get("reason"), "loaded but outputs missing")
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
    """One-shot, idempotent migration of legacy storage layouts.

    Two layouts are folded in. Flat storage (`data/catalogs`, `data/profiles/*.md`,
    `data/profile.md`) moves into per-user homes. And for an install that sets
    `HELIOAI_DATA_DIR`: the index, catalogues and profile that older versions kept
    deriving from the *default* data directory move under the configured one — see
    `_migrate_split_data_dir`.
    """
    import shutil
    from pathlib import Path

    from helioai.config import settings
    from helioai.workspace import DEFAULT_USER, user_home

    moved = _migrate_split_data_dir()

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


def _migrate_split_data_dir() -> int:
    """Move `chroma/`, `catalogs/` and `profile.md` from the default data directory to
    the configured one, when `HELIOAI_DATA_DIR` points elsewhere.

    Earlier versions derived those three from the default directory whatever the variable
    said, so a Docker volume held sessions under `/app/data` and the index under
    `/app/data/helioai`. Each item moves only when the destination does not exist yet,
    which keeps the command re-runnable and never overwrites a rebuilt index.

    Returns:
        How many items moved.
    """
    import shutil

    from helioai.config import _default_data_dir, settings

    legacy_root = _default_data_dir()
    if legacy_root.resolve() == settings.data_dir.resolve():
        return 0
    moved = 0
    for src, dst in (
        (legacy_root / "chroma", settings.rag.chroma_dir),
        (legacy_root / "catalogs", settings.catalogs.catalogs_dir),
        (legacy_root / "profile.md", settings.profile.profile_path),
    ):
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"migrate-storage: {src} -> {dst}")
            moved += 1
    return moved


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


_COMMANDS = (
    "history",
    "index",
    "profile",
    "export",
    "migrate-storage",
    "mcp-install",
    "serve",
    "doctor",
)


def _global_options(args: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Pull `--session`, `--dev` and `--resume` out of argv, wherever they appear.

    `parse_known_args` so that everything else — a subcommand and its own flags, or the
    words of a question — comes back untouched for the router. A flag with its value
    missing is an argparse error (exit 2), not the `IndexError` the hand-rolled
    `args.index()` parsing raised on `helioai --session`.
    """
    p = argparse.ArgumentParser(prog="helioai", add_help=False)
    p.add_argument("--session")
    p.add_argument("--dev", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_known_args(args)


def _run_command(command: str, argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog=f"helioai {command}", add_help=False)
    if command == "history":
        p.add_argument("action", nargs="?", choices=("delete",))
        p.add_argument("session_id", nargs="?")
        ns = p.parse_args(argv)
        if ns.action == "delete":
            if not ns.session_id:
                p.error("history delete needs a session id prefix")
            _delete_session(ns.session_id)
        else:
            _show_history()
    elif command == "index":
        p.add_argument("--rebuild", action="store_true")
        _run_index(rebuild=p.parse_args(argv).rebuild)
    elif command == "profile":
        p.parse_args(argv)
        _run_profile()
    elif command == "export":
        p.add_argument("session_id", nargs="?")
        _run_export(p.parse_args(argv).session_id)
    elif command == "migrate-storage":
        p.parse_args(argv)
        _run_migrate_storage()
    elif command == "mcp-install":
        _run_mcp_install(argv)
    elif command == "doctor":
        from helioai.doctor import run_doctor

        p.add_argument("--online", action="store_true")
        p.add_argument("--json", action="store_true", dest="as_json")
        ns = p.parse_args(argv)
        raise SystemExit(run_doctor(online=ns.online, as_json=ns.as_json))
    elif command == "serve":
        if "--web" in argv:
            p.add_argument("--web", action="store_true")
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=7890)
            ns = p.parse_args(argv)
            from helioai.interfaces.web.app import serve_web

            serve_web(host=ns.host, port=ns.port)
        else:
            # The MCP server parses its own flags (--http, --host, --port, ...).
            from helioai.mcp_server import main as mcp_main

            sys.argv = [sys.argv[0]] + argv
            mcp_main()


def main() -> None:
    """Entry point for the `helioai` command.

    Routes subcommands (index, export, history, profile, serve, doctor, ...) and
    otherwise runs either a one-shot query or the interactive prompt.

    `--help` is answered before anything else runs. The default branch of this
    router treats an unrecognised argument as a question, so until it was
    handled, `helioai --help` created a workspace and billed an LLM call to ask
    the model what `--help` meant — the first thing anyone types after
    `pip install`. Printing `__doc__` keeps the help and the module's own
    documentation as one string. Only a standalone `--help` token counts: a quoted
    question that happens to contain the words is still a question.
    """
    global _SESSION_ID

    args = sys.argv[1:]
    if {"-h", "--help"} & set(args):
        print(__doc__)
        return

    options, rest = _global_options(args)

    if rest and rest[0] in _COMMANDS and rest[0] != "doctor":
        # Storage commands need the user bound; doctor must not touch the workspace.
        from helioai.workspace import set_user

        set_user(_USER_ID)
        _run_command(rest[0], rest[1:])
        return
    if rest and rest[0] == "doctor":
        _run_command("doctor", rest[1:])
        return

    from helioai.config import dev_unlock, settings
    from helioai.workspace import cleanup_old_runs, set_user

    set_user(_USER_ID)
    cleanup_old_runs()
    restricted = not dev_unlock(settings.dev.token if options.dev else None)
    if options.session:
        _SESSION_ID = options.session

    if options.resume:
        session_id = _pick_session()
        if session_id:
            _SESSION_ID = session_id
        _interactive(restricted=restricted)
        return

    if not rest:
        _interactive(restricted=restricted)
        return

    asyncio.run(_run_query(" ".join(rest), restricted=restricted))


if __name__ == "__main__":
    main()
