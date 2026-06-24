"""Interactive CLI for HelioAI.

Usage:
    helioai                  # interactive readline session
    helioai "your query"     # one-shot query
    helioai index            # rebuild speasy catalog index
    helioai index --rebuild  # force full reindex
    helioai export [id]      # export a session as a reproducible .ipynb
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


def _render_event(ev: dict) -> None:
    name, data = ev["event"], ev["data"]
    nested = "sub_agent_ctx" in data
    pad = "    " if nested else "  "

    if name == "reply":
        print(f"\n\033[92m{data['text']}\033[0m\n")

    elif name == "tool_call":
        tool = data["name"]
        args = data.get("arguments") or {}
        args_str = ", ".join(f"{k}={repr(v)[:60]}" for k, v in args.items())
        print(f"{pad}\033[90m→ {tool}({args_str})\033[0m")

    elif name == "tool_result":
        summary = data.get("summary", "")
        print(f"{pad}\033[90m← {data['name']}: {summary}\033[0m")

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
                print(f"{pad}\033[90m{data['stdout']}\033[0m")
            for path in paths:
                print(f"{pad}\033[93m  → {path}\033[0m")
                _open_file(path)
        elif kind == "data_preview":
            param = data.get("param_id", "")
            n = data.get("n_points", 0)
            print(f"{pad}\033[93m📈 {param} — {n} points\033[0m")
            if data.get("preview"):
                for line in (data["preview"] or "").split("\n")[:5]:
                    print(f"{pad}\033[90m  {line}\033[0m")

    elif name == "error":
        print(f"\n\033[91m✗ {data['message']}\033[0m\n")

    elif name == "done":
        n = data.get("n_iterations", 0)
        print(f"  \033[90m({n} iteration(s))\033[0m")


async def _run_query(query: str, *, restricted: bool = True) -> None:
    import helioai.tools.setup  # noqa: F401  registers all tools
    from helioai.core.agent_loop import stream_chat
    from helioai.logging_config import setup_logging

    setup_logging("WARNING")

    llm = _build_llm_client()
    async for ev in stream_chat(llm, _USER_ID, _SESSION_ID, query, restricted=restricted):
        _render_event(ev)
        if ev["event"] == "done":
            from helioai.workspace import get_session_dir

            print(f"  \033[90m📂 workspace: {get_session_dir()}\033[0m")


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
    from helioai.config import settings

    p = settings.profile.profile_path
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.touch()
    editor = os.environ.get("EDITOR", "vi")
    subprocess.run([editor, str(p)])


def _interactive(*, restricted: bool = True) -> None:
    import readline  # noqa: F401 — enables history & editing

    mode = "" if restricted else " \033[33m[dev mode]\033[0m"
    print(f"\033[1mHelioAI\033[0m{mode} — type your query, Ctrl+D to exit\n")
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


def main() -> None:
    global _SESSION_ID
    from helioai.config import dev_unlock, settings
    from helioai.workspace import set_user

    set_user(_USER_ID)
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
