"""Jupyter IPython magics for HelioAI.

Load with:
    %load_ext helioai.interfaces.jupyter_magic

Cell magic:
    %%helioai
    solar wind density ACE 2005-01-17

Line magics:
    %helioai_session reset
    %helioai_provider groq|gemini|azure
    %helioai_history
    %helioai_resume <session_id>
    %helioai_export [session_id]
    %helioai_dev on|off   — toggle dev mode (bypasses helio-only scope guardrail)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import uuid

from IPython.core.magic import Magics, cell_magic, line_magic, magics_class
from IPython.display import HTML, Image, Markdown, display

_SESSION_ID = str(uuid.uuid4())
_USER_ID = "jupyter"
_llm = None
_dev_restricted: bool = True  # True = helio-only (default); False = unlocked via dev token


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _get_llm():
    global _llm
    if _llm is not None:
        return _llm
    from helioai.core.llm.factory import build_llm_client

    _llm = build_llm_client()
    return _llm


def _render_jupyter_event(ev: dict) -> None:
    name, data = ev["event"], ev["data"]
    nested = "sub_agent_ctx" in data
    pad = "    " if nested else ""

    if name == "tool_call":
        args = ", ".join(f"{k}={repr(v)[:40]}" for k, v in (data.get("arguments") or {}).items())
        print(f"{pad}  → {data['name']}({args})")

    elif name == "tool_result":
        print(f"{pad}  ← {data['name']}: {data.get('summary', '')}")

    elif name == "sub_agent_start":
        role = data.get("role", "")
        desc = (data.get("description") or "")[:80]
        print(f"\n⚡ spawning {role} — {desc}…")

    elif name == "sub_agent_end":
        role = data.get("role", "")
        summary = (data.get("summary") or "")[:100]
        icon = "✗" if data.get("error") else "✓"
        print(f"{icon} {role}: {data.get('error') or summary}\n")

    elif name == "skill_loaded":
        print(f"{pad}  📖 skill: {data['name']}")

    elif name == "artifact":
        kind = data.get("kind", "")
        if kind == "image":
            for path in data.get("figure_paths", []):
                display(Image(filename=path))
        elif kind == "data_preview":
            param = data.get("param_id", "")
            n = data.get("n_points", 0)
            print(f"  📈 {param} — {n} points")
            if data.get("preview"):
                print(data["preview"])

    elif name == "reply":
        display(Markdown(data.get("text", "")))

    elif name == "done":
        print(f"\n  ({data.get('n_iterations', 0)} iteration(s))")
        from helioai.workspace import get_session_dir

        ws = get_session_dir()
        display(HTML(f"<small style='color:#8b949e'>📂 {ws}</small>"))

    elif name == "error":
        print(f"✗ Error: {data.get('message', '')}")


@magics_class
class HelioAIMagics(Magics):
    @cell_magic
    def helioai(self, line: str, cell: str) -> None:
        import helioai.tools.setup  # noqa: F401
        from helioai.core.agent_loop import stream_chat
        from helioai.logging_config import setup_logging

        setup_logging("WARNING")

        async def _run():
            async for ev in stream_chat(
                _get_llm(), _USER_ID, _SESSION_ID, cell.strip(), restricted=_dev_restricted
            ):
                _render_jupyter_event(ev)

        _run_async(_run())

    @line_magic
    def helioai_session(self, line: str) -> None:
        global _SESSION_ID
        parts = line.strip().split(maxsplit=1)
        cmd = parts[0] if parts else ""
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "reset":
            from helioai.core.session import store

            store.reset(_USER_ID, _SESSION_ID)
            _SESSION_ID = str(uuid.uuid4())
            print(f"Session reset. New id: {_SESSION_ID[:8]}")
        elif cmd == "delete":
            from helioai.core.session import store
            from helioai.workspace import _root

            if not arg:
                print("Usage: %helioai_session delete <session_id_prefix>")
                return
            all_ids = store.all_sessions(_USER_ID)
            matches = [s for s in all_ids if s.startswith(arg)]
            if not matches:
                print(f"No session matching {arg!r}.")
                return
            sid = matches[0]
            wdir = store.get_workspace_dir(_USER_ID, sid)
            store.reset(_USER_ID, sid)
            if wdir:
                import shutil

                ws_path = _root() / wdir
                if ws_path.exists():
                    shutil.rmtree(ws_path, ignore_errors=True)
            if sid == _SESSION_ID:
                _SESSION_ID = str(uuid.uuid4())
                print(f"Current session deleted. New id: {_SESSION_ID[:8]}")
            else:
                print(f"Session {sid[:8]} deleted.")
        else:
            print(f"Unknown command: {cmd!r}. Use 'reset' or 'delete <id>'.")

    @line_magic
    def helioai_provider(self, line: str) -> None:
        global _llm
        provider = line.strip().lower()
        if provider not in ("groq", "gemini", "azure"):
            print(f"Unknown provider {provider!r}. Use: groq | gemini | azure")
            return
        import os

        os.environ["HELIOAI_LLM_PROVIDER"] = provider
        _llm = None
        print(f"Provider switched to {provider!r}.")

    @line_magic
    def helioai_history(self, line: str) -> None:
        from helioai.core.session import store

        summaries = store.list_summaries(_USER_ID)
        if not summaries:
            print("No history found.")
            return
        rows = "".join(
            f"<tr>"
            f"<td><code>{s['session_id'][:8]}</code></td>"
            f"<td>{s['updated_at'][:16].replace('T', ' ')}</td>"
            f"<td style='text-align:center'>{s['n_messages']}</td>"
            f"<td>{s['first_message']}</td>"
            f"</tr>"
            for s in summaries
        )
        display(
            HTML(
                "<table><thead><tr>"
                "<th>Session</th><th>Updated</th><th>Msgs</th><th>First message</th>"
                "</tr></thead><tbody>" + rows + "</tbody></table>"
            )
        )

    @line_magic
    def helioai_profile(self, line: str) -> None:
        from helioai.config import settings

        parts = line.strip().split(maxsplit=1)
        cmd = parts[0] if parts else ""
        arg = parts[1].strip().strip("\"'") if len(parts) > 1 else ""
        p = settings.profile.profile_path

        if cmd == "show":
            content = p.read_text(encoding="utf-8").strip() if p.exists() else ""
            display(Markdown(content if content else "_(profil vide)_"))
        elif cmd == "set":
            if not arg:
                print('Usage: %helioai_profile set "your preferences here"')
                return
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(("\n" if p.stat().st_size > 0 else "") + arg + "\n")
            print(f"Profile updated ({p}).")
        else:
            print('Usage: %helioai_profile show | set "<text>"')

    @line_magic
    def helioai_export(self, line: str) -> None:
        from helioai.core.session import store
        from helioai.export import export_session_notebook

        prefix = line.strip()
        session_id = _SESSION_ID
        if prefix:
            matches = [s for s in store.all_sessions(_USER_ID) if s.startswith(prefix)]
            if not matches:
                print(f"No session matching {prefix!r}.")
                return
            session_id = matches[0]
        path = export_session_notebook(_USER_ID, session_id)
        from IPython.display import FileLink

        display(FileLink(str(path), result_html_prefix="📓 Exported notebook: "))

    @line_magic
    def helioai_resume(self, line: str) -> None:
        global _SESSION_ID
        prefix = line.strip()
        if not prefix:
            print("Usage: %helioai_resume <session_id>")
            return
        from helioai.core.session import store

        all_ids = store.all_sessions(_USER_ID)
        matches = [s for s in all_ids if s.startswith(prefix)]
        if not matches:
            print(f"No session found matching {prefix!r}.")
            return
        _SESSION_ID = matches[0]
        msgs = store.get_or_create(_USER_ID, _SESSION_ID)
        print(f"Resumed session {_SESSION_ID[:8]} ({len(msgs)} messages).")

    @line_magic
    def helioai_dev(self, line: str) -> None:
        global _dev_restricted
        from helioai.config import dev_unlock, settings

        cmd = line.strip().lower()
        if cmd == "on":
            if not dev_unlock(settings.dev.token):
                print("Dev token not configured or incorrect. Set HELIOAI_DEV_TOKEN in .env.")
                return
            _dev_restricted = False
            print("Dev mode ON — scope guardrail disabled.")
        elif cmd == "off":
            _dev_restricted = True
            print("Dev mode OFF — helio-only scope guardrail active.")
        else:
            status = "OFF (restricted)" if _dev_restricted else "ON (unrestricted)"
            print(f"Dev mode: {status}. Usage: %helioai_dev on | off")


def load_ipython_extension(ipython) -> None:
    ipython.register_magics(HelioAIMagics)
