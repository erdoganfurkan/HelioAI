"""Interactive CLI for HelioAI.

Usage:
    helioai                  # interactive readline session
    helioai "your query"     # one-shot query
    helioai index            # rebuild speasy catalog index
    helioai index --rebuild  # force full reindex
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


def _build_llm_client():
    from helioai.config import settings
    provider = settings.llm.provider
    if provider == "groq":
        from helioai.core.llm.groq import GroqClient
        cfg = settings.llm.groq
        return GroqClient(api_key=cfg.api_key, model=cfg.model,
                          max_output_tokens=cfg.max_output_tokens, temperature=cfg.temperature)
    if provider == "gemini":
        from helioai.core.llm.gemini import GeminiClient
        cfg = settings.llm.gemini
        return GeminiClient(api_key=cfg.api_key, model=cfg.model,
                            max_output_tokens=cfg.max_output_tokens, temperature=cfg.temperature)
    raise RuntimeError(f"Unknown LLM provider: {provider!r}. Set HELIOAI_LLM_PROVIDER=groq|gemini")


def _render_event(ev: dict) -> None:
    name, data = ev["event"], ev["data"]

    if name == "reply":
        print(f"\n\033[92m{data['text']}\033[0m\n")

    elif name == "tool_call":
        tool = data["name"]
        args = data.get("arguments") or {}
        args_str = ", ".join(f"{k}={repr(v)[:60]}" for k, v in args.items())
        print(f"  \033[90m→ {tool}({args_str})\033[0m")

    elif name == "tool_result":
        summary = data.get("summary", "")
        print(f"  \033[90m← {data['name']}: {summary}\033[0m")

    elif name == "sub_agent_start":
        print(f"  \033[94m⚡ spawning {data['role']}...\033[0m")

    elif name == "sub_agent_end":
        role = data.get("role", "")
        summary = (data.get("summary") or "")[:80]
        print(f"  \033[94m✓ {role}: {summary}\033[0m")

    elif name == "skill_loaded":
        print(f"  \033[95m📖 skill loaded: {data['name']}\033[0m")

    elif name == "artifact":
        kind = data.get("kind", "")
        if kind == "image":
            n = len(data.get("figures", []))
            print(f"  \033[93m📊 {n} figure(s) generated\033[0m")
            if data.get("stdout"):
                print(f"  \033[90m{data['stdout']}\033[0m")
        elif kind == "data_preview":
            param = data.get("param_id", "")
            n = data.get("n_points", 0)
            print(f"  \033[93m📈 {param} — {n} points\033[0m")
            if data.get("preview"):
                for line in (data["preview"] or "").split("\n")[:5]:
                    print(f"  \033[90m  {line}\033[0m")

    elif name == "error":
        print(f"\n\033[91m✗ {data['message']}\033[0m\n")

    elif name == "done":
        n = data.get("n_iterations", 0)
        print(f"  \033[90m({n} iteration(s))\033[0m")


async def _run_query(query: str) -> None:
    import helioai.tools.setup  # noqa: F401  registers all tools
    from helioai.core.agent_loop import stream_chat
    from helioai.logging_config import setup_logging
    setup_logging("WARNING")

    llm = _build_llm_client()
    async for ev in stream_chat(llm, _USER_ID, _SESSION_ID, query):
        _render_event(ev)


def _run_index(rebuild: bool = False) -> None:
    from helioai.indexer import build_index  # helioai/indexer.py
    build_index(rebuild=rebuild)


def _interactive() -> None:
    import readline  # noqa: F401 — enables history & editing
    print("\033[1mHelioAI\033[0m — type your query, Ctrl+D to exit\n")
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
        asyncio.run(_run_query(query))


def main() -> None:
    args = sys.argv[1:]

    if not args:
        _interactive()
        return

    if args[0] == "index":
        rebuild = "--rebuild" in args
        _run_index(rebuild=rebuild)
        return

    if args[0] == "serve":
        print("MCP server coming in Phase 3 — not yet implemented.")
        return

    # One-shot query
    query = " ".join(args)
    asyncio.run(_run_query(query))


if __name__ == "__main__":
    main()
