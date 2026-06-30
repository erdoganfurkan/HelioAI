#!/usr/bin/env python
"""Token bench for HelioAI — measure context cost before/after the trimming work.

Two modes:
  --dry  (default, ZERO LLM cost): measure the static per-call prefix
         (system prompt + tool schemas) re-sent on every LLM call. This is
         exactly what the prompt / schema / skill trimming shrinks, so the
         dry delta tracks phases 1-3 without spending a token.
  --live (spends REAL Azure tokens): run each witness prompt end-to-end and
         report real prompt / completion / cached tokens per LLM call.

Usage:
  uv run python scripts/token_bench.py                 # dry breakdown
  uv run python scripts/token_bench.py --live          # real run, all prompts ×1
  uv run python scripts/token_bench.py --live -n 3     # average over 3 runs
  uv run python scripts/token_bench.py --live --only 1 # just the reference prompt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Witness prompts cover the main flows: lead-only, data_analyst delegation, multi-mission,
# catalog→recipe, event detection. Drop a `witness_prompts.txt` next to this script to override.
DEFAULT_PROMPTS = [
    "Solar wind speed and density from ACE, 2005-01-17 to 2005-01-18, with a plot",
    "Plasma beta for B=5 nT, n=5 cm^-3, T=1e5 K",
    "Compare IMF Bz from ACE and Wind on 2005-01-17",
    "Superposed epoch analysis of IMF Bz across the interplanetary shocks of 2005 from an AMDA shock catalog",
    "Detect interplanetary shocks in ACE solar wind data between 2005-01-15 and 2005-01-20",
]
WITNESS = Path(__file__).resolve().parent / "witness_prompts.txt"

try:
    import tiktoken

    _enc = tiktoken.get_encoding("o200k_base")

    def count_tokens(s: str) -> int:
        return len(_enc.encode(s))

    TOK_NOTE = "tiktoken o200k_base"
except Exception:  # tiktoken not installed → consistent proxy, good enough for deltas

    def count_tokens(s: str) -> int:
        return (len(s) + 3) // 4

    TOK_NOTE = "heuristic len/4 (pip install tiktoken for exact counts)"


def load_prompts() -> list[str]:
    if not WITNESS.exists():
        return DEFAULT_PROMPTS
    out = []
    for line in WITNESS.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _tool_json(t) -> str:
    return json.dumps(
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters or {"type": "object", "properties": {}},
            },
        },
        ensure_ascii=False,
    )


def dry() -> None:
    import helioai.tools.setup  # noqa: F401  registers all tools
    from helioai.core.agent_loop import _INTERNAL_TOOLS, build_lead_system_prompt
    from helioai.core.skills_loader import load_skill as load_skill_body
    from helioai.core.sub_agents import AGENT_ROLES, SUB_SYSTEM_PROMPT_BASE, task_tool_def
    from helioai.tools.registry import registry

    sys_prompt = build_lead_system_prompt(restricted=True)
    tools = registry.list_tool_defs() + _INTERNAL_TOOLS + [task_tool_def()]

    sys_t = count_tokens(sys_prompt)
    rows = [(count_tokens(_tool_json(t)), t.name) for t in tools]
    tool_total = sum(n for n, _ in rows)
    total = sys_t + tool_total

    print(f"# Token bench — DRY (static prefix re-sent on every LLM call)   [{TOK_NOTE}]\n")
    print("## Lead agent, per call")
    print(f"  {'system prompt':40s} {sys_t:>7d} tok  ({len(sys_prompt)} chars)")
    for n, name in sorted(rows, reverse=True):
        print(f"  {'tool: ' + name:40s} {n:>7d} tok")
    print(f"  {'-' * 40} {'-' * 7}")
    print(f"  {f'tools subtotal ({len(tools)})':40s} {tool_total:>7d} tok")
    print(f"  {'LEAD STATIC / CALL':40s} {total:>7d} tok")

    print("\n## Sub-agent static / call (base + role addon + auto-loaded skill)")
    base_t = count_tokens(SUB_SYSTEM_PROMPT_BASE)
    for role, cfg in AGENT_ROLES.items():
        addon_t = count_tokens(cfg.system_addon)
        skill_t = sum(count_tokens(load_skill_body(s)) for s in cfg.auto_load_skills)
        print(
            f"  {role:18s} base {base_t:>5d} + addon {addon_t:>5d} + skill {skill_t:>6d}"
            f" = {base_t + addon_t + skill_t:>6d} tok"
        )
    print("\n(Lead static is paid on every lead turn; sub-agent static on every sub turn.)")

    # ponytail: the only non-trivial bit is the sum — assert it holds.
    assert total == sys_t + sum(n for n, _ in rows)


async def _drain(agen) -> None:
    async for _ in agen:
        pass


async def live_async(n_repeat: int, only: int | None) -> None:
    import helioai.tools.setup  # noqa: F401  registers all tools
    from helioai.core import agent_loop
    from helioai.core.llm.factory import build_llm_client

    client = build_llm_client()
    calls: list[tuple[int, int, int]] = []  # (prompt, completion, cached) per LLM call
    orig_chat = client.chat

    async def wrapped(*a, **k):
        resp = await orig_chat(*a, **k)
        calls.append(
            (
                resp.prompt_tokens or 0,
                resp.completion_tokens or 0,
                getattr(resp, "cached_tokens", 0) or 0,
            )
        )
        return resp

    client.chat = wrapped  # type: ignore[method-assign]

    prompts = load_prompts()
    print("# Token bench — LIVE (real Azure prompt+completion tokens)\n")
    grand = 0
    for idx, prompt in enumerate(prompts, 1):
        if only and idx != only:
            continue
        runs = []
        for _ in range(n_repeat):
            calls.clear()
            uid, sid = f"bench_{uuid.uuid4().hex[:8]}", uuid.uuid4().hex[:8]
            await _drain(agent_loop.stream_chat(client, uid, sid, prompt))
            runs.append(
                (
                    sum(c[0] for c in calls),
                    sum(c[1] for c in calls),
                    sum(c[2] for c in calls),
                    len(calls),
                )
            )
        k = len(runs)
        p = sum(r[0] for r in runs) // k
        co = sum(r[1] for r in runs) // k
        ca = sum(r[2] for r in runs) // k
        nc = sum(r[3] for r in runs) / k
        grand += p + co
        print(f"[{idx}] {prompt[:62]}")
        print(f"    calls~{nc:.1f}  prompt {p}  completion {co}  cached {ca}  TOTAL {p + co}")
    print(f"\nGRAND TOTAL (prompt+completion summed across prompts): {grand}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--live", action="store_true", help="run prompts for real (spends Azure tokens)"
    )
    ap.add_argument("-n", "--repeat", type=int, default=1, help="live runs per prompt to average")
    ap.add_argument(
        "--only", type=int, default=None, help="live: run only witness prompt N (1-based)"
    )
    args = ap.parse_args()
    if args.live:
        asyncio.run(live_async(args.repeat, args.only))
    else:
        dry()


if __name__ == "__main__":
    main()
