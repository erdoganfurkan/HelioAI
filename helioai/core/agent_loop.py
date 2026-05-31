"""The agent decision loop.

Given a user message and a session id, run the LLM in a tool-using loop:
the LLM may emit tool calls, execute them via the ToolRegistry, feed the
results back, and iterate until the LLM produces a final text reply (or
we hit the safety cap).

Two consumption modes share the same generator core (stream_chat):
  - chat()        → collects all events, returns a single ChatResult
  - stream_chat() → async generator, yields one event dict per step

Event shapes:
  tool_call       {turn, name, arguments}
  tool_result     {turn, name, summary}
  sub_agent_start {task_id, role, description}
  sub_agent_end   {task_id, role, summary, n_iterations, error}
  skill_loaded    {name}
  reply           {text}
  done            {n_iterations}
  error           {message}
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from helioai.config import settings
from helioai.core.llm.base import LLMClient, Message, ToolDef
from helioai.core.session import store
from helioai.core.skills_loader import SkillError, load_index as load_skills_index
from helioai.core.skills_loader import load_skill as load_skill_body, list_skill_names
from helioai.core.sub_agents import AGENT_ROLES, TASK_TOOL_NAME, run_subagent, task_tool_def
from helioai.logging_config import get_logger
from helioai.tools.registry import registry

log = get_logger(__name__)


SYSTEM_PROMPT = """You are HelioAI, an expert scientific assistant for heliophysics and space plasma research.

You have access to tools that let you explore and analyze data from 70+ space missions (MMS, Solar Orbiter, Cluster, WIND, ACE, Cassini, MEX, Parker Solar Probe, HelioSwarm…) via the speasy library, run Python code for scientific analysis, and search through 65 000+ parameters.

Discovery tools:
1. `search_parameters` — semantic search over 65k+ speasy parameters. Use English queries. If the user is vague (e.g. "solar wind density"), rewrite as matchable terms (e.g. "solar wind ion number density near Earth ACE WIND").
2. `list_missions` — list available data providers. Use when the user asks what data is available.

Data tool:
3. `get_timeseries(param_id, start, stop)` — download a time series from any speasy provider. Always resolve the parameter id via `search_parameters` first. Use ISO 8601 times.

Analysis tool:
4. `run_python(code)` — execute Python in an isolated sandbox. Pre-imported: speasy (as `spz`), numpy (as `np`), scipy, matplotlib (Agg, plt.show() captures PNG), plasmapy (as `pf`), astropy units (as `u`). Use for: FFTs, power spectra, plasma parameter calculations, event detection, custom visualizations.

Skills:
5. `list_skills()` — index of available procedural skills.
6. `load_skill(name)` — load a skill's procedure. Call before acting on matching requests.

Sub-agents:
7. `task(description, agent_role)` — delegate to a specialist. Context is EMPTY in the sub — pre-resolve all facts (parameter ids, ISO times, missions) in `description`.

Delegate (do NOT call underlying tools yourself) when:
- User needs 2+ different parameters → spawn one `parameter_hunter` per parameter
- User wants to compare multiple missions → spawn `cross_mission`
- User asks for complex analysis (spectra, plasma params, event detection) → spawn `data_analyst`

Workflow rules:
- Always use ISO 8601 times: `2024-01-01T00:00:00`
- Always resolve parameter ids via `search_parameters` before `get_timeseries`
- When `run_python` returns figures (base64 PNG), report to the user that the plot was generated
- Reply in the user's language
- Cite the parameter ids you used
"""


_INTERNAL_TOOLS: list[ToolDef] = [
    ToolDef(
        name="list_skills",
        description=(
            "List the procedural skills available. Returns a markdown table with "
            "name and when_to_use. Call load_skill(name) for the full procedure."
        ),
        parameters={"type": "object", "properties": {}},
    ),
    ToolDef(
        name="load_skill",
        description=(
            "Load the full body of one skill (markdown procedure). Call BEFORE "
            "acting on a request that matches a skill's when_to_use trigger."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name from list_skills."}
            },
            "required": ["name"],
        },
    ),
]

_INTERNAL_TOOL_NAMES: frozenset[str] = frozenset(t.name for t in _INTERNAL_TOOLS)


def _dispatch_internal_tool(name: str, arguments: dict) -> str:
    args = arguments or {}
    try:
        if name == "list_skills":
            return json.dumps({"index": load_skills_index(), "names": list_skill_names()})
        if name == "load_skill":
            skill = (args.get("name") or "").strip()
            return json.dumps({"name": skill, "body": load_skill_body(skill)})
    except SkillError as e:
        return json.dumps({"error": str(e)})
    return json.dumps({"error": f"unknown internal tool {name!r}"})


@dataclass
class ChatResult:
    reply: str
    n_iterations: int
    artifacts: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)


def _summarize_tool_result(result_text: str, max_chars: int = 400) -> str:
    try:
        data = json.loads(result_text)
    except (ValueError, TypeError):
        return result_text[:max_chars]
    if not isinstance(data, dict):
        return str(data)[:max_chars] if not isinstance(data, list) else f"[list, {len(data)} items]"
    if "error" in data:
        return f"error: {str(data['error'])[:max_chars]}"
    keep = {}
    for k, v in data.items():
        if k in {"figures"}:
            keep[k] = f"[{len(v)} PNG(s)]" if v else "[]"
        elif isinstance(v, (str, int, float, bool, type(None))):
            keep[k] = v if not isinstance(v, str) or len(v) <= 120 else v[:117] + "..."
        elif isinstance(v, list):
            keep[k] = f"[{len(v)} items]"
        elif isinstance(v, dict):
            keep[k] = f"{{{len(v)} keys}}"
    return json.dumps(keep, ensure_ascii=False)[:max_chars]


def _extract_artifact(tool_name: str, result_text: str) -> dict | None:
    """Extract renderable artifacts from tool results (plots, data previews)."""
    try:
        data = json.loads(result_text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or "error" in data:
        return None

    # Python sandbox figures
    if tool_name == "run_python" and data.get("figures"):
        return {
            "tool": tool_name,
            "kind": "image",
            "mime": "image/png",
            "figures": data["figures"],
            "stdout": data.get("stdout", ""),
        }

    # Data preview
    if tool_name == "get_timeseries" and "preview" in data:
        return {
            "tool": tool_name,
            "kind": "data_preview",
            "param_id": data.get("param_id"),
            "n_points": data.get("n_points"),
            "units": data.get("units"),
            "preview": data.get("preview"),
        }

    return None


async def stream_chat(
    llm_client: LLMClient,
    user_id: str,
    session_id: str,
    user_text: str,
) -> AsyncIterator[dict]:
    """Async generator core of the agent loop."""
    history = store.get_or_create(user_id, session_id)
    history.append(Message(role="user", content=user_text))

    tools = registry.list_tool_defs() + _INTERNAL_TOOLS + [task_tool_def()]
    log.info("agent_tools_listed", count=len(tools), tools=[t.name for t in tools])

    try:
        for i in range(settings.agent.max_iterations):
            turn = i + 1
            log.info("llm_call_start", turn=turn, n_messages=len(history))
            t0 = time.monotonic()
            response = await llm_client.chat(history, tools, system_prompt=SYSTEM_PROMPT)
            log.info("llm_call_end", turn=turn, duration_ms=int((time.monotonic() - t0) * 1000),
                     has_tool_calls=bool(response.tool_calls))
            history.append(response)

            if not response.tool_calls:
                store.save(user_id, session_id, history)
                yield {"event": "reply", "data": {"text": response.content}}
                yield {"event": "done", "data": {"n_iterations": turn}}
                return

            for tc in response.tool_calls:
                log.info("tool_call_issued", turn=turn, tool=tc.name)
                yield {"event": "tool_call", "data": {"turn": turn, "name": tc.name, "arguments": tc.arguments}}

                sub_artifacts: list[dict] = []
                sub_end_event: dict | None = None

                try:
                    if tc.name == TASK_TOOL_NAME:
                        args = tc.arguments or {}
                        sub_role = args.get("agent_role", "")
                        sub_desc = args.get("description", "")
                        yield {
                            "event": "sub_agent_start",
                            "data": {"task_id": tc.id, "role": sub_role, "description": sub_desc[:200]},
                        }
                        sub_result = await run_subagent(
                            role=sub_role,
                            description=sub_desc,
                            parent_session_id=session_id,
                            user_id=user_id,
                            llm_client=llm_client,
                            task_id=tc.id,
                        )
                        sub_artifacts = sub_result.artifacts
                        result = json.dumps({
                            "summary": sub_result.summary,
                            "n_iterations": sub_result.n_iterations,
                            "artifacts": sub_result.artifacts,
                            "error": sub_result.error,
                        })
                        sub_end_event = {
                            "task_id": tc.id,
                            "role": sub_role,
                            "summary": (sub_result.summary or "")[:200],
                            "n_iterations": sub_result.n_iterations,
                            "error": sub_result.error,
                        }
                    elif tc.name in _INTERNAL_TOOL_NAMES:
                        result = _dispatch_internal_tool(tc.name, tc.arguments)
                    else:
                        result = await registry.call_tool(tc.name, tc.arguments)
                except Exception as e:
                    log.exception("tool_call_failed", turn=turn, tool=tc.name)
                    result = json.dumps({"error": str(e)})
                    if tc.name == TASK_TOOL_NAME:
                        sub_end_event = {
                            "task_id": tc.id, "role": "", "summary": "", "n_iterations": 0, "error": str(e),
                        }

                yield {
                    "event": "tool_result",
                    "data": {"turn": turn, "name": tc.name, "summary": _summarize_tool_result(result)},
                }

                if tc.name == "load_skill":
                    try:
                        payload = json.loads(result)
                        if payload.get("body") and not payload.get("error"):
                            yield {"event": "skill_loaded", "data": {"name": payload.get("name", "")}}
                    except (ValueError, TypeError):
                        pass

                artifact = _extract_artifact(tc.name, result)
                if artifact:
                    yield {"event": "artifact", "data": artifact}
                for sub_art in sub_artifacts:
                    yield {"event": "artifact", "data": sub_art}
                if sub_end_event is not None:
                    yield {"event": "sub_agent_end", "data": sub_end_event}

                history.append(Message(role="tool", tool_call_id=tc.id, content=result))

        log.warning("agent_loop_capped", max_iterations=settings.agent.max_iterations)
        store.save(user_id, session_id, history)
        yield {"event": "error", "data": {"message": f"agent loop exceeded {settings.agent.max_iterations} iterations"}}

    except asyncio.CancelledError:
        store.save(user_id, session_id, history)
        raise


async def chat(llm_client: LLMClient, user_id: str, session_id: str, user_text: str) -> ChatResult:
    """Non-streaming consumer of stream_chat."""
    artifacts: list[dict] = []
    events: list[dict] = []
    reply = ""
    n_iters = 0
    error_msg: str | None = None

    async for ev in stream_chat(llm_client, user_id, session_id, user_text):
        name, data = ev["event"], ev["data"]
        if name == "reply":
            reply = data.get("text", "")
        elif name == "done":
            n_iters = data.get("n_iterations", 0)
        elif name == "artifact":
            artifacts.append(data)
        elif name == "tool_call":
            events.append({"turn": data["turn"], "type": "tool_call", "tool": data["name"], "arguments": data.get("arguments", {})})
        elif name == "tool_result":
            events.append({"turn": data["turn"], "type": "tool_result", "tool": data["name"], "summary": data.get("summary", "")})
        elif name == "error":
            error_msg = data.get("message", "unknown agent error")

    if error_msg is not None:
        raise RuntimeError(error_msg)
    return ChatResult(reply=reply, n_iterations=n_iters, artifacts=artifacts, events=events)
