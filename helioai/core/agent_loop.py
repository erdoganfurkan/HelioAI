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
from helioai.core.sub_agents import TASK_TOOL_NAME, stream_subagent, task_tool_def
from helioai.core.tool_exec import (  # noqa: F401  (re-exported for tests)
    _extract_artifact,
    _summarize_tool_result,
    emit_post_tool_events,
    inject_run_python_args,
)
from helioai.logging_config import get_logger
from helioai.tools.registry import registry

log = get_logger(__name__)


SYSTEM_PROMPT = """You are HelioAI, an expert scientific assistant for heliophysics and space plasma research.

You have access to tools that let you explore and analyze data from 70+ space missions (MMS, Solar Orbiter, Cluster, WIND, ACE, Cassini, MEX, Parker Solar Probe, HelioSwarm…) via the speasy library, run Python code for scientific analysis, and search through 65 000+ parameters.

Discovery tools:
1. `search_parameters` — semantic search over 65k+ speasy parameters. Use English queries. If the user is vague (e.g. "solar wind density"), rewrite as matchable terms (e.g. "solar wind ion number density near Earth ACE WIND"). For several parameters, pass `queries=[...]` to resolve them in one call.
2. `list_missions` — list available data providers. Use when the user asks what data is available.

Data tool:
3. `get_timeseries(param_id, start, stop)` — download a time series from any speasy provider. Always resolve the parameter id via `search_parameters` first. Use ISO 8601 times.

Plasma physics tools (direct, no code needed):
4. `plasma_beta(B_nT, n_cm3, T_eV)` — plasma β with regime interpretation.
5. `gyrofrequency(B_nT, particle)` — cyclotron frequency in Hz (proton/electron/alpha).
6. `debye_length(n_cm3, T_eV)` — electron Debye length in m and km.
7. `alfven_speed(B_nT, n_cm3)` — Alfvén speed in km/s.
8. `inertial_length(n_cm3, particle)` — plasma skin depth in km.
9. `power_spectrum(values, dt_s)` — PSD via Welch, returns peak frequency.

Sandbox tool:
10. `run_python(code)` — isolated Python. Pre-imported: speasy (spz), numpy (np), scipy, matplotlib (plt.show() saves to disk), plasmapy (pf), astropy units (u). Helpers: export(name, array), param_card(var, param_id), clean(values) — converts CDF fill values (|x|≥1e30) and ±inf to NaN before plotting. Use for custom analysis not covered by tools above.

Skills:
5. `list_skills()` — index of available procedural skills.
6. `load_skill(name)` — load a skill's procedure. Call before acting on matching requests.

Sub-agents:
7. `task(description, agent_role)` — delegate to a specialist. Context is EMPTY in the sub — pre-resolve all facts (parameter ids, ISO times, missions) in `description`.

Delegate (do NOT call underlying tools yourself) when:
- Parameter ids are unknown → spawn ONE `parameter_hunter` for ALL parameters at once (list them in the description); it batches them in a single search
- User wants analysis, plots, spectra, multi-mission comparison, or event detection → spawn `data_analyst`
- User asks for plasma quantities (β, gyrofrequency, Debye length…) → spawn `plasma_physicist`

Recommended orchestration order (skip a step if the info is already known):
1. `parameter_hunter` — resolve vague names to speasy param_ids
2. `data_analyst` — download, analyse, plot, compare missions, detect events
3. `plasma_physicist` — plasma parameter computations
4. You (main agent) — interpret and reply, always citing the param_ids used

Workflow rules:
- Always use ISO 8601 times: `2024-01-01T00:00:00`
- Always resolve parameter ids via `search_parameters` before `get_timeseries`
- When `run_python` returns figure_paths, tell the user the plot was saved and is being displayed
- When `run_python` returns exports, interpret the numerical summaries (shape, min/max/mean/std) to answer the user
- In run_python code, call export("name", array) to share numerical results; plt.show() saves the figure to disk
- Reply in the user's language
- Cite the parameter ids you used
"""


def _load_user_profile() -> str:
    """Return the user profile content, or '' when the file does not exist."""
    p = settings.profile.profile_path
    try:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return ""


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


async def stream_chat(
    llm_client: LLMClient,
    user_id: str,
    session_id: str,
    user_text: str,
) -> AsyncIterator[dict]:
    """Async generator core of the agent loop."""
    import helioai.workspace as _ws

    _ws_token = _ws.set_session(session_id)

    history = store.get_or_create(user_id, session_id)
    history.append(Message(role="user", content=user_text))

    existing_dir = store.get_workspace_dir(user_id, session_id)
    if existing_dir:
        _label_token = _ws.set_label(existing_dir)
    else:
        label = _ws.make_session_label(user_text, session_id)
        store.save(user_id, session_id, history)
        store.set_workspace_dir(user_id, session_id, label)
        _label_token = _ws.set_label(label)

    tools = registry.list_tool_defs() + _INTERNAL_TOOLS + [task_tool_def()]
    log.info("agent_tools_listed", count=len(tools), tools=[t.name for t in tools])

    effective_prompt = SYSTEM_PROMPT
    profile = _load_user_profile()
    if profile:
        effective_prompt = f"{SYSTEM_PROMPT}\n\n## User profile\n{profile}"

    try:
        for i in range(settings.agent.max_iterations):
            turn = i + 1
            log.info("llm_call_start", turn=turn, n_messages=len(history))
            t0 = time.monotonic()
            response = await llm_client.chat(history, tools, system_prompt=effective_prompt)
            log.info(
                "llm_call_end",
                turn=turn,
                duration_ms=int((time.monotonic() - t0) * 1000),
                has_tool_calls=bool(response.tool_calls),
            )
            history.append(response)

            if not response.tool_calls:
                store.save(user_id, session_id, history)
                yield {"event": "reply", "data": {"text": response.content}}
                yield {"event": "done", "data": {"n_iterations": turn}}
                return

            for tc in response.tool_calls:
                log.info("tool_call_issued", turn=turn, tool=tc.name)
                yield {
                    "event": "tool_call",
                    "data": {"turn": turn, "name": tc.name, "arguments": tc.arguments},
                }

                sub_end_event: dict | None = None

                try:
                    if tc.name == TASK_TOOL_NAME:
                        args = tc.arguments or {}
                        sub_role = args.get("agent_role", "")
                        sub_desc = args.get("description", "")
                        yield {
                            "event": "sub_agent_start",
                            "data": {
                                "task_id": tc.id,
                                "role": sub_role,
                                "description": sub_desc[:200],
                            },
                        }
                        async for sub_ev in stream_subagent(
                            role=sub_role,
                            description=sub_desc,
                            parent_session_id=session_id,
                            user_id=user_id,
                            llm_client=llm_client,
                            task_id=tc.id,
                        ):
                            if sub_ev["event"] == "sub_agent_end":
                                end_data = sub_ev["data"]
                                result = json.dumps(
                                    {
                                        "summary": end_data.get("summary", ""),
                                        "n_iterations": end_data.get("n_iterations", 0),
                                        "artifacts": end_data.get("artifacts", []),
                                        "error": end_data.get("error"),
                                    }
                                )
                                sub_end_event = {
                                    "task_id": tc.id,
                                    "role": sub_role,
                                    "summary": end_data.get("summary", "")[:200],
                                    "n_iterations": end_data.get("n_iterations", 0),
                                    "error": end_data.get("error"),
                                }
                            else:
                                yield sub_ev
                    elif tc.name in _INTERNAL_TOOL_NAMES:
                        result = _dispatch_internal_tool(tc.name, tc.arguments)
                    else:
                        result = await registry.call_tool(
                            tc.name, inject_run_python_args(tc.name, tc.arguments)
                        )
                except Exception as e:
                    log.exception("tool_call_failed", turn=turn, tool=tc.name)
                    result = json.dumps({"error": str(e)})
                    if tc.name == TASK_TOOL_NAME:
                        sub_end_event = {
                            "task_id": tc.id,
                            "role": sub_role if "sub_role" in locals() else "",
                            "summary": "",
                            "n_iterations": 0,
                            "error": str(e),
                        }

                for ev in emit_post_tool_events(tc.name, result, tool_result_extra={"turn": turn}):
                    yield ev
                if sub_end_event is not None:
                    yield {"event": "sub_agent_end", "data": sub_end_event}

                history.append(Message(role="tool", tool_call_id=tc.id, content=result))

        log.warning("agent_loop_capped", max_iterations=settings.agent.max_iterations)
        store.save(user_id, session_id, history)
        yield {
            "event": "error",
            "data": {"message": f"agent loop exceeded {settings.agent.max_iterations} iterations"},
        }

    except asyncio.CancelledError:
        store.save(user_id, session_id, history)
        raise

    finally:
        _ws.reset_session(_ws_token)
        _ws.reset_label(_label_token)


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
            events.append(
                {
                    "turn": data["turn"],
                    "type": "tool_call",
                    "tool": data["name"],
                    "arguments": data.get("arguments", {}),
                }
            )
        elif name == "tool_result":
            events.append(
                {
                    "turn": data["turn"],
                    "type": "tool_result",
                    "tool": data["name"],
                    "summary": data.get("summary", ""),
                }
            )
        elif name == "error":
            error_msg = data.get("message", "unknown agent error")

    if error_msg is not None:
        raise RuntimeError(error_msg)
    return ChatResult(reply=reply, n_iterations=n_iters, artifacts=artifacts, events=events)
