"""Sub-agents for delegating focused heliophysics subtasks.

Each role declares a tool whitelist, a system addon, and an optional set
of skills auto-loaded into the sub's system prompt. Sub-agents run in
isolation with a fresh context — the lead's history is invisible to them.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

import structlog

from helioai.core.llm.base import LLMClient, Message, ToolDef
from helioai.core.skills_loader import SkillError, load_skill as load_skill_body
from helioai.core.tool_exec import compact_history, emit_post_tool_events, inject_run_python_args
from helioai.logging_config import get_logger
from helioai.tools.registry import registry

log = get_logger(__name__)

TASK_TOOL_NAME = "task"


@dataclass(frozen=True)
class SubAgentRole:
    name: str
    description: str
    system_addon: str
    allowed_tools: tuple[str, ...]
    max_turns: int = 5
    auto_load_skills: tuple[str, ...] = ()


SUB_SYSTEM_PROMPT_BASE = """You are a focused sub-agent inside HelioAI. The lead agent delegated a narrow task to you.

- Do exactly what the description asks, nothing more.
- You have NO access to the lead agent's conversation. Everything you need is in the description.
- Reply with a short final summary of what you did and the facts the lead needs (parameter ids, key values, findings).
- Make a reasonable assumption when something is unclear, then proceed.
- Always cite the speasy parameter ids you used.
- For any derived quantity (theta_Bn, beta, V_A, MVAB normal, compression ratio…), state in your reply the method/recipe/library you used and its scientific reference (e.g. "theta_Bn by coplanarity, recipe theta_bn, Schwartz 1998"). Never report a result without naming how it was computed.
- Use your tools to complete the task. If a tool call fails, retry with a corrected argument before concluding.
"""


AGENT_ROLES: dict[str, SubAgentRole] = {
    "parameter_hunter": SubAgentRole(
        name="parameter_hunter",
        description="Resolve a vague parameter description into one or more speasy parameter ids.",
        system_addon=(
            "You specialise in parameter discovery. Given a description that may list "
            "SEVERAL parameters, resolve them all in ONE batched call: "
            "search_parameters(queries=[...]). Pass provider= (amda/cda/csa/ssc) to scope "
            "when a provider is named. Re-query only the weak results. Reply with each "
            "id, units, mission, and available time range. "
            "Do not download or analyze data — that is the lead's job."
        ),
        allowed_tools=("search_parameters", "list_missions"),
        max_turns=4,
        auto_load_skills=("parameter_hunter",),
    ),
    "data_analyst": SubAgentRole(
        name="data_analyst",
        description=(
            "Download and analyze heliophysics data: plots, spectra, statistics, "
            "multi-mission comparison, and event detection (shocks, reconnection, CMEs…)."
        ),
        system_addon=(
            "You specialise in data analysis, visualisation, multi-mission comparison, "
            "and plasma event detection. "
            "For a standard named computation (theta_Bn, MVAB normal, Rankine-Hugoniot, Walén test, "
            "pressure balance, pitch-angle, superposed epoch), load_recipe and reuse it instead of "
            "writing your own — only write custom code when no recipe matches. "
            "Use search_parameters if any parameter id is missing or unclear. "
            "CRITICAL — to avoid sandbox timeouts: always call get_timeseries BEFORE run_python "
            "to download data outside the sandbox, then access it via load_data('name') inside run_python. "
            "NEVER call spz.get_data() inside run_python for data you can download with get_timeseries first. "
            "If get_timeseries returns a `quality` block with `notable: true`, report it (missing %, gaps, "
            "5sigma outliers) — deterministic checks, not guesses. "
            "run_python is the ONLY tool that produces figures. "
            "A text description of a plot is not a plot. Always call run_python. "
            "For multi-mission work: call get_timeseries once per mission, then load all datasets "
            "in a single run_python call via load_data(). "
            "For SEA: call get_events_timeseries first, then load_recipe('superposed_epoch'), then run_python. "
            "For event detection: implement threshold / derivative / boundary criteria in "
            "run_python; report event times, key signatures, and SPASE PhenomenonType if applicable."
        ),
        allowed_tools=(
            "search_parameters",
            "get_timeseries",
            "get_events_timeseries",
            "load_recipe",
            "run_python",
        ),
        max_turns=8,
        auto_load_skills=("data_analyst",),
    ),
    "plasma_physicist": SubAgentRole(
        name="plasma_physicist",
        description="Compute plasma parameters (gyrofrequency, Debye length, plasma beta, etc.).",
        system_addon=(
            "You specialise in plasma physics calculations. Use run_python with "
            "plasmapy (imported as `pf`) and astropy units (imported as `u`). "
            "Example: pf.gyrofrequency(B=40*u.nT, particle='p+').to(u.Hz). "
            "Return values with units and physical interpretation."
        ),
        allowed_tools=("run_python", "search_parameters"),
        max_turns=4,
        auto_load_skills=("plasma_physicist",),
    ),
}


@dataclass
class SubAgentResult:
    summary: str = ""
    artifacts: list[dict] = field(default_factory=list)
    n_iterations: int = 0
    error: str | None = None


def task_tool_def() -> ToolDef:
    role_lines = "\n".join(f"  - `{r.name}`: {r.description}" for r in AGENT_ROLES.values())
    return ToolDef(
        name=TASK_TOOL_NAME,
        description=(
            "Spawn a specialist sub-agent for ONE focused subtask. "
            "Required when: (1) param ids are unknown → spawn ONE `parameter_hunter` for ALL the parameters at once (list them in the description); "
            "(2) user wants analysis, plots, spectra, multi-mission comparison, or event detection → spawn `data_analyst`; "
            "(3) user wants plasma physics calculations (β, gyrofrequency, Debye length…) → spawn `plasma_physicist`. "
            "The sub runs in isolation (empty context) — pre-resolve every fact "
            "(param ids, ISO times, missions) inside `description`.\n\nRoles:\n" + role_lines
        ),
        parameters={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Self-contained task description (1-3 sentences) with all needed facts.",
                },
                "agent_role": {
                    "type": "string",
                    "enum": sorted(AGENT_ROLES.keys()),
                    "description": "Sub-agent role to spawn.",
                },
            },
            "required": ["description", "agent_role"],
        },
    )


def _build_system_prompt(role: SubAgentRole) -> tuple[str, list[str]]:
    loaded: list[str] = []
    skill_blocks: list[str] = []
    for skill_name in role.auto_load_skills:
        try:
            body = load_skill_body(skill_name).strip()
            skill_blocks.append(f"## Skill: {skill_name}\n\n{body}")
            loaded.append(skill_name)
        except SkillError as e:
            log.warning(
                "subagent_skill_load_failed", role=role.name, skill=skill_name, error=str(e)
            )
    prompt = SUB_SYSTEM_PROMPT_BASE + "\n" + role.system_addon
    if skill_blocks:
        prompt += "\n\n# Pre-loaded skills\n\n" + "\n\n".join(skill_blocks)
    return prompt, loaded


async def stream_subagent(
    role: str,
    description: str,
    *,
    parent_session_id: str,
    user_id: str,
    llm_client: LLMClient,
    task_id: str | None = None,
) -> AsyncIterator[dict]:
    """Async generator that runs a sub-agent and yields progress events.

    Yields the same event types as stream_chat (tool_call, tool_result,
    skill_loaded, artifact) enriched with sub_agent_ctx={role, task_id},
    then a final sub_agent_end event carrying summary/artifacts/n_iterations/error.
    """
    import helioai.workspace as _ws

    if task_id is None:
        task_id = uuid.uuid4().hex[:8]
    ctx = {"role": role, "task_id": task_id}

    _ws_token = _ws.set_session(parent_session_id)

    if role not in AGENT_ROLES:
        known = ", ".join(sorted(AGENT_ROLES))
        yield {
            "event": "sub_agent_end",
            "data": {
                "task_id": task_id,
                "role": role,
                "summary": "",
                "n_iterations": 0,
                "error": f"unknown agent_role {role!r}. Known: {known}",
                "artifacts": [],
            },
        }
        return

    role_cfg = AGENT_ROLES[role]

    structlog.contextvars.bind_contextvars(
        parent_session_id=parent_session_id,
        sub_role=role,
        sub_task_id=task_id,
    )

    try:
        system_prompt, skills_loaded = _build_system_prompt(role_cfg)
        for skill_name in skills_loaded:
            yield {"event": "skill_loaded", "data": {"name": skill_name, "sub_agent_ctx": ctx}}

        allowed = set(role_cfg.allowed_tools)
        tools = registry.list_tool_defs(only=allowed)

        log.info(
            "subagent_start",
            role=role,
            description=description[:200],
            allowed_tools=sorted(allowed),
            max_turns=role_cfg.max_turns,
        )

        history: list[Message] = [Message(role="user", content=description)]
        artifacts: list[dict] = []
        final_text = ""
        n_iters = 0
        t0 = time.monotonic()
        capped = False

        for i in range(role_cfg.max_turns):
            n_iters = i + 1
            tc_choice = "required" if i == 0 else "auto"
            response = await llm_client.chat(
                compact_history(history), tools, system_prompt=system_prompt, tool_choice=tc_choice
            )
            history.append(response)

            if not response.tool_calls:
                final_text = response.content or ""
                break

            for tc in response.tool_calls:
                log.info("tool_call_issued", turn=n_iters, tool=tc.name, sub_role=role)
                yield {
                    "event": "tool_call",
                    "data": {
                        "turn": n_iters,
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "sub_agent_ctx": ctx,
                    },
                }

                if tc.name not in allowed:
                    log.warning("subagent_tool_denied", role=role, tool=tc.name)
                    result = json.dumps(
                        {
                            "error": f"tool {tc.name!r} not available to {role!r}. Allowed: {sorted(allowed)}"
                        }
                    )
                else:
                    result = await registry.call_tool(
                        tc.name, tc.arguments, trusted=inject_run_python_args(tc.name)
                    )

                for ev in emit_post_tool_events(
                    tc.name,
                    result,
                    tool_result_extra={"turn": n_iters, "sub_agent_ctx": ctx},
                    common_extra={"sub_agent_ctx": ctx},
                ):
                    if ev["event"] == "artifact":
                        artifacts.append(
                            {k: v for k, v in ev["data"].items() if k != "sub_agent_ctx"}
                        )
                    yield ev

                history.append(Message(role="tool", tool_call_id=tc.id, content=result))
        else:
            capped = True
            final_text = f"(sub-agent {role!r} reached its {role_cfg.max_turns}-turn cap)"

        log.info(
            "subagent_end",
            role=role,
            n_iterations=n_iters,
            duration_ms=int((time.monotonic() - t0) * 1000),
            capped=capped,
            n_artifacts=len(artifacts),
        )

        yield {
            "event": "sub_agent_end",
            "data": {
                "task_id": task_id,
                "role": role,
                "summary": final_text[:200],
                "n_iterations": n_iters,
                "error": None,
                "artifacts": artifacts,
            },
        }

    except Exception as e:
        log.exception("subagent_error", role=role, task_id=task_id)
        yield {
            "event": "sub_agent_end",
            "data": {
                "task_id": task_id,
                "role": role,
                "summary": "",
                "n_iterations": n_iters if "n_iters" in dir() else 0,
                "error": str(e),
                "artifacts": [],
            },
        }

    finally:
        _ws.reset_session(_ws_token)
        structlog.contextvars.unbind_contextvars("parent_session_id", "sub_role", "sub_task_id")


async def run_subagent(
    role: str,
    description: str,
    *,
    parent_session_id: str,
    user_id: str,
    llm_client: LLMClient,
    task_id: str | None = None,
) -> SubAgentResult:
    """Backward-compat wrapper — consumes stream_subagent and returns SubAgentResult."""
    async for ev in stream_subagent(
        role=role,
        description=description,
        parent_session_id=parent_session_id,
        user_id=user_id,
        llm_client=llm_client,
        task_id=task_id,
    ):
        if ev["event"] == "sub_agent_end":
            d = ev["data"]
            return SubAgentResult(
                summary=d.get("summary", ""),
                artifacts=d.get("artifacts", []),
                n_iterations=d.get("n_iterations", 0),
                error=d.get("error"),
            )
    return SubAgentResult(error="stream_subagent yielded no sub_agent_end")
