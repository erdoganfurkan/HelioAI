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
from typing import Any

import structlog

from helioai.core.llm.base import LLMClient, Message, ToolDef
from helioai.core.skills_loader import SkillError, load_skill as load_skill_body
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
- Never ask clarifying questions — make a reasonable assumption and proceed.
- Always cite the speasy parameter ids you used.
"""


AGENT_ROLES: dict[str, SubAgentRole] = {
    "parameter_hunter": SubAgentRole(
        name="parameter_hunter",
        description="Resolve a vague parameter description into one or more speasy parameter ids.",
        system_addon=(
            "You specialise in parameter discovery. Given a description "
            "(mission, quantity, region…), find the matching speasy parameter id(s) "
            "via search_parameters. Reply with the id(s), units, mission, and "
            "available time range. Do not download or analyze data — that is the lead's job."
        ),
        allowed_tools=("search_parameters", "list_missions"),
        max_turns=4,
    ),
    "data_analyst": SubAgentRole(
        name="data_analyst",
        description="Analyze already-retrieved data with Python: spectra, statistics, plasma params.",
        system_addon=(
            "You specialise in quantitative analysis. The lead has already resolved "
            "parameter ids and time intervals. Use get_timeseries to load data then "
            "run_python for analysis (FFT, power spectrum, statistics, PlasmaPy calculations). "
            "Return key numerical results and figures."
        ),
        allowed_tools=("get_timeseries", "run_python"),
        max_turns=6,
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
    ),
    "cross_mission": SubAgentRole(
        name="cross_mission",
        description="Compare the same physical quantity across 2+ missions or time intervals.",
        system_addon=(
            "You specialise in multi-mission comparison. The lead has specified which "
            "quantities and missions to compare. Resolve parameter ids per mission via "
            "search_parameters, then download via get_timeseries. Summarise differences "
            "and commonalities. Use run_python for aligned comparison plots when useful."
        ),
        allowed_tools=("search_parameters", "get_timeseries", "run_python"),
        max_turns=6,
    ),
    "event_detector": SubAgentRole(
        name="event_detector",
        description="Identify plasma events: shocks, reconnection, magnetopause crossings, CMEs.",
        system_addon=(
            "You specialise in event detection. Use get_timeseries to fetch relevant "
            "parameters, then run_python to implement detection criteria. "
            "Common markers: B rotation (reconnection), density jump + B compression (shocks), "
            "velocity reversal (current sheets). Report event times and key signatures."
        ),
        allowed_tools=("search_parameters", "get_timeseries", "run_python"),
        max_turns=6,
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
            "Required when: (1) user needs 2+ parameters → one `parameter_hunter` per parameter; "
            "(2) user wants complex analysis → spawn `data_analyst`; "
            "(3) user wants plasma physics calculations → spawn `plasma_physicist`; "
            "(4) multi-mission comparison → `cross_mission`; "
            "(5) event detection → `event_detector`. "
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
            log.warning("subagent_skill_load_failed", role=role.name, skill=skill_name, error=str(e))
    prompt = SUB_SYSTEM_PROMPT_BASE + "\n" + role.system_addon
    if skill_blocks:
        prompt += "\n\n# Pre-loaded skills\n\n" + "\n\n".join(skill_blocks)
    return prompt, loaded


async def run_subagent(
    role: str,
    description: str,
    *,
    parent_session_id: str,
    user_id: str,
    llm_client: LLMClient,
    task_id: str | None = None,
) -> SubAgentResult:
    if role not in AGENT_ROLES:
        known = ", ".join(sorted(AGENT_ROLES))
        return SubAgentResult(error=f"unknown agent_role {role!r}. Known: {known}")

    role_cfg = AGENT_ROLES[role]

    from helioai.core.agent_loop import _extract_artifact

    if task_id is None:
        task_id = uuid.uuid4().hex[:8]

    structlog.contextvars.bind_contextvars(
        parent_session_id=parent_session_id,
        sub_role=role,
        sub_task_id=task_id,
    )

    try:
        system_prompt, skills_loaded = _build_system_prompt(role_cfg)

        allowed = set(role_cfg.allowed_tools)
        tools = registry.list_tool_defs(only=allowed)

        log.info("subagent_start", role=role, description=description[:200],
                 allowed_tools=sorted(allowed), max_turns=role_cfg.max_turns)

        history: list[Message] = [Message(role="user", content=description)]
        artifacts: list[dict] = []
        final_text = ""
        n_iters = 0
        t0 = time.monotonic()
        capped = False

        for i in range(role_cfg.max_turns):
            n_iters = i + 1
            response = await llm_client.chat(history, tools, system_prompt=system_prompt)
            history.append(response)

            if not response.tool_calls:
                final_text = response.content or ""
                break

            for tc in response.tool_calls:
                if tc.name not in allowed:
                    log.warning("subagent_tool_denied", role=role, tool=tc.name)
                    result = json.dumps({
                        "error": f"tool {tc.name!r} not available to {role!r}. Allowed: {sorted(allowed)}"
                    })
                else:
                    result = await registry.call_tool(tc.name, tc.arguments)

                art = _extract_artifact(tc.name, result)
                if art:
                    artifacts.append(art)

                history.append(Message(role="tool", tool_call_id=tc.id, content=result))
        else:
            capped = True
            final_text = f"(sub-agent {role!r} reached its {role_cfg.max_turns}-turn cap)"

        log.info("subagent_end", role=role, n_iterations=n_iters,
                 duration_ms=int((time.monotonic() - t0) * 1000),
                 capped=capped, n_artifacts=len(artifacts))
        return SubAgentResult(summary=final_text, artifacts=artifacts, n_iterations=n_iters)

    finally:
        structlog.contextvars.unbind_contextvars("parent_session_id", "sub_role", "sub_task_id")
