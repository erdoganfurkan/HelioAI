"""Sub-agents for delegating focused heliophysics subtasks.

Each role declares a tool whitelist, a system addon, and an optional set
of skills auto-loaded into the sub's system prompt. Sub-agents run in
isolation with a fresh context — the lead's history is invisible to them.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import structlog

from helioai.core.llm.base import LLMClient, Message, ToolDef
from helioai.core.skills_loader import SkillError
from helioai.core.skills_loader import load_skill as load_skill_body
from helioai.core.tool_exec import (
    _history_tool_result,
    check_answer,
    compact_history,
    emit_post_tool_events,
    inject_run_python_args,
    unknown_id_correction,
)
from helioai.core.vision import maybe_review
from helioai.logging_config import get_logger
from helioai.tools.registry import registry

log = get_logger(__name__)

TASK_TOOL_NAME = "task"


@dataclass(frozen=True)
class SubAgentRole:
    """A specialised agent: its prompt, its tool whitelist and its turn budget.

    `allowed_tools` is enforced, not advisory — a role calling outside its set
    gets an error naming what it may use, and the tool is never dispatched.
    """

    name: str
    description: str
    system_addon: str
    allowed_tools: tuple[str, ...]
    max_turns: int = 5
    auto_load_skills: tuple[str, ...] = ()


SUB_SYSTEM_PROMPT_BASE = """You are a focused sub-agent inside HelioAI. The lead agent delegated a narrow task to you.

- Do exactly what the description asks, nothing more.
- You have NO access to the lead agent's conversation. Everything you need is in the description — including, when the session already holds downloaded data, the list of dataset names you can read with `load_data("name")`. Never re-download something that list already covers.
- Reply with a short final summary of what you did and the facts the lead needs (parameter ids, key values, findings).
- Make a reasonable assumption when something is unclear, then proceed.
- Always cite the speasy parameter ids you used.
- For any derived quantity (theta_Bn, beta, V_A, MVAB normal, compression ratio…), state in your reply the method/recipe/library you used and its scientific reference (e.g. "theta_Bn by coplanarity, recipe theta_bn, Schwartz 1998"). Never report a result without naming how it was computed.
- Use your tools to complete the task. If a tool call fails, retry with a corrected argument before concluding.
- `load_data()` returns arrays with NaN wherever the mission declared a fill value. Use `np.nanargmax`/`nanargmin`/`nanmean`/`nanstd`: plain `np.argmax` returns the index of the first NaN, because no comparison ever displaces it — that points a shock detector at a data gap.
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
            "pressure balance, pitch-angle, superposed epoch), load_recipe FIRST and reuse it — "
            "unconditionally, even when you already know the formula. A recipe carries calibrated "
            "parameters (averaging windows, physical constants) and a self-test that code written "
            "from memory does not have; only write custom code when no recipe matches. "
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
        # 8 was not enough for a multi-spacecraft job: discovering that a CDA
        # ephemeris does not cover the requested year costs a turn per candidate,
        # and the notebook's Act IV spent four of them before doing any analysis.
        # ponytail: raising the cap, not indexing time coverage — see tasks/todo.md.
        max_turns=12,
        auto_load_skills=("data_analyst",),
    ),
    "librarian": SubAgentRole(
        name="librarian",
        description=(
            "Find and distill peer-reviewed literature (NASA ADS): papers on an event, "
            "method, instrument or region, returned as citable references."
        ),
        system_addon=(
            "You specialise in literature search on NASA ADS via find_papers. "
            "Funnel strategy: one broad query first, then at most TWO refinements "
            "(author:, title:, year_start/year_end bounds, sort='citations' for "
            "foundational papers, sort='date' for recent work). Never more than 3 "
            "find_papers calls. Reply with 3-5 references maximum, each formatted as "
            "'Author et al. (year), bibcode — one line on what it contributes'. "
            "When the description provides computed values (theta_Bn, Mach number, "
            "shock speed…), explicitly compare them with the ranges reported in the "
            "papers. If nothing relevant comes back, say so and suggest a sharper "
            "query — never pad the reply with weak matches."
        ),
        allowed_tools=("find_papers",),
        max_turns=4,
        auto_load_skills=("librarian",),
    ),
    "plasma_physicist": SubAgentRole(
        name="plasma_physicist",
        description=(
            "Compute plasma properties (beta, gyrofrequency, Debye length...) and quantify "
            "plasma structures (shock jump conditions, theta_Bn, Walén test, MVAB, pressure balance)."
        ),
        system_addon=(
            "You specialise in plasma physics and quantitative structure analysis. "
            "For single-point estimates (beta, gyrofrequency, Debye length, Alfvén speed, inertial length), "
            "use run_python with PlasmaPy (imported as `pf`) and astropy units (imported as `u`), or direct tools. "
            "For time series analysis or derived quantities: always call get_timeseries BEFORE run_python "
            "to download data outside the sandbox, then access it via load_data('name') inside run_python. "
            "For a standard named computation — shock jump conditions (Rankine-Hugoniot), theta_Bn, "
            "Walen test, MVAB discontinuity normal, magnetopause pressure balance — call load_recipe first, "
            "unconditionally, even when you already know the formula: the recipes carry their scientific "
            "reference and calibrated parameters, so a derivation is attributable instead of improvised. "
            "Always export computed values with units using export('name', value, units='...') and "
            "document custom methods using document_method('name', reference='...', method='...'). "
            "Return values with units and physical interpretation."
        ),
        allowed_tools=(
            "search_parameters",
            "get_timeseries",
            "list_recipes",
            "load_recipe",
            "run_python",
        ),
        max_turns=8,
        auto_load_skills=("plasma_physicist",),
    ),
}


@dataclass
class SubAgentResult:
    """What a finished sub-agent hands back to the lead agent."""

    summary: str = ""
    artifacts: list[dict] = field(default_factory=list)
    n_iterations: int = 0
    error: str | None = None


def task_tool_def() -> ToolDef:
    """Build the `task` tool definition offered to the lead agent.

    Deliberately not registered in the ToolRegistry: the agent loop intercepts
    `task` and spawns a sub-agent instead of dispatching a function.

    Returns:
        A ToolDef whose `agent_role` enum lists every available role.
    """
    role_lines = "\n".join(f"  - `{r.name}`: {r.description}" for r in AGENT_ROLES.values())
    return ToolDef(
        name=TASK_TOOL_NAME,
        # Routing rules live in the lead's system prompt, not here. Stating them in both
        # places let them drift into contradiction — this text used to call
        # `parameter_hunter` "required" for unknown ids while the prompt said never to run
        # it first, and the model resolved the conflict differently from one run to the
        # next: three delegations, three, then none, on the same notebook.
        description=(
            "Spawn a specialist sub-agent for ONE focused subtask. "
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


def _with_inventory(description: str) -> str:
    """Prepend the session's already-downloaded datasets to a sub-agent's task.

    The files are reachable — a sub-agent shares the lead's session directory, so
    `load_data(name)` works — but nothing told it which names exist, so it re-downloaded
    parameters the lead had just fetched (the same Wind field four times across one
    notebook). It also made the `plasma_physicist` skill's "the lead fetches the data,
    reach it with load_data(name)" impossible to follow.

    Silent on failure: a missing or unreadable manifest must not stop the task.
    """
    try:
        import helioai.workspace as _ws
        from helioai.datastore import read_manifest

        datasets = read_manifest(_ws.get_session_dir()).get("datasets", {})
    except Exception as e:  # noqa: BLE001 — an inventory is a convenience, never a gate
        log.debug("subagent_inventory_failed", error=str(e))
        return description
    if not datasets:
        return description

    lines = [
        f"- `{name}` — {e.get('param_id', '?')} [{e.get('start', '?')} → {e.get('stop', '?')}]"
        for name, e in datasets.items()
    ]
    return (
        'Already downloaded in this session — read these with `load_data("name")` inside '
        "`run_python` instead of downloading them again:\n"
        + "\n".join(lines)
        + f"\n\nYour task:\n{description}"
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


def _findings(artifacts: list[dict]) -> dict:
    """Table of the values this sub-agent actually computed, keyed by export name.

    The summary is prose the lead re-reads and paraphrases; this is the part that has
    an origin. A number absent from here was never measured by this run, whatever the
    summary says about it — which is exactly the distinction that was missing when a
    sub-agent computed 14.5 nT and the lead published 13.02 nT.

    min/max are carried only when they differ from the mean, so a scalar export stays
    a single number and a time series keeps the range a reply is likely to quote.
    """
    out: dict = {}
    for art in artifacts:
        if art.get("kind") != "exports":
            continue
        for name, stats in (art.get("values") or {}).items():
            if not isinstance(stats, dict) or stats.get("error"):
                continue
            entry = {
                "value": stats.get("mean"),
                "units": stats.get("units", ""),
                "code_path": art.get("code_path"),
            }
            if stats.get("min") != stats.get("max"):
                entry["min"], entry["max"] = stats.get("min"), stats.get("max")
            out[name] = entry
    return out


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

    `summary` is the sub-agent's whole deliverable and is emitted in full: it becomes
    the lead's tool result, so anything cut here is a measurement the lead can no
    longer report and will be tempted to invent. Callers that display it truncate on
    their own side. Reaching the turn cap is reported as an `error`, not as a result,
    for the same reason — a lead handed a capped run has nothing to summarise.
    """
    import helioai.workspace as _ws

    if task_id is None:
        task_id = uuid.uuid4().hex[:8]
    ctx = {"role": role, "task_id": task_id}

    # The user has to be bound too, not just the session: `_root()` resolves the workspace
    # under `current_user()`, which defaults to "web". It works today only because the lead
    # already bound it, so a sub-agent started out of band wrote to the wrong user's files.
    _ws_token = _ws.set_session(parent_session_id)
    _user_token = _ws.set_user(user_id) if user_id else None

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

        history: list[Message] = [Message(role="user", content=_with_inventory(description))]
        artifacts: list[dict] = []
        final_text = ""
        n_iters = 0
        t0 = time.monotonic()
        capped = False
        retried_bogus_ids = False

        for i in range(role_cfg.max_turns):
            n_iters = i + 1
            tc_choice = "required" if i == 0 else "auto"
            response = await llm_client.chat(
                compact_history(history), tools, system_prompt=system_prompt, tool_choice=tc_choice
            )
            history.append(response)

            if not response.tool_calls:
                final_text = response.content or ""
                # Same reasoning as the lead loop: a fabricated id must cost a turn, not
                # a footnote on an answer that has already been written.
                if not retried_bogus_ids:
                    from helioai.tools.rag import extract_ids, unknown_ids

                    bogus_now = unknown_ids(extract_ids(final_text))
                    if bogus_now:
                        retried_bogus_ids = True
                        log.warning("subagent_invented_ids_retry", role=role, ids=bogus_now)
                        history.append(
                            Message(role="user", content=unknown_id_correction(bogus_now))
                        )
                        continue
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

                result, figure_verdict = await maybe_review(tc.name, result)
                if figure_verdict:
                    yield {
                        "event": "figure_review",
                        "data": {"turn": n_iters, "text": figure_verdict, "sub_agent_ctx": ctx},
                    }

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

                history.append(
                    Message(
                        role="tool",
                        tool_call_id=tc.id,
                        content=_history_tool_result(tc.name, result),
                    )
                )
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

        final_text, bogus, bypassed_recipes = check_answer(final_text, history, artifacts)
        if bogus:
            log.warning("subagent_invented_ids", role=role, ids=bogus)
            yield {
                "event": "invalid_ids",
                "data": {"ids": bogus, "sub_agent_ctx": ctx},
            }

        if bypassed_recipes:
            log.warning("subagent_recipe_bypassed", role=role, recipes=bypassed_recipes)
            yield {
                "event": "recipe_bypassed",
                "data": {"recipes": bypassed_recipes, "sub_agent_ctx": ctx},
            }

        yield {
            "event": "sub_agent_end",
            "data": {
                "task_id": task_id,
                "role": role,
                "findings": _findings(artifacts),
                "summary": final_text,
                "n_iterations": n_iters,
                "error": final_text if capped else None,
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
                "findings": _findings(artifacts) if "artifacts" in dir() else {},
                "summary": "",
                "n_iterations": n_iters if "n_iters" in dir() else 0,
                "error": str(e),
                "artifacts": [],
            },
        }

    finally:
        _ws.reset_session(_ws_token)
        if _user_token is not None:
            _ws.reset_user(_user_token)
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
    """Run a sub-agent to completion and return only its outcome.

    Non-streaming wrapper over `stream_subagent` — same arguments.

    Args:
        role: One of the whitelisted roles (parameter_hunter, data_analyst,
            plasma_physicist, librarian).
        description: The task handed to the sub-agent, in natural language.
        parent_session_id: The lead conversation this run belongs to.
        user_id: Storage namespace of that conversation.
        llm_client: Provider client shared with the lead.
        task_id: Optional id echoed in events, for UI correlation.

    Returns:
        SubAgentResult — `summary` (full final text), `artifacts`,
        `n_iterations`, and `error` (set when the run failed or hit its cap).
    """
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
