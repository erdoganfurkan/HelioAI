"""The plan the model announced, kept as data, and how the run then followed it.

`present_plan(title, steps)` was a display: the loop forwarded it to the interfaces as a
`plan` event and forgot it. What the model then did was left to the reader to compare
with what it had said — three tool calls later nobody remembers step 2 named
`get_timeseries`, and a plan that promised the `theta_bn` recipe and ran hand-written
arithmetic instead looked exactly like one that was followed. `Plan` is the announced
plan as data; `adherence` places the turn's tool calls against it and reports, in one
`plan_report` event, which planned tools were used, which were not, and which tools were
used without being planned. The report describes; it never blocks, corrects or retries.

Two things the first live run settled. A step's `tool` field is prose — the model wrote
"search_parameters + get_timeseries" and "task (data_analyst)" — so it is read word by
word against the tools the lead knows, not compared whole. And a planned tool the lead
had a sub-agent run is a step done, not a deviation: the lead planned the analysis in its
analyst's tools and then delegated every step, and the report said `0/4 used, unplanned:
task`. A sub-agent's calls therefore satisfy the plan; they are never unplanned — the
second run planned in delegations alone (`task (data_analyst)`) and its analyst's six
tools were all "unplanned" — because the `task` step covers whatever the role does, and
the role's whitelist, not the lead's plan, governs it. What the lead does with its own
hands is held to the plan, `task` excepted: delegating is how a step gets done. The
scaffolding calls (the plan itself, the skills, `search_tools`, `final_answer`) count for
nothing on either side.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass

SCAFFOLDING: frozenset[str] = frozenset(
    {"present_plan", "list_skills", "load_skill", "search_tools", "final_answer"}
)
DELEGATION = "task"

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class Step:
    """One announced step.

    Attributes:
        description: What the step does, as the model put it.
        tool: The tool it said it would use, or None when it named none.
    """

    description: str
    tool: str | None = None


@dataclass(frozen=True)
class Plan:
    """A plan as `present_plan` delivered it, with the payload's looseness removed.

    Attributes:
        title: The plan's one-line title.
        steps: The steps in order; a step the model wrote as a bare string is kept as a
            description without a tool.
    """

    title: str
    steps: tuple[Step, ...] = ()

    @classmethod
    def from_payload(cls, data: dict) -> Plan:
        """Build a plan from a `present_plan` payload or a `plan` event's data.

        Args:
            data: `{"title": str, "steps": [{"description", "tool"} | str, ...]}`, any
                part of which the model may have left out.

        Returns:
            The plan, with malformed steps dropped rather than raised on.
        """
        steps: list[Step] = []
        for raw in data.get("steps") or []:
            if isinstance(raw, str):
                steps.append(Step(raw))
            elif isinstance(raw, dict):
                tool = str(raw.get("tool") or "").strip() or None
                steps.append(Step(str(raw.get("description") or ""), tool))
        return cls(str(data.get("title") or ""), tuple(steps))

    def tools(self, known: Collection[str] | None = None) -> list[str]:
        """The tools the plan names, once each, in the order of their first mention.

        Args:
            known: The tools the lead can call. A step's `tool` field is read word by
                word and only these words count; None keeps every word, for a caller
                with no registry at hand.

        Returns:
            Tool names, first mention first.
        """
        words = (w for s in self.steps if s.tool for w in _WORD.findall(s.tool))
        return list(dict.fromkeys(w for w in words if known is None or w in known))


def calls_made(events: Iterable[dict]) -> tuple[list[str], list[str]]:
    """The tools a turn called: the lead's own, and its sub-agents', each once in order.

    Args:
        events: The turn's events; `tool_call` events carrying a `sub_agent_ctx` are a
            sub-agent's, the others the lead's. Scaffolding calls count for neither.

    Returns:
        `(own, delegated)`, tool names in the order of their first call.
    """
    own: dict[str, None] = {}
    delegated: dict[str, None] = {}
    for ev in events:
        if ev.get("event") != "tool_call":
            continue
        name = ev["data"].get("name")
        if not name or name in SCAFFOLDING:
            continue
        (delegated if "sub_agent_ctx" in ev["data"] else own).setdefault(name, None)
    return list(own), list(delegated)


def delegations_made(events: Iterable[dict]) -> list[dict]:
    """How each delegation of the turn ended: the role, its turns, whether it was capped.

    A plan written in delegations alone ("task (data_analyst)") is always followed by
    delegating, so the tools say nothing; what a reader wants to know is whether the
    role finished. The run after the Runner extraction is the case: the librarian hit
    its four-turn cap, was re-delegated and finished in two — visible in the trace,
    invisible in a count of tools.

    Args:
        events: The turn's events; the lead's own `sub_agent_end` re-emissions count
            (those without a `sub_agent_ctx`).

    Returns:
        `[{role, n_iterations, capped}]` in order of completion.
    """
    return [
        {
            "role": ev["data"].get("role", ""),
            "n_iterations": int(ev["data"].get("n_iterations") or 0),
            "capped": bool(ev["data"].get("capped", False)),
        }
        for ev in events
        if ev.get("event") == "sub_agent_end" and "sub_agent_ctx" not in ev["data"]
    ]


def adherence(plan: Plan, events: Iterable[dict], known: Collection[str] | None = None) -> dict:
    """Compare what the run did with what the plan said.

    Args:
        plan: The plan the turn opened with.
        events: The turn's events, as yielded.
        known: The tools the lead can call, to read the plan's `tool` fields against.

    Returns:
        The `plan_report` payload — `title`, `planned` (the tools the plan named),
        `executed` (the tools the lead called itself), `delegated` (the tools its
        sub-agents called), `delegations` (each sub-agent run: role, turns, capped),
        `unplanned_tools` (the lead's own calls that were never planned, `task`
        excepted), `missed_tools` (planned, called by neither) and `ratio`, the share
        of planned tools that were called, or None when the plan named no tool and
        there is nothing to hold the run to.
    """
    planned = plan.tools(known)
    own, delegated = calls_made(events)
    done = list(dict.fromkeys(own + delegated))
    followed = [t for t in planned if t in done]
    return {
        "title": plan.title,
        "planned": planned,
        "executed": own,
        "delegated": delegated,
        "delegations": delegations_made(events),
        "unplanned_tools": [t for t in own if t not in planned and t != DELEGATION],
        "missed_tools": [t for t in planned if t not in done],
        "ratio": round(len(followed) / len(planned), 2) if planned else None,
    }
