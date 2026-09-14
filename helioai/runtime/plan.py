"""The plan the model announced, kept as data, and how the run then followed it.

`present_plan(title, steps)` was a display: the loop forwarded it to the interfaces as a
`plan` event and forgot it. What the model then did was left to the reader to compare
with what it had said — three tool calls later nobody remembers step 2 named
`get_timeseries`, and a plan that promised the `theta_bn` recipe and ran hand-written
arithmetic instead looked exactly like one that was followed. `Plan` is the announced
plan as data; `adherence` places the turn's tool calls against it and reports, in one
`plan_report` event, which planned tools were used, which were not, and which tools were
used without being planned. The report describes; it never blocks, corrects or retries.
Only the lead's own calls count — a step the lead delegated with `task` is a `task` call,
which is what the plan should have said — and the calls that are scaffolding rather than
steps (the plan itself, the skills, `search_tools`, `final_answer`) count for nothing on
either side.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

SCAFFOLDING: frozenset[str] = frozenset(
    {"present_plan", "list_skills", "load_skill", "search_tools", "final_answer"}
)


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

    @property
    def tools(self) -> list[str]:
        """The tools the plan names, once each, in the order of their first step."""
        return list(dict.fromkeys(s.tool for s in self.steps if s.tool))


def executed_tools(events: Iterable[dict]) -> list[str]:
    """The tools the lead itself called during a turn, once each, in call order.

    Args:
        events: The turn's events; only `tool_call` events without a `sub_agent_ctx`
            count, and the scaffolding calls never do.

    Returns:
        Tool names in the order of their first call.
    """
    names = (
        ev["data"].get("name")
        for ev in events
        if ev.get("event") == "tool_call" and "sub_agent_ctx" not in ev["data"]
    )
    return list(dict.fromkeys(n for n in names if n and n not in SCAFFOLDING))


def adherence(plan: Plan, events: Iterable[dict]) -> dict:
    """Compare what the run did with what the plan said.

    Args:
        plan: The plan the turn opened with.
        events: The turn's events, as yielded.

    Returns:
        The `plan_report` payload — `title`, `planned` (the tools the plan named),
        `executed` (the tools the lead called), `unplanned_tools` (called, never
        planned), `missed_tools` (planned, never called) and `ratio`, the share of
        planned tools that were called, or None when the plan named no tool and there
        is nothing to hold the run to.
    """
    planned = plan.tools
    executed = executed_tools(events)
    followed = [t for t in planned if t in executed]
    return {
        "title": plan.title,
        "planned": planned,
        "executed": executed,
        "unplanned_tools": [t for t in executed if t not in planned],
        "missed_tools": [t for t in planned if t not in executed],
        "ratio": round(len(followed) / len(planned), 2) if planned else None,
    }
