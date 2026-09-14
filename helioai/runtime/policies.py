"""What makes the one loop the lead agent or a delegated role.

A `Policy` is data: the prompt, the tools the model sees and the ones it may actually
call, the turn budget, the first turn's `tool_choice`, and the few behavioural switches
the two loops disagreed on. The lead's policy is built from the constants of
`agent_loop`; a role's from `sub_agents.AGENT_ROLES`, which stays the one place roles are
declared.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from helioai.core.llm.base import ToolDef


@dataclass(frozen=True)
class Policy:
    """How one run of the loop behaves.

    Attributes:
        name: `"lead"` or the role name; used in logs and as the `agent` of usage rows.
        system_prompt: The instructions placed before the history on every call.
        tools: The definitions the model is shown.
        max_turns: How many model calls a run may make before it is capped.
        tool_choice_first: `tool_choice` of the first call — `required` for a role, so
            it cannot answer from memory without looking anything up; `auto` for the lead.
        allowed: The tools a run may call. `None` for the lead, who may call anything
            the registry holds; a role's whitelist is enforced at dispatch, not advised.
        sandbox_no_network: Whether `run_python` is denied a network namespace.
        sub_agent_ctx: `{role, task_id}` added to every event a role yields, so an
            interface can nest its trace under the lead's; `None` for the lead.
        comment_replies: Whether text the model writes alongside tool calls is shown as
            a `reply`. The lead thinks aloud for the reader; a role's asides are noise
            the lead never sees.
        stream_replies: Whether the model's text is shown as it is generated
            (`reply_delta` events, then the usual `reply`). The lead's answer is read by
            a person waiting for it; a role's goes to the lead, whole.
        stop_on_empty_reply: Whether a response with neither text nor tool calls ends
            the run as a failure. The lead's does — the usual cause is the output budget,
            and the error names the setting to raise; a role's simply ends with an empty
            summary.
        bogus_retry: Whether an answer quoting ids absent from the catalogue buys the
            model one more turn, with the correction appended, before it is accepted.
        provider: The provider this run's client talks to — for the usage rows, which
            bill per provider. None means the lead's configured one.
        model: The model, when the run was given one of its own (`HELIOAI_ROLE_MODELS`).
        deferred: Tools whose definitions are withheld from the model until it asks for
            them with `search_tools`, or calls one by name. Twenty-one definitions rode
            on every call of a lead turn; the ten formulary and catalogue ones are used
            in a minority of sessions and cost a third of that payload each time.
        final_answer: Whether the model may close the run with `final_answer(answer,
            claims)` — the answer plus the list of numbers it states, each with its
            source — instead of a plain message. Prose is still accepted; the claims are
            what lets a verdict compare numbers by name instead of by regex.
    """

    name: str
    system_prompt: str
    tools: tuple[ToolDef, ...]
    max_turns: int
    tool_choice_first: str = "auto"
    allowed: frozenset[str] | None = None
    sandbox_no_network: bool = False
    sub_agent_ctx: Mapping[str, str] | None = None
    comment_replies: bool = False
    stream_replies: bool = False
    stop_on_empty_reply: bool = False
    bogus_retry: bool = True
    provider: str | None = None
    model: str | None = None
    deferred: frozenset[str] = frozenset()
    final_answer: bool = False

    @property
    def event_extra(self) -> dict:
        """The keys added to every event of the run: a role's context, or nothing."""
        return {"sub_agent_ctx": dict(self.sub_agent_ctx)} if self.sub_agent_ctx else {}
