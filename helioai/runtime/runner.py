"""The agent loop, written once.

`Runner.run` is the turn-taking skeleton both `stream_chat` and `stream_subagent` used
to carry as separate copies: call the model with a compacted history, start the turn's
tool calls together, dispatch each in the model's order, review the figures, emit the
events, append the results, retry once when the answer quotes ids that exist in no
catalogue, stop at the cap. What differs between the lead and a role is a `Policy` and
two small hooks — how a tool call may be intercepted before it reaches the registry
(the lead's `task` and internal tools, a role's whitelist), and what to do with each
model call's usage.

The run yields the same `{"event", "data"}` dicts the loops yielded, in the same order,
and ends with a `RunEnd` the wrapper turns into its own closing events — the lead's
`reply … done`, a role's `sub_agent_end`.
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator, Callable
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Any

from helioai.core.event_display import describe_tool_call
from helioai.core.events import make
from helioai.core.llm.base import LLMClient, Message, ToolCall, ToolDef
from helioai.core.session import strip_orphan_tool_calls
from helioai.core.tool_exec import (
    _history_tool_result,
    cancel_pending,
    compact_history,
    emit_post_tool_events,
    start_tool_calls,
    trusted_args,
    unknown_id_correction,
)
from helioai.core.vision import maybe_review
from helioai.logging_config import get_logger
from helioai.runtime.context import RunContext
from helioai.runtime.policies import Policy
from helioai.tools import registry as _registry_module
from helioai.tools.registry import ToolRegistry
from helioai.tools.results import ToolResult

log = get_logger(__name__)

Intercept = Callable[[ToolCall, int], AsyncIterator[dict | ToolResult] | None]
OnLLMCall = Callable[[int, Message], None]


@dataclass
class RunEnd:
    """How a run ended, handed to the wrapper as the last item of `Runner.run`.

    Attributes:
        final_text: The model's closing text; `None` when the run was capped and there
            is no answer to check.
        turns: Model calls made.
        artifacts: Every artifact the run produced, without a role's `sub_agent_ctx` —
            what `check_answer` confronts the answer with.
        usage: Token counts summed over the run's model calls, as the providers
            reported them.
        capped: The turn budget ran out before the model answered.
        empty: The model returned neither text nor a tool call and the policy stops
            there (the lead's output-budget failure).
        claims: The numbers the model says its answer states, each with `name`, `value`,
            `units` and `source`, when it closed with `final_answer`; empty for prose.
    """

    final_text: str | None
    turns: int
    artifacts: list[dict]
    usage: dict
    capped: bool = False
    empty: bool = False
    claims: list[dict] = field(default_factory=list)


def _zero_usage() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "n_calls": 0}


SEARCH_TOOLS_NAME = "search_tools"
FINAL_ANSWER_NAME = "final_answer"

FINAL_ANSWER_DEF = ToolDef(
    name=FINAL_ANSWER_NAME,
    description=(
        "Deliver your final answer together with the list of numbers it states. Call it "
        "alone, once every other tool call of the turn has returned. `answer` is the full "
        "text of the reply; `claims` names each quantity you report with its value, units "
        "and where it comes from."
    ),
    parameters={
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "The complete reply, as prose."},
            "claims": {
                "type": "array",
                "description": "One entry per number stated in the answer.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The export or dataset the number comes from, "
                            "or a short label when it does not.",
                        },
                        "value": {"description": "The number as stated in the answer."},
                        "units": {"type": "string"},
                        "source": {
                            "type": "string",
                            "description": "The export name it was computed as, "
                            "'literature' for a published value, 'asserted' for a number "
                            "you did not compute.",
                        },
                    },
                    "required": ["name", "value"],
                },
            },
        },
        "required": ["answer"],
    },
)

_MAX_CLAIMS = 50


def _claims_from(raw: object) -> list[dict]:
    """Keep the well-formed claims of a `final_answer` call and drop the rest quietly:
    a malformed entry must not cost the answer it travels with."""
    if not isinstance(raw, list):
        return []
    claims: list[dict] = []
    for item in raw[:_MAX_CLAIMS]:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or "value" not in item
        ):
            continue
        claims.append(
            {
                "name": item["name"].strip(),
                "value": item["value"],
                "units": str(item.get("units") or ""),
                "source": str(item.get("source") or "asserted"),
            }
        )
    return claims


SEARCH_TOOLS_DEF = ToolDef(
    name=SEARCH_TOOLS_NAME,
    description=(
        "Find and enable the specialised tools that are not listed in this turn's tool set "
        "(plasma formulary, event catalogues). Describe what you need in a few words; the "
        "matching tools become callable from the next turn."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What the step needs, e.g. 'plasma beta', 'event catalog'.",
            }
        },
        "required": ["query"],
    },
)


@dataclass
class Runner:
    """One run of the loop under one policy.

    Attributes:
        policy: What makes this run the lead or a role.
        llm: The provider client the run calls.
        intercept: Called with each tool call and the turn before the registry is
            consulted. Returns `None` to let the call reach the registry, or an async
            iterator that yields events to forward, then exactly one `ToolResult`, then
            any events to emit *after* the result's own (`tool_result`, artifacts) — how
            the lead re-emits a `sub_agent_end` and a `plan`.
        on_llm_call: Called with the turn and the model's reply after each call; the
            lead records the usage row there. The run also sums usage into `usage`.
        registry: Where tool calls are dispatched. The process-wide registry unless a
            caller hands in another — the wrappers pass their own module's reference,
            which is what the tests that stub a registry patch.
        ctx: Who runs, in which session, writing where. Bound to the workspace
            contextvars for the whole run — the one place they are set — and the source
            of the directories the writing tools receive as trusted arguments. None
            leaves the ambient bindings alone, for a caller that manages its own.
        artifacts, usage, turns: Progress so far, readable while the run is in flight
            and after it raised — a role that blew up still reports what it measured.
    """

    policy: Policy
    llm: LLMClient
    intercept: Intercept | None = None
    on_llm_call: OnLLMCall | None = None
    registry: ToolRegistry = field(default_factory=lambda: _registry_module.registry)
    ctx: RunContext | None = None
    artifacts: list[dict] = field(default_factory=list, init=False)
    usage: dict = field(default_factory=_zero_usage, init=False)
    turns: int = field(default=0, init=False)
    revealed: set[str] = field(default_factory=set, init=False)

    async def run(self, history: list[Message]) -> AsyncIterator[dict | RunEnd]:
        """Drive the model over `history` until it answers, stops or hits the cap.

        Args:
            history: The conversation so far, appended to in place — the assistant's
                replies and every tool result land here, so the caller persists the
                same list it passed.

        Yields:
            Event dicts, then one `RunEnd`.
        """
        with self.ctx.bound() if self.ctx is not None else nullcontext():
            started: dict = {}
            try:
                async for item in self._turns(history, started):
                    yield item
            finally:
                cancel_pending(started)

    async def _turns(self, history: list[Message], started: dict) -> AsyncIterator[dict | RunEnd]:
        policy = self.policy
        extra = policy.event_extra
        retried_bogus_ids = False
        for i in range(policy.max_turns):
            turn = self.turns = i + 1
            history[:] = strip_orphan_tool_calls(history)
            response: Message | None = None
            async for item in self._model_turn(history, turn, first=(i == 0)):
                if isinstance(item, Message):
                    response = item
                else:
                    yield make("reply_delta", text=item, **extra)
            assert response is not None
            history.append(response)

            claims: list[dict] = []
            if policy.final_answer and self._is_final_answer(response):
                # The answer arrived as a tool call. The history keeps it as the plain
                # assistant message it is — the export, the replay and the next turn's
                # context read assistant text, and the wire needs no tool reply for a
                # call that no longer exists.
                args = response.tool_calls[0].arguments or {}
                final_text = str(args.get("answer") or "")
                claims = _claims_from(args.get("claims"))
                history[-1] = replace(response, content=final_text, tool_calls=None)
                response = history[-1]

            if not response.tool_calls:
                final_text = response.content or ""
                if policy.stop_on_empty_reply and not final_text.strip():
                    yield self._end("", empty=True)
                    return
                if policy.bogus_retry and not retried_bogus_ids:
                    from helioai.tools.rag import extract_ids, unknown_ids

                    bogus = unknown_ids(extract_ids(final_text))
                    if bogus:
                        retried_bogus_ids = True
                        log.warning("invented_ids_retry", agent=policy.name, ids=bogus, turn=turn)
                        history.append(
                            Message(
                                role="user",
                                content=unknown_id_correction(bogus),
                                origin="correction",
                            )
                        )
                        continue
                yield self._end(final_text, claims=claims)
                return

            if policy.comment_replies and response.content and response.content.strip():
                yield make("reply", text=response.content)

            # The dict is shared with `run`, whose `finally` cancels whatever is left in
            # it: cleared and refilled rather than rebound, so it stays the same object.
            started.clear()
            started.update(
                start_tool_calls(
                    response.tool_calls,
                    allowed=policy.allowed,
                    registry=self.registry,
                    ctx=self.ctx,
                )
            )
            for tc in response.tool_calls:
                log.info("tool_call_issued", agent=policy.name, turn=turn, tool=tc.name)
                yield make(
                    "tool_call",
                    turn=turn,
                    name=tc.name,
                    arguments=tc.arguments,
                    display=describe_tool_call(tc.name, tc.arguments),
                    **extra,
                )
                result, trailing = None, []
                try:
                    if tc.name == SEARCH_TOOLS_NAME and policy.deferred:
                        result = self._search_tools(tc)
                    elif tc.name == FINAL_ANSWER_NAME and policy.final_answer:
                        result = ToolResult.failure(
                            tc.name,
                            "final_answer must be called alone, once the other tool calls of "
                            "the turn have returned; call it again by itself",
                        )
                    elif tc.name in policy.deferred and tc.name not in self.revealed:
                        # The model named a deferred tool without asking for it: the name
                        # was right, so it already knows the tool, and refusing would only
                        # cost a turn. Revealed and run.
                        self.revealed.add(tc.name)
                        log.info("deferred_tool_called_directly", agent=policy.name, tool=tc.name)
                    handler = (
                        self.intercept(tc, turn) if self.intercept and result is None else None
                    )
                    if handler is not None:
                        async for item in handler:
                            if isinstance(item, ToolResult):
                                result = item
                            elif result is None:
                                yield item
                            else:
                                trailing.append(item)
                    if result is None:
                        result = await self._dispatch(tc, started)
                except Exception as e:
                    log.exception("tool_call_failed", agent=policy.name, turn=turn, tool=tc.name)
                    result = ToolResult.failure(tc.name, str(e) or type(e).__name__)

                result, figure_verdict = await maybe_review(tc.name, result)
                if figure_verdict:
                    yield make("figure_review", turn=turn, text=figure_verdict, **extra)
                for ev in emit_post_tool_events(
                    tc.name,
                    result,
                    tool_result_extra={"turn": turn, **extra},
                    common_extra=extra,
                ):
                    if ev["event"] == "artifact":
                        self.artifacts.append(
                            {k: v for k, v in ev["data"].items() if k != "sub_agent_ctx"}
                        )
                    yield ev
                for ev in trailing:
                    yield ev
                history.append(
                    Message(
                        role="tool",
                        tool_call_id=tc.id,
                        name=tc.name,
                        content=_history_tool_result(tc.name, result.for_llm()),
                    )
                )
        yield self._end(None, capped=True)

    async def _model_turn(
        self, history: list[Message], turn: int, *, first: bool
    ) -> AsyncIterator[str | Message]:
        """One model call: text deltas when the policy streams, then the reply."""
        policy = self.policy
        log.info("llm_call_start", agent=policy.name, turn=turn, n_messages=len(history))
        t0 = time.monotonic()
        kwargs: dict[str, Any] = {"system_prompt": policy.system_prompt}
        # A policy with no opinion on tool_choice leaves the client's default alone,
        # exactly as the lead always did; a role asks for a tool on its first turn.
        if policy.tool_choice_first != "auto":
            kwargs["tool_choice"] = policy.tool_choice_first if first else "auto"
        payload, tools = compact_history(history), self.visible_tools()
        # A duck-typed client without `stream_chat` is a client that does not stream.
        stream = getattr(self.llm, "stream_chat", None) if policy.stream_replies else None
        if stream is not None:
            response = None
            async for item in stream(payload, tools, **kwargs):
                if isinstance(item, Message):
                    response = item
                elif item:
                    yield item
            assert response is not None, "stream_chat must end with the reply"
        else:
            response = await self.llm.chat(payload, tools, **kwargs)
        log.info(
            "llm_call_end",
            agent=policy.name,
            turn=turn,
            duration_ms=int((time.monotonic() - t0) * 1000),
            has_tool_calls=bool(response.tool_calls),
        )
        self.usage["prompt_tokens"] += response.prompt_tokens
        self.usage["completion_tokens"] += response.completion_tokens
        self.usage["cached_tokens"] += response.cached_tokens
        self.usage["n_calls"] += 1
        if self.on_llm_call is not None:
            self.on_llm_call(turn, response)
        yield response

    def visible_tools(self) -> list[ToolDef]:
        """The definitions the model is shown this call: the policy's tools minus the
        deferred ones it has not asked for, plus `search_tools` while any are withheld."""
        policy = self.policy
        withheld = policy.deferred - self.revealed
        shown = [t for t in policy.tools if t.name not in withheld]
        if withheld:
            shown.append(SEARCH_TOOLS_DEF)
        if policy.final_answer:
            shown.append(FINAL_ANSWER_DEF)
        return shown

    @staticmethod
    def _is_final_answer(response: Message) -> bool:
        calls = response.tool_calls or []
        return len(calls) == 1 and calls[0].name == FINAL_ANSWER_NAME

    def _search_tools(self, tc: ToolCall) -> ToolResult:
        query = str((tc.arguments or {}).get("query") or "").lower()
        words = [w for w in re.findall(r"[a-z0-9_]+", query) if len(w) > 2]
        deferred = [t for t in self.policy.tools if t.name in self.policy.deferred]
        matches = [
            t
            for t in deferred
            if any(w in t.name.lower() or w in t.description.lower() for w in words)
        ]
        # A query that matches nothing still means "I need one of these": all of them
        # are revealed rather than sending the model on a second guess.
        chosen = matches or deferred
        self.revealed.update(t.name for t in chosen)
        return ToolResult.from_raw(
            SEARCH_TOOLS_NAME,
            {
                "enabled": [t.name for t in chosen],
                "tools": [{"name": t.name, "description": t.description} for t in chosen],
                "note": "these tools are callable from your next turn",
            },
        )

    async def _dispatch(self, tc: ToolCall, started: dict) -> ToolResult:
        if tc.id in started:
            return await started[tc.id]
        trusted = trusted_args(tc.name, self.ctx, no_network=self.policy.sandbox_no_network)
        return await self.registry.call_tool(tc.name, tc.arguments, trusted=trusted)

    def _end(
        self,
        final_text: str | None,
        *,
        capped: bool = False,
        empty: bool = False,
        claims: list[dict] | None = None,
    ) -> RunEnd:
        return RunEnd(
            final_text=final_text,
            turns=self.turns,
            artifacts=self.artifacts,
            usage=self.usage,
            capped=capped,
            empty=empty,
            claims=list(claims or []),
        )
