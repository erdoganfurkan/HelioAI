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

import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from helioai.core.event_display import describe_tool_call
from helioai.core.events import make
from helioai.core.llm.base import LLMClient, Message, ToolCall
from helioai.core.session import strip_orphan_tool_calls
from helioai.core.tool_exec import (
    _history_tool_result,
    cancel_pending,
    compact_history,
    emit_post_tool_events,
    inject_run_python_args,
    start_tool_calls,
    unknown_id_correction,
)
from helioai.core.vision import maybe_review
from helioai.logging_config import get_logger
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
    """

    final_text: str | None
    turns: int
    artifacts: list[dict]
    usage: dict
    capped: bool = False
    empty: bool = False


def _zero_usage() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "n_calls": 0}


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
        artifacts, usage, turns: Progress so far, readable while the run is in flight
            and after it raised — a role that blew up still reports what it measured.
    """

    policy: Policy
    llm: LLMClient
    intercept: Intercept | None = None
    on_llm_call: OnLLMCall | None = None
    registry: ToolRegistry = field(default_factory=lambda: _registry_module.registry)
    artifacts: list[dict] = field(default_factory=list, init=False)
    usage: dict = field(default_factory=_zero_usage, init=False)
    turns: int = field(default=0, init=False)

    async def run(self, history: list[Message]) -> AsyncIterator[dict | RunEnd]:
        """Drive the model over `history` until it answers, stops or hits the cap.

        Args:
            history: The conversation so far, appended to in place — the assistant's
                replies and every tool result land here, so the caller persists the
                same list it passed.

        Yields:
            Event dicts, then one `RunEnd`.
        """
        policy = self.policy
        extra = policy.event_extra
        retried_bogus_ids = False
        started: dict = {}
        try:
            for i in range(policy.max_turns):
                turn = self.turns = i + 1
                history[:] = strip_orphan_tool_calls(history)
                response = await self._call_model(history, turn, first=(i == 0))
                history.append(response)

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
                            log.warning(
                                "invented_ids_retry", agent=policy.name, ids=bogus, turn=turn
                            )
                            history.append(
                                Message(
                                    role="user",
                                    content=unknown_id_correction(bogus),
                                    origin="correction",
                                )
                            )
                            continue
                    yield self._end(final_text)
                    return

                if policy.comment_replies and response.content and response.content.strip():
                    yield make("reply", text=response.content)

                started = start_tool_calls(
                    response.tool_calls, allowed=policy.allowed, registry=self.registry
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
                        handler = self.intercept(tc, turn) if self.intercept else None
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
                        log.exception(
                            "tool_call_failed", agent=policy.name, turn=turn, tool=tc.name
                        )
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
        finally:
            cancel_pending(started)

    async def _call_model(self, history: list[Message], turn: int, *, first: bool) -> Message:
        policy = self.policy
        log.info("llm_call_start", agent=policy.name, turn=turn, n_messages=len(history))
        t0 = time.monotonic()
        kwargs: dict[str, Any] = {"system_prompt": policy.system_prompt}
        # A policy with no opinion on tool_choice leaves the client's default alone,
        # exactly as the lead always did; a role asks for a tool on its first turn.
        if policy.tool_choice_first != "auto":
            kwargs["tool_choice"] = policy.tool_choice_first if first else "auto"
        response = await self.llm.chat(compact_history(history), list(policy.tools), **kwargs)
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
        return response

    async def _dispatch(self, tc: ToolCall, started: dict) -> ToolResult:
        if tc.id in started:
            return await started[tc.id]
        trusted = inject_run_python_args(tc.name, no_network=self.policy.sandbox_no_network)
        return await self.registry.call_tool(tc.name, tc.arguments, trusted=trusted)

    def _end(self, final_text: str | None, *, capped: bool = False, empty: bool = False) -> RunEnd:
        return RunEnd(
            final_text=final_text,
            turns=self.turns,
            artifacts=self.artifacts,
            usage=self.usage,
            capped=capped,
            empty=empty,
        )
