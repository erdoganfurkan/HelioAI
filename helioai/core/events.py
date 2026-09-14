"""The event contract between the agent loops and every interface, in one place.

`stream_chat` and `stream_subagent` yield `{"event": kind, "data": {...}}` dicts. The
kinds were listed in three docstrings that disagreed with each other and with the code
(`artifact`, `figure_review`, `invalid_ids` and `recipe_bypassed` were missing from the
module docstring of `agent_loop`; `sub_agents` claimed to yield the same set as the lead
and did not). This module is the list the code and the tests are held to:
`tests/test_events_contract.py` greps the emitters and the three renderers and fails
when a kind appears in one and not the other.

Payloads stay dicts — every interface consumes them as JSON — and every emitter builds
them through `make` and `artifact`, which check the kind and its required keys as the
event is produced. A branch that forgets a key fails in the test that exercises it, not
as a blank line in one interface a month later. The dict shape is exactly what the loops
always yielded; consumers see no change.
"""

from __future__ import annotations

from typing import Any

# kind → the payload keys every emitter provides (extra keys may travel along;
# `sub_agent_ctx` is added to any kind a sub-agent emits).
KINDS: dict[str, tuple[str, ...]] = {
    # The question that opens a turn. Yielded and journaled first, so a session replays
    # from its journal alone; the CLI and the notebook leave it unrendered — the person
    # who typed it is looking at it.
    "user": ("text",),
    # A slice of the reply as the model writes it; the full `reply` still follows, so
    # renderers that ignore deltas lose nothing. Not journaled: the `reply` is.
    "reply_delta": ("text",),
    "reply": ("text",),
    "tool_call": ("turn", "name", "arguments", "display"),
    "tool_result": ("turn", "name", "summary", "display"),
    "artifact": ("kind", "tool"),
    "skill_loaded": ("name",),
    "plan": ("title", "steps"),
    # How the turn followed the plan it opened with: the tools the plan named, the ones
    # the lead called, and the two differences. Descriptive only — emitted once, at the
    # end of a turn that presented a plan, and never blocks or corrects anything.
    "plan_report": (
        "title",
        "planned",
        "executed",
        "delegated",
        "unplanned_tools",
        "missed_tools",
        "ratio",
    ),
    "figure_review": ("turn", "text"),
    "provenance": ("matched", "contradicted", "derived", "unsourced", "details"),
    # The one judgement on an answer that named its numbers (`final_answer`): each claim
    # placed against the ledger by name, plus the id, recipe and figure checks the
    # other kinds report one by one. Emitted only when there are claims to judge.
    "verdict": (
        "matched",
        "contradicted",
        "unsourced",
        "unknown_ids",
        "recipe_flags",
        "figure_reviews",
        "claims",
    ),
    "invalid_ids": ("ids",),
    "recipe_bypassed": ("recipes",),
    "sub_agent_start": ("task_id", "role", "description"),
    "sub_agent_end": ("task_id", "role", "summary", "n_iterations", "error", "findings", "usage"),
    "error": ("message",),
    "done": ("n_iterations",),
}

# `artifact.data.kind` values, each with the keys a renderer may rely on.
ARTIFACT_KINDS: dict[str, tuple[str, ...]] = {
    "image": ("figure_paths",),
    "exports": ("values",),
    "parameter_card": ("param_id",),
    "recipe_used": ("name", "reference", "description"),
    "code": ("code_path", "name"),
    "catalog_preview": ("catalog_id", "columns", "sample"),
}

# Kinds the lead emits and a sub-agent never does, and the reverse.
LEAD_ONLY: frozenset[str] = frozenset(
    {
        "user",
        "reply_delta",
        "reply",
        "plan",
        "plan_report",
        "provenance",
        "verdict",
        "sub_agent_start",
        "error",
        "done",
    }
)

# Kinds the journal leaves out: transient by nature, and fully covered by another kind.
NOT_JOURNALED: frozenset[str] = frozenset({"reply_delta"})

# Artifact kinds an interface may leave unrendered on purpose. `exports` is recorded
# in the provenance ledger and shown through `findings`, not as a card of its own.
RENDER_OPTIONAL: frozenset[str] = frozenset({"exports"})


def make(kind: str, /, **data: Any) -> dict:
    """Build one event, checked against the contract.

    Args:
        kind: A key of `KINDS`. Positional-only, so an artifact payload — which carries
            its own `kind` key — can be splatted straight into `make("artifact", **art)`.
        **data: The payload. Every key `KINDS[kind]` lists must be present; extra keys
            (a sub-agent's `sub_agent_ctx`, a turn number) travel along untouched.

    Returns:
        `{"event": kind, "data": data}` — the shape every interface reads.

    Raises:
        KeyError: For a kind the contract does not know; add it to `KINDS` first.
        ValueError: For a payload missing a required key.
    """
    try:
        required = KINDS[kind]
    except KeyError:
        raise KeyError(f"{kind!r} is not an event kind; add it to core/events.py first") from None
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{kind} event is missing {missing}")
    return {"event": kind, "data": data}


def artifact(kind: str, /, **data: Any) -> dict:
    """Build one artifact payload — the `data` of an `artifact` event — checked likewise.

    Args:
        kind: A key of `ARTIFACT_KINDS`.
        **data: At least the keys `ARTIFACT_KINDS[kind]` lists, plus `tool`.

    Returns:
        `{"kind": kind, **data}`, ready for `make("artifact", **payload)`.

    Raises:
        KeyError: For an artifact kind the contract does not know.
        ValueError: For a payload missing a required key.
    """
    try:
        required = ARTIFACT_KINDS[kind]
    except KeyError:
        raise KeyError(
            f"{kind!r} is not an artifact kind; add it to core/events.py first"
        ) from None
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{kind} artifact is missing {missing}")
    return {"kind": kind, **data}
