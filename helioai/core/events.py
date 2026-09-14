"""The event contract between the agent loops and every interface, in one place.

`stream_chat` and `stream_subagent` yield `{"event": kind, "data": {...}}` dicts. The
kinds were listed in three docstrings that disagreed with each other and with the code
(`artifact`, `figure_review`, `invalid_ids` and `recipe_bypassed` were missing from the
module docstring of `agent_loop`; `sub_agents` claimed to yield the same set as the lead
and did not). This module is the list the code and the tests are held to:
`tests/test_events_contract.py` greps the emitters and the three renderers and fails
when a kind appears in one and not the other.

Payloads stay dicts for now — every interface consumes them as JSON — so this is a
contract, not a type system. E1 of the 0.3.0 plan turns each kind into a dataclass
whose `to_dict()` is exactly today's shape.
"""

from __future__ import annotations

# kind → the payload keys every emitter provides (extra keys may travel along;
# `sub_agent_ctx` is added to any kind a sub-agent emits).
KINDS: dict[str, tuple[str, ...]] = {
    "reply": ("text",),
    "tool_call": ("turn", "name", "arguments", "display"),
    "tool_result": ("turn", "name", "summary", "display"),
    "artifact": ("kind", "tool"),
    "skill_loaded": ("name",),
    "plan": ("title", "steps"),
    "figure_review": ("turn", "text"),
    "provenance": ("matched", "contradicted", "derived", "unsourced", "details"),
    "invalid_ids": ("ids",),
    "recipe_bypassed": ("recipes",),
    "sub_agent_start": ("task_id", "role", "description"),
    "sub_agent_end": ("task_id", "role", "summary", "n_iterations", "error", "findings"),
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
    {"reply", "plan", "provenance", "sub_agent_start", "error", "done"}
)

# Artifact kinds an interface may leave unrendered on purpose. `exports` is recorded
# in the provenance ledger and shown through `findings`, not as a card of its own.
RENDER_OPTIONAL: frozenset[str] = frozenset({"exports"})
