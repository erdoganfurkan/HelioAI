"""The HelioAI runtime: one agent loop, driven by policies.

`stream_chat` (the lead) and `stream_subagent` (a delegated role) grew as two copies of
the same loop — call the model, start the tool calls, review the figures, emit the
events, append the results — and drifted the way two copies do: the sub-agent truncated
nothing, then truncated differently; a guard-rail added to one was missing from the other.
`runner.Runner` is the loop, written once; `policies.Policy` is what makes it the lead or
a role. The two public generators are wrappers that build a policy and finish the run
their own way.

Nothing an interface consumes changes: the runner yields the same `{"event", "data"}`
dicts, in the same order, that the two loops yielded — the event-journal golden and the
loop tests are the proof.
"""
