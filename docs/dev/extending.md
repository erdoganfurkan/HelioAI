# Extending HelioAI

Four things you are likely to add. Each is deliberately small.

## Add a tool

Write an `async` function and register it in `helioai/tools/setup.py`:

```python
from helioai.tools.registry import registry

@registry.register(
    name="magnetopause_distance",
    description="Standoff distance of the magnetopause from the Shue et al. (1997) model.",
    parameters={
        "type": "object",
        "properties": {
            "pdyn_nPa": {"type": "number", "description": "Solar wind dynamic pressure"},
            "bz_nT": {"type": "number", "description": "IMF Bz in GSM"},
        },
        "required": ["pdyn_nPa", "bz_nT"],
    },
)
async def magnetopause_distance(pdyn_nPa: float, bz_nT: float) -> dict:
    """Return the subsolar standoff distance in Earth radii."""
    ...
```

Rules that are enforced by tests:

- **It must be `async`.** `call_tool` always awaits; a sync function fails at dispatch.
- **The schema must match the signature.** `tests/test_tool_registration.py` compares every
  declared property against `inspect.signature`. A schema advertising a parameter the
  function cannot accept is invisible until the model sends it and the call dies.
- **The description is the model's only clue.** Describe *when* to use it, not just what it
  computes.
- **Return a dict**, not a formatted string. Formatting is the interface's job.

Errors should be returned, not raised: `{"error": "..."}` keeps the conversation alive,
while an exception aborts the turn.

## Add a recipe

Drop a `.py` file in `helioai/data/recipes/` with a header, including the mandatory
`# reference:` line. See [Recipes and provenance](../guide/recipes.md). Recipes are exec'd
against an injected namespace, so they are excluded from linting.

To make the agent actually reach for it, add it to the task→recipe table in
`helioai/core/skills/data_analyst/SKILL.md` — `data_analyst` has no `list_recipes` tool, so
it only knows the recipes it is told about.

## Add an LLM provider

If it speaks the OpenAI chat-completions format — most do — it is a table entry, not a
class. In `helioai/core/llm/factory.py`:

```python
OPENAI_COMPAT = {
    ...
    "openrouter": {
        "config": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
    },
}
```

Add the matching dataclass in `config.py` (model, `max_output_tokens`, `temperature`,
`api_key`), wire it in `_load()`, and add the env var to `.env.example` — a setting with no
`.env.example` entry is a setting nobody will find.

Only write a client class if the wire format genuinely differs, as Gemini's does.

## Add a sub-agent role

In `helioai/core/sub_agents.py`, add a `SubAgentRole`:

```python
"event_detector": SubAgentRole(
    name="event_detector",
    description="Detect shocks, reconnection events and CMEs in a time series.",
    system_addon="...",
    allowed_tools=["get_timeseries", "run_python"],
    auto_load_skills=["data_analyst"],
    max_turns=8,
),
```

The whitelist is a real boundary, not a hint — a role calling outside its set gets an error
naming what it may use, and the tool is never dispatched. Keep `max_turns` tight: a role
that needs more than a handful of turns is usually two roles.

## Before you open a PR

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

Every behaviour change needs a test, and tests use real objects where practical — a real
temporary SQLite database rather than a mocked `SessionStore`, a real ChromaDB
`PersistentClient` on `tmp_path` rather than the shared in-memory one. Mocks that drift
from the real component have caused production bugs here before.

See [CONTRIBUTING.md](https://github.com/erdoganfurkan/HelioAI/blob/main/CONTRIBUTING.md)
for what will and will not be accepted.
