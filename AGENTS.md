# AGENTS.md — HelioAI

An open source scientific AI agent for plasma and heliophysics data analysis, built on
speasy (70+ missions, 83k indexed products), PlasmaPy and ChromaDB.

You work on parts of this repository that can be **proven correct offline**: tests,
documentation, typing, mechanical refactors, dependency hygiene, and bug fixes that ship with
a reproducing test. You cannot validate agent behaviour here — that needs API keys, a 342 MB
search index and live data downloads, none of which exist in your environment. Do not claim a
change works because it looks right.

## How to verify your change

```bash
uv sync --extra dev
.venv/bin/python -m pytest        # 818 tests, coverage floor 70%
.venv/bin/ruff format --check .
.venv/bin/ruff check .
```

- **Never use `uv run pytest`.** It resynchronises the venv and pulls ~4 GB of CUDA wheels.
- `uv sync --extra X` **purges** extras it is not given. Always accumulate:
  `uv sync --extra dev --extra solarmach`.
- CI runs `ruff format --check`, so run `ruff format` **and** `ruff check` before committing.
- Coverage has **no `omit` list** and a 70% floor. Do not add exclusions to raise the number.

If your change is not covered by the test suite, say so plainly in the pull request instead of
asserting it works.

## Off limits

Changes here cannot be verified offline, so do not make them:

- `helioai/core/skills/` — these are **prompts**, not code. They are excluded from ruff on
  purpose (ruff reformats Python inside the markdown). Never reformat them.
- `helioai/data/recipes/` — also excluded from ruff, for the same reason.
- `helioai/core/llm/` — provider clients. A change here is only provable against a live
  provider.
- `helioai/tools/sandbox.py` — the bubblewrap isolation is the only real security boundary in
  the project. Do not "simplify" it.
- `data/` — real runtime data, and it is gitignored, so anything removed there is
  unrecoverable. Never delete under it.
- Prompt text anywhere. Rewording a prompt changes agent behaviour you cannot measure.

## Architecture

```
helioai/
├── config.py               settings singleton, loaded once; validates the selected provider's key
├── datastore.py            npz + manifest.json per session — the basis of reproducible export
├── export.py               session -> standalone .ipynb
├── indexer.py              speasy catalogue -> ChromaDB
├── mcp_server.py           MCP stdio + streamable HTTP server
├── core/
│   ├── agent_loop.py       stream_chat — the main agent loop
│   ├── sub_agents.py       stream_subagent — delegation + tool whitelist
│   ├── tool_exec.py        shared by BOTH loops — put shared logic here, never duplicate it
│   ├── session.py          per-(user_id, session_id) SQLite history
│   ├── skills/             6 SKILL.md prompt files
│   └── llm/                base, openai_compat (groq/ollama/azure/opencode), azure, gemini, factory
├── tools/
│   ├── registry.py         ToolRegistry — JSON dispatch to async functions
│   ├── setup.py            registers the 17 tools on import — without it the registry is EMPTY
│   ├── rag.py              hybrid BM25 + dense, fused by RRF
│   └── ...                 speasy, catalogs, plasmapy, sandbox, recipes, literature, mcp_client
└── interfaces/             cli.py, jupyter_magic.py, web/ (FastAPI + SSE)
```

## Conventions

- **Every tool must be `async`** — the registry always `await`s — and its JSON schema must
  match its signature. `tests/test_tool_registration.py` enforces this.
- **No explanatory comments.** The code should be self-documenting. The exception is
  **public docstrings, which are mandatory**, and which must document the *decision* behind
  the code rather than paraphrase the signature.
- Simplicity first: touch as little code as possible. No premature abstraction.
- Never assume — verify a path, an API or a variable before using it.

## Dependencies

Python 3.12 / 3.13 / 3.14, core packages on a SPEC-0 24-month window, checked in CI by the
official `heliophysicsPy/pyhc-actions/phep3-compliance` action.

- Floors with `>=`. **No upper bounds** unless absolutely necessary, and then only with a
  tracking issue. There are currently none, and adding one needs its reason in the diff.
- `[tool.uv] constraint-dependencies = ["torch>=2.6"]` exists only so the lock resolves a
  torch with cp313/cp314 wheels. It is **not** a direct dependency.
- Lifting an upper bound **relocks the entire graph**. If you do it, read the full `uv lock`
  diff and say what moved.

The lint `select` is pinned explicitly in `pyproject.toml` (`E4,E7,E9,F,I,UP,B`): inheriting
ruff's defaults broke CI when a release widened them. Keep it pinned.

## Documentation

MkDocs Material + mkdocstrings, built with `--strict` — a single dead link fails the build.

```bash
uv sync --extra docs && .venv/bin/python -m mkdocs serve
```

## Commits

- Never add `Co-Authored-By` trailers.
- Never modify `.gitignore` without explaining each line you touched.
- One logical change per pull request, with the verification output pasted in.
