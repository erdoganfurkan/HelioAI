## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## Why

<!-- The problem it solves. For a scientific method, cite the paper. -->

## Checklist

- [ ] `.venv/bin/ruff format .` and `.venv/bin/ruff check .` both pass
- [ ] `.venv/bin/python -m pytest` passes (not `uv run pytest`, which re-syncs the venv)
- [ ] Behaviour changes are covered by a test
- [ ] No new runtime dependency (or it was agreed in an issue first)
- [ ] Docs updated if the change is user-visible
