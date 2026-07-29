# Reproducible export

Any session exports to a self-contained `.ipynb` whose cells run in a plain Jupyter kernel
— with no HelioAI installed, no agent, and no sandbox.

```bash
helioai export           # most recent session
helioai export a3f9      # by session id prefix
```

```python
%helioai_export          # in Jupyter
```

## What gets rewritten, and why

The code the agent runs is not the code you want to keep. Inside the sandbox it uses
helpers that only exist there, so a raw dump would be a notebook that cannot run. The
export rewrites the boundary:

| Sandbox | Exported |
|---|---|
| `load_data('bz')` | `spz.get_data('amda/imf_bz', '2005-01-16', '2005-01-18')` |
| `load_data('bz_events')` | `spz.get_data(id, [[s1,e1], [s2,e2], ...])` over the OK events |
| `export('name', arr)` | `print(...)` of the same summary |
| `clean(arr)` | a real fill-value mask, inlined |
| `param_card(...)`, `document_method(...)` | stripped — agent-only UI helpers |

The rewrite is possible because `datastore.py` records the `param_id`, `start` and `stop`
behind every dataset key in a manifest, so a `load_data` call can be turned back into the
`speasy` call that produced it. Imports are added at the top, and anything the export
cannot resolve is left untouched rather than guessed at.

This came out of the first external demo. The reviewer's objection was blunt and correct:
the code shown was not code he could take away and re-run. Rewriting the boundary was the
answer.

## What the notebook contains

1. **Setup** — imports and any shims still required.
2. **One cell per analysis step**, in order, as standalone code.
3. **Methods & data acknowledgements** — every recipe and reference used, assembled by
   scanning the session's tool calls, plus the data-provider acknowledgements.

## Verifying it really runs

The claim is only worth something if it is checked. The repository ships
`verify_export.sh`, which copies a notebook into a temporary directory **without** the
`data/` tree and executes it with `nbconvert`:

```bash
uv pip install nbconvert ipykernel
./verify_export.sh path/to/session.ipynb
```

Running it outside the repo is the whole point — it proves the notebook depends on
`speasy` and public data, not on your local workspace.

## Getting the code without leaving the session

The web UI's code panel shows the same standalone rewrite for each step, and the `/code`
endpoint returns it directly. You do not have to export the whole session just to copy one
analysis.

## Why this matters beyond convenience

An agent that produces a number and a plot is a black box you have to trust. An agent that
produces the script, the method citation and a runnable notebook is a tool a reviewer can
check. That is the difference this project is built around — see
[Recipes and provenance](recipes.md) for the other half.
