# Contributing to HelioAI

Contributions are welcome — bug reports, fixes, new scientific recipes, and
documentation all help. This page covers how to get set up, what the project
expects from a change, and, just as importantly, **what will not be accepted**
so nobody spends a weekend on a patch that was never going to land.

## Getting set up

HelioAI uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/erdoganfurkan/HelioAI.git
cd HelioAI
uv sync --extra dev
```

Copy `.env.example` to `.env` and set at least one LLM provider key. Most tests
run without one; the ones that need a live model are skipped automatically.

Building the parameter index is a one-time ~10 minute job and is **not** needed
for the test suite:

```bash
uv run helioai index
```

## Running the tests

```bash
.venv/bin/python -m pytest                      # full suite
.venv/bin/python -m pytest tests/test_rag.py -v # one module
```

Call the interpreter in `.venv/` directly rather than `uv run pytest`: `uv run`
re-syncs the environment first, which pulls the multi-gigabyte CUDA build of
torch behind `sentence-transformers`.

Note that `uv sync --extra X` *purges* extras that are not listed, so always
combine them: `uv sync --extra dev --extra solarmach`. Otherwise pytest
disappears from the environment.

## Before you open a pull request

Run both of these. CI runs `ruff format --check`, so a correctly linted but
unformatted branch still fails:

```bash
uv run ruff format .
uv run ruff check .
```

Optionally install the hooks so this happens automatically:

```bash
pre-commit install
```

Every change to behaviour needs a test. Tests use real objects wherever
practical — a real (temporary) SQLite database rather than a mocked
`SessionStore`, a real ChromaDB `PersistentClient` on `tmp_path` rather than the
shared in-memory client. Mocks that drift from the real component have burned
this project before.

## Dependency version policy

HelioAI follows [PHEP 3](https://doi.org/10.5281/zenodo.17794207), the PyHC
adoption of [SPEC 0](https://scientific-python.org/specs/spec-0000/):

* Python minor versions are supported for **36 months** after release.
* Upstream core Scientific Python packages (`numpy`, `scipy`, `matplotlib`,
  `pandas`, `scikit-image`, `networkx`, `scikit-learn`, `xarray`, `ipython`,
  `zarr`) are supported for **24 months** after release.
* New versions of both are adopted within **6 months** of release.

Practically: use `>=` floors. Upper bounds (`numpy<2`) and exact pins
(`scipy==1.10`) are only acceptable when something is genuinely broken, and a
tracking issue must be opened at the same time to remove them. CI enforces this
with the PyHC PHEP 3 compliance action.

## What will not be accepted

* **Anything outside heliophysics and space plasma physics.** HelioAI is a
  domain tool, not a general-purpose data agent.
* **Interactive visual exploration features** — multi-panel exploratory
  plotting, manual event labelling, high-performance interactive rendering.
  That is [SciQLop](https://github.com/SciQLop/SciQLop)'s job and it does it far
  better. HelioAI interoperates with it (catalogs are written in the standard
  `speasy` format) rather than competing. HelioAI's value starts at a sentence
  that triggers an analysis; if a feature's value requires the user to look and
  manipulate, it belongs upstream.
* **New runtime dependencies added without discussion.** PyHC standard 10 asks
  projects to import the minimum necessary, and every runtime dependency is
  weight for every user. Open an issue first. Optional features belong in an
  extra (see `solarmach`).
* **Vendored copies of code from other projects**, and any dependency under a
  copyleft licence. HelioAI is MIT and intends to stay usable by everyone.
  Scientific methods reimplemented from a published paper are fine and welcome —
  cite the paper in the recipe's `# reference:` header.
* **Changes with no test**, except pure documentation edits.

## Adding a scientific recipe

Recipes live in `helioai/data/recipes/` and are plain Python modules loaded on
demand by the agent. They sit inside the package so they ship with the wheel.
Each one **must** carry a `# reference:` line naming the paper the method comes
from — provenance is a core promise of this project, not a nicety. Look at
`helioai/data/recipes/mvab.py` for the shape.

Recipes are exec'd in the sandbox with an injected namespace (`export`, the
loaded parameters), so they legitimately reference undefined names and are
excluded from linting.

## Reporting bugs

Open an issue with the HelioAI version, your Python version and OS, the LLM
provider, and the exact prompt or call that failed. If the agent produced code,
include it — the generated script is saved in your workspace directory.

For security issues, do **not** open a public issue; see [SECURITY.md](SECURITY.md).
