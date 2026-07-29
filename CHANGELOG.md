# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project uses [semantic versioning](https://semver.org/). While the version stays below
1.0, the public API may change between minor releases.

## [Unreleased]

## [0.2.0] — 2026-07-29

First published release. `0.1.0` was never released to PyPI, so this is the first version
anyone can install.

### Added

- **Documentation site** built with MkDocs Material and mkdocstrings, deployed to GitHub
  Pages: installation, quickstart, four user guides, three developer guides, and an API
  reference covering 285 objects. Built with `--strict` in CI, so a broken cross-reference
  fails the build.
- **Community files** — `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1),
  `CONTRIBUTING.md` including what will *not* be accepted, `SECURITY.md`, and issue and
  pull-request templates.
- **`SECURITY.md` documents the sandbox threat model** for the first time: `run_python`
  executes model-written code, and bubblewrap isolation is Linux-only — on macOS and
  Windows it degrades to a plain subprocess.
- **Ollama support**, which the README had advertised while the client raised
  `NotImplementedError` and the factory refused the name. Ollama serves an
  OpenAI-compatible API, so it needed no client of its own.
- **Two runnable example notebooks** in `examples/`, output-free.
- **PHEP 3 compliance check** in CI, via the official PyHC action.

### Changed

- **Python 3.12, 3.13 and 3.14** are now supported and tested; 3.11 is dropped. This
  follows [PHEP 3](https://doi.org/10.5281/zenodo.17794207)'s 36-month window, and
  `plasmapy` already required `>=3.12`.
- **One LLM client instead of four.** Groq, Ollama and Azure all speak the OpenAI
  chat-completions format and now share `OpenAICompatClient`; a provider is a `base_url`
  table entry rather than a class. Gemini keeps a native client because its wire format
  genuinely differs.
- Dependency floors raised to the PHEP 3 window: `numpy>=2.1`, `matplotlib>=3.10`,
  `scipy>=1.15`, `ipython>=8.27`.
- Ruff's rule selection is pinned explicitly rather than inherited from its defaults.
- The version is single-sourced from `helioai/__init__.py`.

### Fixed

- **`pip install helioai` produced an unusable install.** The wheel never contained
  `data/recipes/`, so an installed copy had zero recipes, and every storage path resolved
  under `site-packages/`, so the agent tried to write ChromaDB, the session database and
  user workspaces inside the installed package. Recipes now ship inside the package and
  user data goes to `$XDG_DATA_HOME/helioai`; running from a clone is unchanged.
- A closure in `mcp_client.discover_and_register` captured the loop variable instead of
  binding it, so every server after the first would have been discovered against the wrong
  specification had the coroutine ever been deferred.
- The Jupyter demo notebook called the PlasmaPy tools without `await`, returning coroutine
  objects.

### Removed

- The `groq` SDK, now reached through the shared OpenAI-compatible client.

### Security

- The sandbox's platform-dependent isolation is documented rather than implicit.

### Notes

- `mcp` is capped below 2.0: that release is a breaking rewrite of both the client and
  server APIs HelioAI uses. Lifting the cap is tracked with the `mcp_server.py` migration.
- Test coverage is now measured across every module with no exclusion list — 627 tests,
  77%. The previous 60% floor sat *below* the project's real coverage, because the
  excluded set included modules at 100% while genuinely thin ones were never on it.

[Unreleased]: https://github.com/erdoganfurkan/HelioAI/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/erdoganfurkan/HelioAI/releases/tag/v0.2.0
