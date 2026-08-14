# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project uses [semantic versioning](https://semver.org/). While the version stays below
1.0, the public API may change between minor releases.

## [Unreleased]

## [0.2.0] — 2026-08-14

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
- **A provenance ledger.** Every value a run actually computes is recorded with its
  units, and the final answer marks which of its numbers were measured rather than
  asserted. Sub-agents report what they measured, not only what they claim, and a
  contradiction check flags prose that disagrees with the ledger. This is the feature
  that separates a plausible answer from a checkable one.
- **Temporal coverage in the search index.** `search_parameters` takes the window under
  study and demotes products that do not cover it, instead of letting the agent discover
  each gap one download at a time — four turns out of eight went that way in a measured
  run.
- **`magnitude()` and `save_path()` in the sandbox.** A hand-written `sqrt(bx²+by²+bz²)`
  silently consumed data gaps; and generated code had no documented way to name its own
  writable directory, so a run could report writing a file that existed nowhere once the
  sandbox exited.
- **Two recipes**: `shock_timing_2sc`, which refuses to infer a shock normal from Δr/Δt
  alone rather than returning a circular one, and `fill_values`.
- **A recipe-bypass detector**: loading a recipe is not using it, and the two were
  indistinguishable in the logs.
- **`opencode` provider** (Zen Go gateway), OpenAI-compatible, reusing the shared client.
- **Parameter ids are verified against the index** before they reach an answer. The
  retrieval was already correct; the model rewrote correct ids into plausible fictions,
  once by grafting two real datasets together. Prompting alone did not fix it.

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
- **speasy is imported on first use inside the sandbox, not at every spawn.** Importing
  it refreshes an inventory over the network, so running `print("hello")` depended on
  CDAWeb being reachable. The full test suite went from 244s to 105s.
- **Declared fill values are blanked at the source**, once, rather than at each reader —
  a single surviving sentinel turned a 510 km/s mean into 3987.
- `rankine_hugoniot` owns its averaging windows instead of leaving them to be chosen by
  hand at each call site.
- The delegation decision no longer depends on sampling: three identical runs previously
  launched three, three and zero sub-agents.
- Example notebook 02 moved to the 2015 St Patrick's Day storm, whose coverage was
  measured rather than assumed, with reference values checked against the recipes.
- The answer checks run on the lead loop as well as on sub-agents. Installed in one loop
  out of two, they were silent on half of all executions — and that half looked healthy.

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
- **`.env` was ignored once pip-installed** — it was read relative to the package
  directory, so following the README (`pip install helioai`, then copy `.env.example`)
  raised `AZURE_OPENAI_API_KEY is not set` no matter what the file contained.
- **`helioai profile` and `%helioai_profile` edited a file nothing reads.** Namespacing
  storage per user split the path the commands wrote from the one the agent loads, and
  the web UI became the only interface where the documented feature worked.
- **Reasoning models**, three distinct failures: a turn returning neither text nor tool
  call was treated as fatal and abandoned the request, inline `<think>` reasoning leaked
  into replies, and an output budget shared between hidden reasoning and tool arguments
  truncated large `run_python` calls mid-JSON — which surfaced as the model being blamed
  for malformed calls.
- **Windows**, three ways: repository files were read with the OS default encoding rather
  than UTF-8, every workspace path was rejected because the containment check assumed a
  POSIX separator (so figures 404'd), and the sandbox environment allowlist was
  POSIX-only, which broke Winsock initialisation in every sandbox test.
- Rate limiting honours `Retry-After` instead of waiting out a window that never applied.
- Jupyter reused an async client across event loops, so a second cell hit a closed loop.
- The reported cadence was measured on the whole time grid, fill rows included, and after
  the preview was downsampled — announcing 8 ms for protons sampled every 3.08 s, and
  describing the preview rather than the data the agent would actually load.
- The speasy inventory is seeded into each sandbox session instead of being rebuilt,
  which had been exceeding the timeout and killing runs mid-download.
- A failed sandbox run is readable: line numbers are remapped past the preamble, the real
  exception replaces "exited with code 1", and the traceback stays in context.
- `plan`, `figure_review` and `invalid_ids` were emitted and rendered only by the web UI;
  the CLI and Jupyter dropped them silently.
- A repeated download is answered from the session manifest, and a failure says what
  failed.

### Removed

- The `groq` SDK, now reached through the shared OpenAI-compatible client.
- The `httpx2` development dependency, which nothing imported — `httpx` is a direct
  runtime dependency and the test client always had it.
- `indexer.py` at the repository root, a five-line shim for `helioai index`.

### Security

- The sandbox's platform-dependent isolation is documented rather than implicit.
- **A client-supplied `session_id` reached the filesystem unvalidated.** It became a path
  component in the workspace label, the export filename and the `rmtree` behind
  `DELETE /api/sessions/{id}`; the message slug was sanitised but the id pasted after it
  was not. Ids are now reduced to `[A-Za-z0-9_-]` where they become paths, the delete
  carries its own containment check because the label is persisted data, and the web
  request rejects a malformed id outright.
- **A loopback bind is not a boundary**: `serve --web` pins the `Host` header, since any
  web page can point a hostname at `127.0.0.1` and CORS does not cover it.
- **`helioai-mcp --http` publishes every tool, `run_python` included, unauthenticated.**
  That is the normal contract over stdio and remote code execution over HTTP; a
  non-loopback bind now warns, and SECURITY.md says so.
- **The sandbox says when it is not isolating anything.** The privilege drop runs in the
  forked child, where it cannot log and swallows its own failure — so a host without
  bubblewrap ran model-written code under the server's uid in silence.
- **The Docker image installs bubblewrap**, which SECURITY.md had promised it did while
  no such line existed; `HELIOAI_DATA_DIR` now points into the declared volume, which was
  inert (sessions and the index were written outside it and lost on every recreate); and
  compose publishes on loopback rather than on every interface of the host.

### Notes

- `mcp` is capped below 2.0: that release is a breaking rewrite of both the client and
  server APIs HelioAI uses. Lifting the cap is tracked with the `mcp_server.py` migration.
- Test coverage is now measured across every module with no exclusion list — 795 tests,
  80%. The previous 60% floor sat *below* the project's real coverage, because the
  excluded set included modules at 100% while genuinely thin ones were never on it.
- The Docker image's bubblewrap support has not been exercised in a running container:
  a container policy can still deny the user namespace. HelioAI tests bubblewrap
  functionally before using it and logs `sandbox_not_isolated` when it falls back, so
  the logs answer the question on any host.

[Unreleased]: https://github.com/erdoganfurkan/HelioAI/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/erdoganfurkan/HelioAI/releases/tag/v0.2.0
