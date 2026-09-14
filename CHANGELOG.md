# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project uses [semantic versioning](https://semver.org/). While the version stays below
1.0, the public API may change between minor releases.

## [Unreleased]

### Fixed

- **`HELIOAI_MAX_OUTPUT_TOKENS` now reaches the `opencode` provider.** The override
  iterated a hand-written list of providers that predated opencode, so the exact
  setting the "model returned neither text nor a tool call" error tells you to raise
  did nothing for the provider the README recommends. Every provider declared on
  `LLMConfig` is now covered, including ones added later.
- **`run_python` advertised a 30 s default timeout while its signature said 60 s.**
  The signature was raised in 0.2.0 for cold Python starts; the schema the model reads
  was not. The registration tests now check that every schema default, `required`
  list and public parameter agrees with the function it describes, so the two cannot
  drift apart again.
- **`%helioai_provider` did nothing.** It wrote `HELIOAI_LLM_PROVIDER` into the
  environment, which `settings` reads exactly once, at import — so the confirmation
  printed and the next cell kept the configured provider. The choice is now held by
  the magic and handed to the client factory on every cell; `%helioai_provider` with
  no argument shows the current one, and the accepted names come from the factory.
- **Two turns on one session could corrupt its transcript.** The session store hands
  every caller the same in-memory history, and nothing stopped two requests — two
  browser tabs, two MCP calls — from appending to it at once and persisting the
  interleaved result. A turn now holds a per-session lock from its first read to its
  last save; the web UI answers `409 Conflict` to a second request instead of leaving a
  tab that looks hung while it waits.
- **A session's manifest and provenance ledger could lose entries or be left
  half-written.** Both are read-modify-write JSON files with no lock, so two
  downloads persisted at once kept only one of them; and both were rewritten in place,
  so a crash mid-write left an unreadable session. Writers now hold a per-directory
  lock and swap a complete file in atomically — the pattern the HELIO4CAST cache
  already used.
- **`HELIOAI_DATA_DIR` now relocates everything.** It moved the session store and the
  per-user homes, but the index, the saved catalogues and the profile kept deriving
  from the default directory computed before the variable was read — so the Docker
  volume held two trees. The dead `HELIOAI_WORKSPACE` / `workspace_dir` setting, read
  and consumed by nothing since storage became per-user, is gone.
- **A sub-agent's measured values never reached the screen.** The `sub_agent_end`
  event the lead re-emits carried the prose summary but dropped `findings` — the table
  of values the run actually computed, the one part of the report with an origin. The
  CLI, the notebook and the browser now list them under the sub-agent's line.
- **The automated "these ids are not in the catalogue" correction replayed as a
  question the user had asked.** The loop injects it as a `user` message because that
  is the only role the provider clients forward from history; the persisted copy now
  carries `origin="correction"`, the web replay shows it as a system note, and the
  exported notebook writes it as HelioAI's note rather than under **You:**. What the
  model sees is byte-for-byte unchanged.
- **The two catalog tools now state, and share, one window rule.** Both select events
  by where they *begin* — the convention for a superposed-epoch analysis and for "every
  ICME of 2015" — but `get_events_timeseries` described its window as "events in this
  window", which reads as overlap, and each tool carried its own copy of the filter.
  One helper, one wording, and tests on the edge events.
- `import helioai` no longer creates directories: the session store now creates its
  database and schema on first use rather than at import.
- `httpx2` is declared as a dependency. `tools/mcp_client.py` imports it directly (the
  MCP SDK's streamable-HTTP client is typed on it) but only received it as a transitive
  dependency of `mcp`.
- **Documentation drift, swept and then pinned.** `.env.example` gains
  `HELIOAI_CATALOGS_DIR`, `HELIOAI_OLLAMA_HEADERS` and `HELIOAI_LOG_LEVEL`, which the code
  read and nothing documented; a test now fails when any environment variable read
  under `helioai/` is missing from it — or listed there and read nowhere. The pull
  request template no longer recommends `uv run pytest`, which the contributing guide
  forbids; the recipes guide lists `fill_values`; the installation guide lists
  `opencode`; README and AGENTS.md stop quoting a test count that is stale the week
  after it is written.
- **Web hygiene.** Nominative tokens are compared in constant time (`hmac.compare_digest`,
  as the dev token and the MCP bearer already were); every response carries a
  `Content-Security-Policy` restricted to the server's own origin and
  `X-Content-Type-Options: nosniff`; the Host-header guard against DNS rebinding is
  built by `harden_for_host`, so the test client exercises what uvicorn serves; the
  browser console no longer receives every artifact's absolute server path.
- **The CLI parses its arguments with `argparse`.** `helioai --session` (value missing)
  crashed with an `IndexError`; `serve --web --port` likewise. Both are now argument
  errors, subcommand flags may come in any order, and `--session`/`--dev`/`--resume` may
  follow the question. `--help` still prints the module documentation and a quoted
  question containing the words is still a question.
- **`serve --web` on a non-loopback address with no users configured now refuses to
  start**, as `helioai-mcp --http` already did without a token: a bare
  `docker run -p 7890:7890` published an unauthenticated `run_python` on every host
  interface. `docker-compose.yml`, which publishes on loopback, carries the explicit
  opt-out (`HELIOAI_ALLOW_UNAUTHENTICATED_PUBLIC=1`) a container needs to bind `0.0.0.0`.
- **A download no longer freezes every other user.** The data tools are `async def`
  because the registry awaits them, but speasy, ChromaDB, the embedding model and
  `numpy.savez_compressed` are synchronous — called inline, a 60-second download stalled
  the event loop, and with it every stream on the web and MCP servers. The seven data
  tools now run their bodies in a worker thread (`asyncio.to_thread`, which carries the
  session context with it), at most four speasy downloads at a time; the figure review
  encodes its PNGs there too.
- **The tool calls of one turn overlap.** The prompt asks the model to batch its
  downloads in a single turn; both loops then ran them one after another, so a
  data_analyst's first turn took the sum of three or four downloads. Registry tools are
  now started together and their results consumed in the model's order, so every event
  and every `tool` message keeps the sequence it had. `run_python` stays sequential (it
  numbers its scripts from the disk), as do `task` and the internal tools.

### Added

- **The eleven shipped recipes have tests.** `tests/recipes/` rebuilds the sandbox
  namespace from the same helper source the notebook export ships, runs every recipe
  (its placeholders, demos and own `assert`s included), and checks each method on a
  synthetic input whose answer is known: a constructed shock normal for `theta_bn`, a
  cloud of known variances for `mvab`, a pure rotational discontinuity for the Walén
  slope, mass conservation for the Rankine–Hugoniot speed, scaled copies for the
  superposed-epoch median, a planar front for two-spacecraft timing. Until now a
  regression in a recipe was invisible before a demo. Marked `recipes`; skip them with
  `-m "not recipes"`.
- **`helioai doctor`** answers the questions every support request starts with: which
  `.env` was read, is the provider key there, is the index built and how old is it, is
  `run_python` really sandboxed or on the fallback path (and why), how big the speasy
  inventory and the workspaces have grown. Offline by default, `--online` probes the
  provider once, `--json` for bug reports and CI; exit code 1 when a check fails.
- **The event contract lives in one place.** `core/events.py` lists every kind the
  agent loops emit with its payload keys and every artifact kind; a static test holds
  the emitters, the CLI, the notebook magic and the browser to that list, so a kind
  added to a loop and forgotten in one interface — or documented and never emitted —
  fails in CI. Three docstrings used to carry their own, disagreeing copies.
- **Token usage is kept.** Every provider reports what a call cost and every count was
  dropped at save time, so nothing could say what a session — or a user — had spent.
  Each lead call is now a row in a `usage` table (turn, agent, provider, prompt /
  completion / cached tokens); a sub-agent reports its total on `sub_agent_end` and is
  charged to the parent session under its role. `helioai history` shows a tokens column,
  `GET /api/me` returns the caller's totals for the day, the month and all time — the
  number a per-user quota compares against.

### Removed

- The cross-encoder reranking stage of parameter search — measured to degrade results
  and disabled for a year (`RAGConfig` keeps the measurement in its docstring); the
  `rerank_*` settings go with it. Also gone: `run_subagent` (no callers) and the
  `data_preview` artifact renderers in the three interfaces (no emitter).

### Changed

- **If you set `HELIOAI_DATA_DIR` (the Docker image does), run `helioai migrate-storage`
  once after upgrading.** It moves the index, the catalogues and the profile from the
  default directory to the configured one, never overwrites, and can be re-run. The
  `search_parameters` error names the legacy copy when it exists, so an upgraded
  install is not sent into an hour-long rebuild.
- The session database gains `messages.origin` and `messages.name` columns, added
  automatically the first time an existing database is opened. `name` records which
  tool produced a `tool` message; readers no longer have to recognise a tool by the
  shape of its JSON (a loaded recipe was identified by having `name`, `code` and
  `metadata` keys at once).

## [0.2.1] — 2026-08-14

Documentation only — no behaviour change. Everything here was already true of the code in
0.2.0; what shipped was the description of it.

### Fixed

- **The `solar_mach` recipe told users to install a package that no longer exists.** Its
  runtime error message read `pip install "helioai[solarmach]"`, a name PyPI stopped
  serving when the distribution was renamed to `helioai-agent` — so a user who hit the
  missing-extra path was handed a command that fails.
- **README claims had drifted from the code**: the test count and coverage (627/77% →
  797/80%), the CI matrix (3.11/3.12 → 3.12/3.13/3.14), the skill count (5 → 6), the recipe
  count (9 → 10), and a "PyPI release" roadmap box left unchecked on a published package.
  The README is what PyPI renders as the project page, so these were the first thing a
  visitor read.
- **The quickstart pointed at the wrong event**: it advertised the 2003 Halloween storm for
  a notebook that has covered the 2015 St. Patrick's Day storm since the coverage of the
  Halloween window was measured unusable.
- **CONTRIBUTING and the developer docs recommended `uv run pytest`**, which re-syncs the
  environment and pulls the multi-gigabyte CUDA build of torch behind
  `sentence-transformers`. They now call the interpreter in `.venv/` directly.

### Added

- **Docstring examples across the public API.** The 19 agent-facing entry points (plasma
  tools, parameter search and download, catalogs, recipes, literature, sandbox, notebook
  export, the `%%helioai` magic) carry `Example:` blocks whose outputs were captured from
  real executions rather than written by hand. The core surface — `stream_chat` (including
  its 14 event kinds), `chat`, `build_index`, `build_llm_client`, `SessionStore`, the
  datastore and workspace helpers, `rag.search`/`search_batch`, the MCP server entry
  points, `to_standalone` and the boundary models — gained argument and return
  documentation it never had.
- **Issue #1** tracks migrating `mcp_server.py` to mcp 2.x, which the `mcp>=1.0,<2` cap in
  `pyproject.toml` had claimed was "tracked separately" without anything actually tracking
  it.

## [0.2.0] — 2026-08-14

First published release. `0.1.0` was never released to PyPI, so this is the first version
anyone can install.

**Install it with `pip install helioai-agent`, then `import helioai`.** PyPI rejects
`helioai` as confusable with the existing `helloai` — its check folds `l` and `i` to `1`,
so both names reduce to the same string. The distribution name is the only thing that
changed; the import package, the CLI commands and the API are all unchanged.

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

- **`pip install` produced an unusable install.** The wheel never contained
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
  directory, so following the README (`pip install helioai-agent`, then copy `.env.example`)
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
- Test coverage is now measured across every module with no exclusion list — 796 tests,
  80%. The previous 60% floor sat *below* the project's real coverage, because the
  excluded set included modules at 100% while genuinely thin ones were never on it.
- The Docker image's bubblewrap support has not been exercised in a running container:
  a container policy can still deny the user namespace. HelioAI tests bubblewrap
  functionally before using it and logs `sandbox_not_isolated` when it falls back, so
  the logs answer the question on any host.

[Unreleased]: https://github.com/erdoganfurkan/HelioAI/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/erdoganfurkan/HelioAI/releases/tag/v0.2.0
