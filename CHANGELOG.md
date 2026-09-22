# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project uses [semantic versioning](https://semver.org/). While the version stays below
1.0, the public API may change between minor releases.

## [Unreleased]

### Fixed

- **A search ranks the same in every process.** Chroma persists its HNSW graph only every
  `sync_threshold` writes — 1000 by default — and replays whatever followed the last persist
  into the in-memory graph at every start, in an order that varies. Measured on 2026-09-22
  with one query embedding in five processes: five different dense top-50 lists, `ssc/mms1`
  at rank 1 in four of them and absent from the fifth — the 325 SSCWeb trajectories added
  last were the replayed tail, and the catalogue collection had never been persisted at
  all. The indexer now opens both collections with a threshold of one (0.11 s per batch on
  the full index, nothing left to replay) and settles a collection built before the setting
  on open: modified, reopened, its tail persisted. `helioai doctor` warns on an index that
  still carries the old threshold and names the one command that settles it.
  `HelioBench/scripts/retrieval_replay.py`, which measures this, found the final fused
  ranking stable on the 30 n1 queries before the fix; the dense channel alone was not, and
  any change to the fusion would have carried its variance into the ranking the agent sees.
  The same opener raises the dense search beam (`ef_search` 100 → 400): the catalogue is
  full of near-twins — trajectories that differ by a spacecraft name, housekeeping variables
  that differ by a suffix — and a narrow approximate search loses the exact one. `ssc/mms1`,
  the true nearest neighbour of "MMS1 spacecraft position GSE 2019", was absent from the
  dense top-50 and is rank 1 now, for 0.7 → 1.2 ms per query; on the 30 HelioBench n1
  queries recall@1 moves from 53.3 % to 56.7 %. A `helioai index` run applies both to an
  existing index.
- **The parameter ranking uses what the index already knew.** Three signals of
  `_rerank_penalty` were dead or missing, each measured on the 30 HelioBench n1 queries
  replayed in five processes with `HelioBench/scripts/retrieval_replay.py` (zero tokens,
  ranks identical across processes throughout). The mission penalty matched `\bmms\b` and
  therefore never "mms1", "c1", "sta" or "vg1" — silent on every multi-spacecraft mission
  exactly when the user named the spacecraft: each mission now has a query pattern with its
  numbered forms (THEMIS-E's "the" left out), and the one MMS task moved from rank 13 to 3,
  `amda/c1_b_gsm` no longer heading an "MMS1 FGM" search (recall@3 86.7 → 90.0 %, MRR 0.727
  → 0.736). `measurement_type`, indexed on 12 850 products and consulted by nothing, now
  demotes a typed product that measures another quantity than the query names and leaves
  the 84 % without a type untouched; and a product whose coverage cannot overlap the
  download window is demoted before the cut, where it used to be merely flagged inside the
  top-k after it — `search` takes the window, the tool passes the one it had. Widening the
  candidate pool (`hybrid_fetch_k` 50 → 100 / 200, or the BM25 cap alone) was measured and
  rejected: MRR fell, two accepted products left the top-k. With the wider dense beam,
  "MMS1 spacecraft position GSE 2019" ranks `amda/mms1_xyz_gse` and `ssc/mms1` first and
  second, where a 2026 IMAP position led before.
- **`superposed_epoch` no longer refuses `nT (1min)` against `nT`.** CDAWeb labels some
  products with their cadence in parentheses after the unit; a live composite of Wind
  `BF1` events was refused three times as "incompatible units" until the caller dropped
  `units` altogether. Only a cadence-shaped annotation — whitespace, then a digit — is
  dropped before units are compared or converted; a parenthesised denominator such as
  `1/(cm2 s sr MeV)` or `W/(m^2 Hz)` is part of the unit and is kept, and so is `(nT)`.
- **`theta_bn` says what its uncertainty is, and lists shock candidates that are shocks.**
  Measured on the real Wind 92 ms field of 2004-11-07: at a fixed crossing and a fixed
  method, the averaging windows alone moved θ_Bn from 41.5° to 68.0° over 49 guard/span
  conventions, while the exported bootstrap read `± 0.19°` — it resamples rows inside two
  fixed windows and measures only that. The recipe now exports
  `theta_bn_window_spread_deg`, the half-range of θ_Bn over a grid of conventions around
  the one it uses (6.2° on that shock, 1.2° on 2015-03-17 04:00 UT, where the CfA shock
  database's own methods span 58.8–66.1°), prints the ensemble, and renames the bootstrap
  `theta_bn_sampling_std_deg` so neither can be read as the other. Windows chosen by the
  caller (`B_up`/`B_dn`) carry no spread and the recipe says so. `find_shock_candidates`
  ranked four ICME-sheath compressions above both real shocks of that day and cut the list
  at five, so the second shock (10:03 UT) was never shown: it now returns every |B| rise
  above a ratio of 1.2, up to ten, screens each against `density` and `speed` when they
  are bound (a fast forward shock steps in all three at once), lists the screened ones
  first, and reports the steepest single-sample rise as the time to pass back as
  `shock_time`. On the same data both real shocks now head the list, at 17:59:12 and
  10:03:43 UT.
- **Two sessions whose ids share their first six characters no longer share a workspace.**
  `make_session_label` keyed the directory on `session_id[:6]`; the web API lets a client
  choose its ids, and a benchmark that named its runs `bench-<question>-<hex>` gave every
  run of a question one directory, so from the second run on the sub-agent's inventory
  handed it the first run's downloads. The loop now passes the user's existing labels and
  the suffix grows until the label is new.
- **The recipe check no longer accuses a run that used the sandbox's Shue or Jelínek
  model.** `mp_shue1998` and `bs_jelinek2012` are published boundary models shipped with
  their reference; a run that called one and exported `magnetopause_r_at_mms_Re` was
  flagged for never loading `pressure_balance`, whose signature contains "magnetopause".
  Choosing another published model is not a hand-written copy of the recipe; a formula
  typed by hand for the same export still is.
- **The CLI no longer prints an answer twice when the model streams it, then delivers it
  through `final_answer` without its markdown.** The final `reply` was not a prefix of
  the streamed text (the bold was gone), so the whole answer was printed again; two
  renderings that say the same words are one answer.
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
- **The test suite no longer writes into the real `data/` directory.** `settings` is a
  singleton built at import, and any test that forgot to repoint it left sessions,
  workspaces and a 300 MB speasy seed under `data/users` of whoever ran `pytest` — found
  twice by `ls`, months apart. Every test now gets its own data directory and session
  database, and the run fails, naming the paths, if anything under the real one changed.
- **A cancelled `run_python` kills its sandbox.** Closing the browser tab cancels the
  stream and, with it, the tool call — but only a timeout used to kill the subprocess, so
  the bubblewrap tree kept running until its own 300 s ceiling, one orphan per abandoned
  question. Cancellation now kills the whole process group before propagating.
- **A session no longer costs 300 MB of disk before its first line of code runs.** The
  speasy inventory the sandbox needs was copied into *every* session directory (269 MB
  on the demo machine, 304 MB here, 76 sessions deep); it is now copied once per user, to
  `users/<user>/.speasy/`, and bind-mounted into each sandbox. A new session weighs
  200 KB. The seed sits beside the workspaces, never inside one, so an export never
  ships it. Expired workspaces are also swept hourly by the web server and when the
  notebook magic or the MCP server loads, not only at CLI startup.

### Added

- **A seam for a System One judge beside the loop — `helioai.core.judgment`, in
  observation.** HelioAI verifies itself thoroughly (claims against the ledger, ids against
  the index, code against the recipes, tools against the plan) and nothing verifies it
  against the question it was asked. Before any such check exists, this module fixes where
  it will live and what it may do: every question HelioAI puts to a judge is written here,
  in one file a reviewer can read in one sitting; the answer is typed (`Noul` → yes / no /
  abstain, `Choice` → one option of a closed set / abstain) and abstention is `None`, never
  a sentinel that could be summed into a plausible wrong answer. The default backend,
  `HELIOAI_JUDGMENT_BACKEND=null`, abstains on everything, so the loop is exactly the one
  that shipped before; `jev` asks TypeSafe's model through the optional `judgment` extra
  and `TYPESAFE_API_KEY`, and only at sites whose `judgment_<site>` experiment is named —
  two axes, so "the judge does not help here" and "the layer costs something" can be told
  apart. Every call is bounded (`HELIOAI_JUDGMENT_TIMEOUT_S`, default 2 s; the round trip
  measured from France is 266 ms median, 373 ms p95) and recorded as one JSON line under
  the session workspace, `judgment.jsonl`, with the state in full — a disagreement that
  cannot be adjudicated later is not a measurement. Nothing here corrects the model. An
  unknown backend is refused where the API key is checked and by `helioai doctor`, which
  gains a `judgment` line. No site asks yet; this is the seam the next entries plug into.
- **`helioai index --classify` fills the measurement type the archive left empty, and
  the indexed text says the dates a product covers.** `measurement_type` is indexed on
  15.6 % of the products (AMDA, CSA) and on none of CDA's 68 000, so every signal built on it
  reached a sixth of the catalogue. With `HELIOAI_JUDGMENT_BACKEND=jev`, the indexer asks the
  judge for the SPASE type of every untyped product before embedding — a `Choice` over the
  twelve types the index already uses, so a filled field is an exact filter. Measured on 200
  products the archive had labelled, label stripped before asking: 89 % agreement where the
  judge's confidence is at least 0.9, and the confident disagreements were the archive's
  own errors (MMS FPI plasma moments labelled MagneticField, a JADE density labelled
  EnergeticParticles). So the floor is 0.9 — below it the field stays empty — and a
  published label the judge contradicts is kept and flagged as `measurement_type_jev`, for a
  person to adjudicate, never replaced. A filled type carries `measurement_type_source:
  "jev"` and its confidence, and enters the text (`Measurement: X.`) where both search
  channels read it; every call is recorded to `judgment_index.jsonl` beside the index. On 50
  live products: 20 filled, 2 flagged, 22 abstained. The same request asks the SPASE
  *region*, because the one the indexer had was a guess: a 40-entry table matched as a
  substring, which against AMDA's 8 435 published dataset targets agrees on 26.9 %, is
  silent on 41 % and wrong on 32 % ("ac" inside "cce_mepa_ion_act" filed AMPTE/CCE
  magnetosheath counts near L1). Measured on 200 of those products, target stripped: the
  judge agrees exactly on 70 %, on the body — Earth, Jupiter, the heliosphere — on 97.1 %
  at confidence ≥ 0.9, and where judge and table differ the judge is right 75 times to the
  table's one; its confident disagreements with the archive are granularity in both
  directions (Helios filed as Heliosphere, a Galileo Io flyby read as Jupiter). So a
  published target is never touched, the table's guess is replaced or the silence filled
  at or above 0.9 (`region_source: "jev"` and the confidence; below it the guess stays,
  marked `table`), and every product now says where its region came from. Without a
  judging backend the flag is a no-op that says so. Separately, every product's text now
  ends with `Coverage: … to …`, as the catalogue index has always said `Survey: … to …`: a
  year in a query used to match nothing and only dilute the rest. All of it changes the
  index; a rebuild applies it.
- **A shipped recipe is offered at the moment a hand-written copy of it runs, not after
  the answer.** The recipe check (`recipe_bypassed`) reads the run's exports at the end of
  the turn and annotates the reply — for the reader, once the model has stopped acting. On
  the fourth live run of the quickstart the analyst loaded `theta_bn`, rewrote the formula
  inline, exported `theta_bn` and reported 54.85° from a window the recipe would not have
  chosen; nothing told it before it answered. The same signals now ride on the `run_python`
  result itself, in both loops: `recipe_available` names the recipe, the reason
  (`not_called`, `not_loaded`) and the exact `run_recipe(...)` line — read off the recipe's
  own source, with the input names it binds through `globals().get`, or its public functions
  when it is a library. `load_recipe` carries the same `run_with` line, so a model that has
  just read a recipe sees the call, not only the code to paste. Both annotate and neither
  blocks: the code ran, its exports stand, and the model may still argue. The lead's prompt
  still does not name `run_recipe` — analysis is what the lead delegates (decided
  2026-09-22, recorded in `tools/setup.py`).
- **Tool results say what the system already knew, in fields, not in prose.** Five
  payload changes, each additive, from the same finding: HelioAI held facts in typed fields
  three calls before it asked a model to guess them from truncated English.
  `get_timeseries` returns `obtained_start`/`obtained_stop` beside the requested window
  (a series clipped to the archive or to a gap was announced with the window asked for —
  the 2026-09-18 θ_Bn of 12° for a 54° shock, every downstream check green), and a partial
  overlap carries `available_start`/`available_stop` as keys as the non-overlapping refusal
  always did. `cadence_ms` rides beside the `cadence` string; `quality.n_gaps` counts what
  the ten-entry `gaps` list cut. A `search_parameters` hit renders `measurement_type` and
  `region` when the archive states them, and `flags` names each reason the ranking pushed
  it down (`other_mission`, `browse_quality`, `outside_window`, `other_quantity`,
  `housekeeping`, `auxiliary`, `model_derived`) — the order carried no reason before, and a
  margin between two RRF sums was rejected as a confidence number because two such sums
  are close by construction. `list_catalogs` takes a `query` and orders by relevance through
  the catalogue index that `helioai index` built and nothing read. `magnitude` refuses
  anything but three components, in the sandbox and in the exported notebook alike. The
  standalone notebook header reads the cell's syntax tree, not its text — a comment
  mentioning `u.nT` no longer imports astropy. And when the series a download returns
  stops short of the window asked for — a gap at the edge, a file not yet delivered — the
  result carries one sentence, `window_note` ("obtained A → B (asked C → D)"), where
  before the asked-for window was announced and only the timestamps knew; the web
  parameter card prints the same sentence. Tolerance is the larger of a minute and 5 % of
  the window, so a cadence offset or a daily file boundary stays silent.
- **The question is read once, into a contract — the `intent` event, in observation.**
  The first site of the judgment seam. With `HELIOAI_JUDGMENT_BACKEND=jev` and the
  `judgment_intent` experiment named, the lead hands the user's question, verbatim and
  alone, to the judge concurrently with the first model call, and reads the answer only
  when the turn is over: what is to be delivered (a value, a figure, a catalogue, an
  explanation, a procedure), which SPASE measurement type the request is about — a `Choice`
  over the index's own vocabulary, so a later comparison with what was retrieved is an
  exact string match — which coordinate frame, and whether an event, an uncertainty, a
  method or two spacecraft were named. A date is asked as three closed choices (year,
  month, day) and assembled by the code with its precision; the judge never writes a
  date, or any free text, into the system. Each field decides or abstains (`None`) —
  "the judge did not say" and "the request named nothing" are kept apart — and every call
  is recorded in `judgment.jsonl`. The event is rendered by the CLI, the web client and the
  magic, and read by nothing yet: the six HelioBench questions with an `expects` block were
  put through it and every decided field agreed with the key; the deliverable abstains on
  a request that asks for a plot and a number at once, which is the field's next shape.
  `RunContext` carries the question (`query`) for the sites that follow. The event then
  carries, under `checks`, what the turn did about the contract — four joins with no model
  in them (`helioai.core.joins`), each an exact operation on fields the turn already had:
  the frame named against the `coord_sys` of every product the sandbox loaded ("GSM asked,
  `BGSE` plotted" was caught by nothing); the date named against the bounds the downloads
  obtained, plus the series that stopped short of their window; the measurement type named
  against the indexed type of the products loaded — a key lookup, exact once `--classify`
  has filled the field; and what was required against what was delivered: an uncertainty
  asked and no claim or export about a spread, a figure asked and none produced. A join
  with nothing to compare reports `None`, not a verdict. The CLI, the web client and the
  magic print only the joins that disagreed, after the contract, so a turn that got it
  right reads exactly as before.
- **Spacecraft positions are searchable: the 314 SSCWeb trajectories are indexed.** An
  SSC inventory node has neither `xmlid` nor `description`, so the indexer skipped every
  one of them and the index held no spacecraft position at all — "where was MMS1" could
  only be answered from an instrument's own ephemeris variable, and the `provider="ssc"`
  the prompt offered returned other providers' hits with no word that SSC was empty.
  Each trajectory is now `ssc/<id>` (the id `spz.get_data` downloads: GSE km by default),
  described the way a position question is asked; and a search filtered on a provider the
  index does not hold says so, with the providers it does. **Run `helioai index` once to
  pick them up.**
- **A position question gets the position, once.** For "MMS1 spacecraft position" the
  six variants of `mms1_mec_pmin_gsm` — the min-B point of the field line, computed with
  a model, under two cadences and three field models — were the whole top-6, and the
  position vector `mms1_mec_r_gsm` ranked 165th. A product whose description says it is
  a model-derived point (min-B, footpoint, field line, apex…) is pushed down on a
  position query unless the query asks for it, and the variants of one product (same
  provider, instrument and variable across BRST/SRVY and field models) cost one slot,
  the others listed under `also_in`. Measured on the live index: `ssc/mms1` then
  `mms1_mec_r_gsm` for the query that used to return six `pmin_gsm`.
- **A search result lists the variables of the dataset it found.** The top hit of each
  `search_parameters` query carries `dataset_variables`: every indexed variable of its
  dataset by name (`cda/WI_H1_SWE` → `Proton_Np_moment`, `Proton_VX_nonlin`,
  `Proton_W_nonlin`, … 41 in all, capped at 40 with the total). One search then shows
  what the dataset holds, and a name that is not in the list does not exist under it —
  the model that found `Proton_Np_moment` no longer asks eleven more times for a
  `Proton_Temp` SWE never had.
- **A role that keeps searching instead of downloading is told to stop.** A
  `data_analyst` spent all twelve of its turns on `search_parameters`, re-asking for
  `Proton_Temp` and `Proton_V_GSE_moment` — names it imagined, which SWE does not have —
  while `Proton_W_nonlin` and `Proton_VX_nonlin` sat in the results of its first call; the
  run ended with nothing downloaded and the lead had to start over. Each role now has a
  search budget (`data_analyst` 3, `plasma_physicist` 2; none for the roles whose job is
  to search): past it, with no data tool called yet, the loop appends one correction that
  lists the product ids the searches already returned and says to download — the same
  `correction` event a replay shows for invented ids.
- **A recipe runs as shipped: `run_recipe(name, inputs)`.** `load_recipe` handed the model
  the source and the model then pasted a part of it into `run_python` — or read its
  constants and rewrote the computation by hand, which is what the recipe check keeps
  catching. The new tool binds the inputs first (`{"B_up": "load_data('b3gsm').values[m_up]"}`,
  each a Python expression evaluated in the sandbox, or a literal), inserts the recipe's
  source verbatim, and — for a recipe that is a library of functions rather than a script,
  such as `rankine_hugoniot` — applies one `call` to the inputs. The numbers come from the
  recipe's own `export()` calls, the script written to the workspace is the one the
  notebook export reproduces, and the run is recorded as a use of the recipe with its
  reference. A recipe whose demo is guarded by `if __name__ == "__main__":` runs its
  functions, not its demo. `data_analyst` and `plasma_physicist` may call it; a recipe run
  this way is exempt from the recipe check. Tested in the real sandbox on the shapes the
  data actually has — a Wind SWE scalar is `(N, 1)`. The `data_analyst` and
  `plasma_physicist` instructions now say to run a recipe this way and to use
  `load_recipe` only to read one; the lead's closing instruction says what a claim's
  `source` is (the exact `export()` name, or `asserted` for a number that was only
  printed) and that a time, a date or an id is not a claim — the first live runs filed a
  normal's components under the angle's export and a timestamp as a number.
- **One agent loop.** `stream_chat` (the lead) and `stream_subagent` (a delegated role)
  were two copies of the same loop — call the model, start the tool calls, review the
  figures, emit the events, append the results — and drifted the way copies do. The loop
  now exists once, `runtime.Runner`, driven by a `runtime.Policy` that says what makes a
  run the lead or a role: the prompt, the tools shown and the tools allowed, the turn
  budget, whether the first turn must call a tool. The two public generators are thin
  wrappers around it and every interface sees exactly the events it saw, in the same
  order — the loop tests and the journal golden did not change.
- **The answer can carry its numbers.** The lead may close an analysis with
  `final_answer(answer, claims)`: the reply as prose, plus one entry per number it states
  — name, value, units and where it comes from (the export it was computed as, a
  published value, or a plain assertion). The history keeps the answer as an ordinary
  assistant message, so the export, the replay and the next turn see nothing new; the
  claims ride on the `reply` event and into the journal, ready to be judged by name
  rather than found by regex. A plain text reply remains accepted. The lead's prompt
  gains one paragraph saying when to use it.
- **The plan is held to.** `present_plan` was a display: the loop forwarded the steps to
  the interfaces and forgot them, so a plan that promised the `theta_bn` recipe and ran
  hand-written arithmetic instead looked exactly like one that was followed. The plan is
  now kept as data (`runtime.plan.Plan`) and, when the turn ends, compared with the tools
  the lead actually called: one `plan_report` event names the planned tools that were
  used, the ones that were not, and the ones used without being planned, with the share
  followed. It describes and never blocks — a capped turn reports how far the plan got
  before its error — and it is journaled and rendered as one line by the three
  interfaces. A step's `tool` field is read word by word against the tools the lead
  knows (the model writes "search_parameters + get_timeseries"), and a planned tool a
  sub-agent ran for the lead is a step done, not a deviation — on the first live run the
  lead delegated every step and the report said `0/4 used, unplanned: task`. What a
  sub-agent does with its delegation is its whitelist's business, never "unplanned"; only
  the lead's own improvised calls are. Since a plan written in delegations alone is
  always "followed", the report also says how each delegation ended — role, turns, and
  whether it hit its cap (`sub_agent_end` now carries `capped`): "3 delegations,
  librarian capped" is the line the run after the Runner extraction needed, when the
  librarian ran out of turns and was re-delegated. The scaffolding calls (the plan
  itself, the skills, `search_tools`, `final_answer`) count for nothing on either side.
- **One verdict on the answer.** The lead's reply was judged from four places — the
  catalogue ids, the recipe bypass, the numbers in the prose, the figures — each with its
  own event and none aware of the others. `runtime.validator` runs them in one call, and
  adds the judgement the claims make possible: each number `final_answer` named is placed
  against the provenance ledger **by name**, with a unit-aware tolerance (`57.2 deg`
  states a recorded `57.16 deg`, `0.0101 uT` states `10.077 nT`, a mean quoted within one
  standard deviation of a series is not a different number), the prose checker's own
  tolerance so one answer is held to one rule, and the rule that any run which produced
  the value sources it. A named scalar the session computed that holds another value is
  `contradicted`; a claim the model marked `literature` or `asserted` never is, nor is a
  claim that gives no units — on the first live run the model filed a normal's
  components under the angle's export, and "n_x stated −0.509, the session computed
  54.85 deg" accuses nothing a reader can act on. The result is a `verdict` event — counts up front, every claim behind them —
  journaled and rendered by the CLI, the notebook and the browser; it is emitted only when
  the answer named its numbers, and the existing `invalid_ids`, `recipe_bypassed` and
  `provenance` events keep flowing for the prose.
- **The answer streams.** The lead's text is shown as the model writes it — token by
  token in the CLI and in the browser, where a live bubble is re-rendered as Markdown
  once the reply is complete; the notebook keeps rendering the finished answer. Every
  OpenAI-compatible provider (Groq, OpenCode, Ollama, Azure) streams; Gemini and any
  client without streaming support hand the reply over whole, so nothing breaks behind
  a caller that streams. An inline `<think>` block is held back until it closes. The
  deltas are not journaled — the `reply` that follows is — so a replay shows the answer
  once. Sub-agents do not stream: their text goes to the lead, whole.
- **Fewer tool definitions per model call.** Twenty-one tool schemas — about 3 800
  tokens — rode on every call of a lead turn. The six plasma-physics tools and the four
  catalog tools are used in a minority of sessions; their definitions are now withheld
  until the agent asks for them with `search_tools` or calls one by name, which leaves
  eleven schemas (about 2 100 tokens) on an ordinary turn. The lead's prompt gains one
  sentence saying so.
- **A delegated role can run on its own model.** `HELIOAI_ROLE_MODELS=parameter_hunter=
  groq:llama-3.3-70b-versatile,librarian=groq` gives a role a provider and a model of its
  own; a `parameter_hunter` resolving ids from search results does not need the lead's
  frontier model, and a `data_analyst` writing the physics does. The role's client is
  built for the run and closed after it, and its tokens are billed to its own provider in
  the `usage` table. Roles not listed keep running on the lead's client, as before.
- **A run knows where it writes.** Who is running, in which session and into which
  directory used to live in three contextvars set by each loop, by the MCP server and by
  nobody in a test, and read back from inside the tools — a sub-agent worked only because
  the lead had bound the label first, and a tool called directly wrote under the default
  user. `runtime.RunContext` carries those facts; the runner binds them in one place, and
  the four tools that write (`run_python`, `get_timeseries`, `get_events_timeseries`,
  `save_catalog`) receive their directories from it as trusted arguments the model cannot
  supply. Each MCP connection gets a context of its own.
- **A session replays from its journal.** Every event a turn yields — the question that
  opened it, each tool call and result, the plan, the figures, the provenance verdict,
  the sub-agent trace — is appended to an `events` table before it reaches the screen.
  Reloading a session in the browser renders that journal with the same code as the live
  stream, so nothing that was shown is lost: the previous replay re-parsed the JSON of
  every tool message and guessed the figures from its shape, and lost the plan, the
  verdicts and everything a sub-agent did. Sessions recorded before the journal existed
  keep the old view (`legacy_replay.py`), used only when there is no journal. The stream
  now opens with a `user` event carrying the question; the CLI and the notebook leave it
  unrendered. The automated correction the loop sends the model when an answer quotes
  ids that exist in no catalogue is a `correction` event too, so a replay shows what the
  model was told as a system note rather than losing it — the one thing the journal did
  not carry.
- **A tool call returns a typed result.** `registry.call_tool` used to serialise every
  tool's dict to JSON on the spot, and five readers downstream — the history, the artifact
  extractor, the figure review, the MCP server, the event display — each parsed that text
  back to learn what the tool had returned. `ToolResult` (`tools/results.py`) carries the
  payload once and `for_llm()`, the text the model reads, once; sixteen results captured
  from live calls pin that text byte for byte, so the model reads exactly what it read
  before. MCP clients now also receive a dict payload as `structuredContent`.
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
- **The event contract lives in one place, and is enforced as events are made.**
  `core/events.py` lists every kind the agent loops emit with its payload keys and every
  artifact kind; a static test holds the emitters, the CLI, the notebook magic and the
  browser to that list, so a kind added to a loop and forgotten in one interface — or
  documented and never emitted — fails in CI. Three docstrings used to carry their own,
  disagreeing copies. Every emission now goes through `events.make` / `events.artifact`,
  which refuse an unknown kind or a payload missing a key the interfaces rely on — the
  `sub_agent_end` of a failed delegation lacked `findings` and `usage`, and the loop
  tests that exercise the branch now say so.
- **Token usage is kept.** Every provider reports what a call cost and every count was
  dropped at save time, so nothing could say what a session — or a user — had spent.
  Each lead call is now a row in a `usage` table (turn, agent, provider, prompt /
  completion / cached tokens); a sub-agent reports its total on `sub_agent_end` and is
  charged to the parent session under its role. `helioai history` shows a tokens column,
  `GET /api/me` returns the caller's totals for the day, the month and all time — the
  number a per-user quota compares against.
- **A configuration reference** (`docs/configuration.md`): every environment variable,
  its default and its effect, grouped by provider / agent / storage / serving. A test
  fails when the code reads a variable the page does not list.
- **CI builds the Docker image** and runs `helioai doctor --json` inside it, printing
  whether bubblewrap is functional in the container — the claim `SECURITY.md` makes and
  nothing verified. Stale CI runs of a pull request are cancelled; Dependabot watches
  the lock, the actions and the base image weekly.

### Removed

- The cross-encoder reranking stage of parameter search — measured to degrade results
  and disabled for a year (`RAGConfig` keeps the measurement in its docstring); the
  `rerank_*` settings go with it. Also gone: `run_subagent` (no callers) and the
  `data_preview` artifact renderers in the three interfaces (no emitter).
- **`HELIOAI_PROFILE`.** It moved a file the agent stopped reading when storage became
  per user: the profile injected into the prompt is `users/<user>/profile.md`, which
  `helioai profile`, `%helioai_profile` and the web UI all edit. The variable was
  documented as "injected into the system prompt" and did nothing. A profile written
  by an older install is still picked up by `helioai migrate-storage`.

### Changed

- **`theta_bn` averages over 13-minute windows, the Harvard-CfA convention, instead of
  8-minute ones.** The CfA shock database publishes θ_Bn method by method, and its
  magnetic-coplanarity (MC) entries rest on 260 field samples per side — 13 min at 3 s — so
  a recipe that is itself an MC estimate can only be compared to it at those windows. On
  CfA shock 00368 (Wind, 2004-11-07 17:59:05 ± 60 s UT) the recipe at guard 2 min / span
  13 min gives **54.3°** against the CfA's **MC 54.5 ± 3.6°**, with the two window means
  matching the CfA's asymptotic states, and the velocity- and mixed-coplanarity normals
  computed on the same windows land within 0.3σ of its VC, MX1, MX2 and MX3; the former
  default gave 49.1° on that shock. The difference was a window convention, not a method
  error — which is exactly what the exported `theta_bn_window_spread_deg` exists to say.
  On 2015-03-17 04:00 UT the same change moves the answer from 62.2° to 60.9° (CfA MC
  58.8 ± 2.7°). The spread grid moves with the default, to spans of 8, 10, 13 and 15 min
  (guards 1, 2, 3): 49.1–57.7° on the 2004 shock, 60.7–63.6° on the 2015 one. The
  quickstart notebook asks for 13-minute windows accordingly; the 62.68° it reported under
  0.3.0 was correct for the 8-minute convention it asked for then.
- **Behaviours that change what the model sees or is told are named experiments, off by
  default, and go on by measurement: `HELIOAI_EXPERIMENTS=deferred_tools,search_budget,
  search_variables`.** Each shipped on the strength of a single live run and none was ever
  measured against the loop it replaced; when the same question came out differently on
  `main` and on this branch, nothing could say which one cost what — and the nineteen
  recorded runs of that question since June spread θ_Bn from 44° to 80°, so one run per
  branch is noise, not a comparison. With the variable unset the lead's prompt, its tool
  set, the roles' prompts and skills, the search budgets and the search payload are the
  reference loop's; naming an experiment changes exactly its prompt text and its `Policy`
  field (`tests/test_experiments.py`). The default is not `main`: `run_recipe` and the
  corrected recipes are in, and so is **`final_answer`**, which graduated on the third
  benchmark — 34 clean runs, one workspace each, four configurations interleaved: no cost
  on any of the five questions, the claim verdict back (zero contradictions over nine),
  rejected candidates named as often as by the reference loop. The two search experiments
  showed no value there (eight id-resolution runs correct without them) and stay off until
  a question that fails without them is recorded; `deferred_tools` was not isolated. An
  unknown experiment name is refused where the API key is checked (`build_llm_client`,
  `helioai doctor`), not at import: the MCP server and the web app must still start.
  `scripts/bench_live.py` runs a fixed question set N times per configuration and scores
  the sessions from the journal against a truth block per question — shock time, θ_Bn
  bands, valid ids, whether the answer states its windows and names what it rejected.
- **Two prompt sentences the third benchmark asked for.** The lead reads "around <date>"
  as the whole UT day (a window centred on midnight put the day's main shock at its edge,
  with no downstream to average, and the analyst — correctly — analysed the other one);
  the `data_analyst` skill screens shock candidates against density and speed through
  `theta_bn`, reports `theta_bn_window_spread_deg` as the ±, binds `B` + `shock_time`
  rather than hand windows, and prints the windows and mean vectors it used.
- **The recipes were reviewed by a heliophysicist and corrected; their numbers change.**
  Each correction shipped with a synthetic test whose answer is known, red before the
  fix, and the shelf now has one contract: a recipe reads its inputs with
  `globals().get` and exports nothing when none is bound, its demo lives under
  `if __name__ == "__main__":`, and every `export()` carries a unit — the provenance
  validator compares unit-aware, and a speed recorded bare could not vouch for
  "V_shock = 579 km/s". Recipe by recipe:
    - `theta_bn`: the upstream/downstream windows can be derived from the shock time
      (`shock_time` + the series `B`; guard 2 min, span 13 min — see *Changed*; a window with a step or a
      trend that looks like the ramp is refused) instead of chosen by the model — three
      live runs on the same shock had given 54.85°, 59.95° and 64.27°. Exports gain the
      normal, the magnetic compression ratio, the two window means, a bootstrap spread of
      the angle and the normal, and the std of B·n̂. The naive "coplanarity residual" is
      identically zero for this estimator and is documented as such rather than exported.
      The placeholder demo no longer runs under `run_recipe`. Run with the series alone,
      the recipe lists the largest |B| jumps of the interval (`find_shock_candidates`:
      time, jump, ratio) and stops, so the crossing time is picked from a list rather
      than hunted with hand-written cells — a live run spent nine of its twelve turns on
      that hunt.
    - `mvab`: the Sonnerup & Scheible (1998, eq. 8.23–8.24) angular uncertainties of the
      normal and Δ⟨B·n⟩, on the covariance convention of the reference (÷N); λ_min ≈ 0
      is now "planar — normal unique, uncertainty undefined", λ_int ≈ λ_min "degenerate",
      neither "well-determined"; a warning under 30 samples; NaN input is an error, not a
      traceback.
    - `walen_test`: a real de Hoffmann-Teller frame (Sonnerup et al. 1987, the 3×3 normal
      equations) replaces the mean subtraction the docstring called one; the HT quality
      (residual electric field, E-field correlation) is reported and a poor frame is said
      to make the slope meaningless; the RD verdict needs 0.7 ≤ |slope| ≤ 1.3 and R² ≥ 0.8
      (Paschmann & Sonnerup 2008) — a slope of 10 with R² = 1 used to be "consistent with
      a rotational discontinuity". `frame="mean"` keeps the previous behaviour.
    - `superposed_epoch`: a gap stays a gap — neither an explicit NaN nor a missing
      timestamp is bridged (`np.interp` bridged both), an epoch with fewer than
      `min_events` contributors is NaN, and the new bootstrap CI on the median is masked
      where the median is; τ is normalised on the event's `start`/`stop`, not on its first
      and last surviving samples; the units come from the events themselves.
    - `pressure_balance`: the reference field is derived from the dipole (IGRF-13 B₀ =
      29 806 nT, Chapman–Ferraro factor 2 → 59.6 nT at 10 R_E) instead of a round 50 nT;
      `P_dyn_nPa` is the standard ρV² and the 0.88 stagnation coefficient (Spreiter et al.
      1966) is exported apart as `P_applied_nPa`; the result says it ignores Bz and points
      to `mp_shue1998`. **`mp_standoff` returns a dict** (`result["r_mp_RE"]` is the former
      float).
    - `sep_onset_poisson_cusum`: the CUSUM runs on the intensities with the Poisson
      reference value of Huttunen-Heikinmaa et al. (2005) — the previous z-score form is
      the same detector divided by σ, and stays as `method="zscore"`; the background is
      median/MAD; detection starts after the background window and its first sample
      counts; a missing sample no longer confirms an onset; units follow the flux.
    - `pitch_angle_dist`: equal-solid-angle bins (in cos α) replace the 1/sin α
      normalisation that amplified Poisson noise at the poles; `bins="deg"` keeps the old
      histogram; the distribution and an anisotropy ratio are exported; the recipe no
      longer switches matplotlib to `dark_background` for the whole session.
    - `rankine_hugoniot`: its eleven exports carry their units.
- **If you set `HELIOAI_DATA_DIR` (the Docker image does), run `helioai migrate-storage`
  once after upgrading.** It moves the index, the catalogues and the profile from the
  default directory to the configured one, never overwrites, and can be re-run. The
  `search_parameters` error names the legacy copy when it exists, so an upgraded
  install is not sent into an hour-long rebuild.
- `plasmapy>=2026.2` (the version the lock already resolved); classifiers now say
  `Development Status :: 4 - Beta`, `Python :: 3 :: Only`, the three operating systems
  (the sandbox is only real on Linux), `Framework :: Jupyter` and `Typing :: Typed`, and
  the wheel ships a `py.typed` marker.
- The session database gains `messages.origin` and `messages.name` columns, added
  automatically the first time an existing database is opened. `name` records which
  tool produced a `tool` message; readers no longer have to recognise a tool by the
  shape of its JSON (a loaded recipe was identified by having `name`, `code` and
  `metadata` keys at once).

## [0.3.0] — 2026-09-21

Two things changed in this release. **HelioAI is now a complete MCP tool provider**: an
agent you already use — Claude Code, Claude Desktop, Codex — can call its tools, follow
its skills as slash commands, read its recipes and receive its figures, with no LLM key of
HelioAI's own. And **a result now carries the evidence that it is what it says it is**: the
data a script loaded is the data it named, the exported notebook computes what the sandbox
computed, and every number in an answer is judged against the quantity it claims to be.

The release was qualified on one scenario, `examples/00_quickstart.ipynb` — the Wind shock
of 2015-03-17 04:00 UT, θ_Bn by magnetic coplanarity — run live five times against two
independent databases (Harvard-CfA 58.8 ± 2.7°, IPShocks 63.1 ± 16.8°). The guided
analysis landed in the band on every run (62.68°); the same analysis asked as a single
delegated question did on runs 3 and 5 and missed once, on run 4, at 54.85° — a hand-written
copy of the recipe over a badly chosen averaging window, which is what the new recipe-check
signal below detects. What that qualification does and does not cover is listed under
*Known limitations*.

### Added

- **MCP 2.0.** `helioai-mcp` migrates to `mcp>=2` and the `<2` cap is gone (closes #1).
  The 11 recipes and 6 skills are fetchable as `recipe://<name>` and `skill://<name>`
  resources, and the six skills are also **MCP prompts** — `/helioai:data_analyst plot
  IMF Bz for 2015-03-17` arrives as one message. Sandbox **figures come back as inline
  image content** (768 px), not as paths on the server's disk. Tool failures are reported
  as `isError`; the 15 tools that change nothing are annotated `readOnlyHint`, so a client
  need not prompt for `list_missions` as it does for `run_python`.
- **Each MCP connection gets its own session workspace.** Every call used to fall through
  to one process-wide temporary directory: two `run_python` calls overwrote each other's
  `code_0.py` and figure, and one client's `load_data()` saw another client's downloads.
- **`helioai mcp-install`** prints ready-to-use client configuration for Claude Code,
  Claude Desktop and Codex, with the server path resolved from *this* interpreter rather
  than from `PATH` — the reason `"command": "helioai-mcp"` did not work from a venv.
  `--write` merges into the JSON configs and refuses a file it cannot parse.
- **`helioai --help`.** It used to be sent to the model as a question.
- **`examples/00_quickstart.ipynb`** — one shock end to end in about three minutes: find
  the parameter, download and plot, θ_Bn with the vetted recipe, export; then the same
  analysis as a single delegated question. The notebook states the expected band and
  cites both external references, so a run can be judged against a number the agent did
  not produce. `%helioai_session new` starts a fresh conversation without deleting the
  previous one, which `reset` does.
- **Token counts.** `Message` carries the prompt, completion and cached token counts every
  provider already sends and every client used to drop.
- **Configurable HTTP headers** for the OpenAI-compatible providers
  (`HELIOAI_<PROVIDER>_HEADERS=name=value,…`, `{uuid}` expanded per client), for gateways
  and proxies that require one.
- **`HELIOAI_LOG_LEVEL`** overrides the level each entry point hardcodes.
- **A third recipe-check signal.** Loading a recipe and calling its function are now
  distinguished: a run that loads `theta_bn`, rewrites the formula by hand and exports the
  result under the recipe's own name is flagged *loaded, never called*. On the live run
  that motivated it, that hand-written copy gave 54.85° where the recipe gives 62.68°.
- **`AGENTS.md`** scopes what a coding agent may change in this repository, and what it
  cannot verify offline.

### Changed

- **The MCP server starts without an LLM key.** Config validation moved from import time to
  `build_llm_client`, where it already was; a fresh install wired into Claude Desktop used to
  die on `AZURE_OPENAI_API_KEY is not set` before serving a tool.
- **`--http` is gated by `HELIOAI_MCP_TOKEN`** and refuses to bind off-loopback without one.
  It used to expose `run_python` unauthenticated with a warning.
- **One readable account of a run in all three interfaces.** Tool traffic is described for
  the reader (`get_timeseries: cda/WI_H0_MFI/B3GSE — WI — 3 s — 1800 points`) instead of
  showing the model's JSON briefing truncated mid-key. The operator's home directory no
  longer reaches the model, so it stops quoting `/home/<user>/…` into answers and
  notebooks. speasy's disabled-provider warning is one line, its traceback at DEBUG; no
  `Loading weights` progress bar inside a conversation.
- **The web UI binds a streaming reply to the session that asked.** Each session owns a
  view; switching sessions while a reply streams no longer lands it in the other session.
  A turn cut short keeps its figure and script. The provider selector follows the server's
  configured provider (it used to send `azure` regardless) and offers Ollama.
- **The index describes products with what the archive publishes** rather than with
  heuristics: the owning mission (780 AMDA parameters could not be found by naming their
  own spacecraft), the processing level (the definitive ACE IMF vector was demoted as
  "browse" for the word *PRELIM* in a Level 2 description), and the SPASE region (1279
  AMDA parameters carried a wrong one, 0 after). **Takes effect after
  `helioai index --rebuild`**; `helioai index` is incremental and leaves an existing
  index as it is.
- **`search_parameters` no longer returns a `score`.** It was written before two of the
  four reranking stages and contradicted the order — rank 1 carried 0.09 and rank 2 carried
  1.0 — which a client model with none of our system prompt read at face value.
- **A fabricated parameter id costs the model a turn.** Detection used to staple a
  correction onto an answer already written and end the loop; the model now has to answer
  again. Once, in both loops. A real *dataset* id (`cda/WI_H2_MFI`) is no longer accused
  of being a fabrication because only parameters are index keys.
- **History compaction keeps the numbers.** A stale tool result never loses `findings`
  or `exports`; `summary` keeps 1 000 characters and `stdout` 400; a loaded recipe is never
  summarised at all. At the previous cap the analyst reloaded the recipe and once rewrote
  the formula from memory.
- **Sub-agent `run_python` runs without a network namespace**, since sub-agents download
  data; the seed of speasy's inventory into each sandbox home shrank from 708 MB to 96 MB.
- **`power_spectrum`** segments a gapped series into contiguous runs and averages Welch
  over them, instead of splicing the gaps shut and corrupting the spectral slope.
- **`rankine_hugoniot`** projects the velocity on the shock normal when `normal=` is given
  (the jump conditions want V·n̂, not |V|), refuses a scalar speed by name, and fixes the
  velocity sign on entry so a flipped coplanarity normal no longer silences the
  consistency check.
- **`theta_bn`** returns `{"error"}` and nothing else on NaN, zero or collinear input — a
  NaN used to come back labelled *quasi-perpendicular*. An `(N, 3)` window is averaged
  over its finite rows, so a gapped interval can be passed as is.
- The Docker image keeps speasy's 840 MB inventory inside the data volume instead of
  re-downloading it on every recreate.
- Every public object documents its inputs and outputs (PyHC standard 8); the 65 that did
  not are now zero. The README is a landing page with two real session recordings; the
  manual lives in `docs/`.

### Fixed

- **`HELIOAI_DATA_DIR` did not move the search index.** Sessions and user homes followed
  it; the Chroma index, the legacy profile and catalog paths stayed at the default
  location, so an installed server pointed at a prepared data directory reported
  `ChromaDB index not found` beside the index. Everything now hangs off `data_dir`. In the
  Docker image this puts the index at `/app/data/chroma` — the same directory as a clone's
  `data/chroma`, which the compose file already mounts — instead of
  `/app/data/helioai/chroma`.
- **`torch` is a declared dependency (`>=2.6`).** HelioAI never imports it, but transformers
  5 requires `torch>=2.5` without saying so, and `uv pip install` of the wheel in a fresh
  environment resolved torch 2.4.1 to escape an unrelated `mpmath` conflict. transformers
  then disabled torch at import, the dense search failed, and `search_parameters` fell
  back to a text scan with no error at the tool boundary.
- **The search fallback note names what failed.** It said `RAG index not built` whatever
  the dense search had raised — on the install above, it told the user to build an index
  that existed. The note now carries the exception.
- `helioai-mcp` answers `initialize` with the package version; it used to send an empty
  string.
- **`load_data("<full id>")` returned the wrong spacecraft.** Two products ending in the
  same parameter name slugged to the same key and the resolver matched on that suffix:
  asking for `MISSION_B`'s field returned `MISSION_A`'s — right units, plausible numbers,
  no error. Both loaders (sandbox and exported notebook) now resolve through the recorded
  `param_id`, and an ambiguous id raises listing the candidates.
- **The exported notebook computed something else than the session.** `load_data()` was
  rewritten to a bare `spz.get_data()` that fed the raw fill sentinel to the same arithmetic
  (mean 5.0 in the session, 50 002 in the notebook); `export(name, data, units=)`,
  `magnitude()` and `interp_to()` did not exist in the notebook; imports the header did
  not provide were stripped; a failed attempt exported as an executable cell and stopped
  *Run All*; recipes loaded by a sub-agent were missing from *Methods*. The exported
  helpers are held to the sandbox's behaviour by tests on the same inputs.
- **Provenance vouched for a number with the wrong quantity.** The ledger held
  `B_downstream = 10 nT` and `density = 25 cm⁻³`; "B downstream = 25 nT" was reported
  *matched*. The quantity the sentence names is judged first; a hit elsewhere needs a unit
  that does not contradict the claim; a negative claim against a positive record is the
  wrong sign, not a magnitude; a quantity exported several times is sourced by any of its
  runs; `1.2e-3` keeps its exponent; `-0,657 nT` is not −657 nT; a day of the month is
  not a measurement; `°` is `deg`; and a component of an exported vector is not a
  contradiction of it.
- **A decimal comma made every measurement look unsourced.** A French reply with correct
  physics came back "0 traced, 6 unsourced".
- `helioai-mcp` was not found on Windows, where it is `helioai-mcp.exe`.
- The `plasma_physicist` skill's templates could not run as written (`await`-less calls to
  async tools, `.unit` for `.units`, `np.interp` across gaps, a nan-unaware fraction that
  reported 0.25 for a true 0.5).
- A tool raising `TimeoutError()` produced `{"error": ""}` and was described as "ok".
- The `spz` escape-hatch test and the session fixture no longer depend on speasy's cache
  server being up; the full suite dropped from 648 s to 353 s.

### Security

- **Host credential stores were readable from the sandbox.** Bubblewrap mounted `/`
  read-only, which left `~/.ssh`, `~/.gnupg`, `~/.config/gh` and their kind readable to
  model-written code. They are now masked with empty tmpfs or `/dev/null` before the
  workspace bind. Linux only, as the whole isolation is; see `SECURITY.md`.
- The non-bubblewrap fallback warns when `no_net` was requested and cannot be honoured
  (closes #5).

### Known limitations

Stated here so that the release is not read as more than it is.

- **Scientific validation covers one recipe on one event.** `theta_bn` on the Wind shock
  of 2015-03-17 was checked against two external databases. The other ten recipes carry
  their references and self-tests but have not been qualified the same way in this release.
- **The retention policy runs at start-up.** Session workspaces older than
  `HELIOAI_WORKSPACE_TTL_S` (7 days) are deleted when the CLI or the web server starts,
  data, scripts and figures included; consulting or exporting an old session does not
  protect it. A demo that spans a restart should run under a dedicated `HELIOAI_DATA_DIR`
  and keep its deliverables outside it.
- **`list_catalogs` appends local catalogs to the shared in-process cache**, so on a
  multi-user web or MCP server one user's `local/` catalogs can appear in another's
  listing until the cache expires.
- **`%helioai_provider` does not switch the provider.** It sets the environment after the
  settings have been read; restart the kernel with `HELIOAI_LLM_PROVIDER` set instead.
- **The recipe check is textual.** It reads the code the model wrote; a run that redefines
  a function under a recipe's own name passes it. It annotates and never blocks.
- **Provenance says where a number came from, not whether it is right.** A recorded value
  that agrees with the prose is *traced*; the physics is checked by the recipe and by the
  reader.
- **Over MCP, the client's model is not ours.** The tools run, the ledger records, but the
  answer checks (provenance, recipe check, fabricated-id guard) live in HelioAI's own agent
  loop and do not run on a client's reply.
- **Sub-agents on DeepSeek in thinking mode** refuse `tool_choice="required"` on their first
  turn; the request is resent as `auto`. No turn is lost; one warning is logged per
  sub-agent.

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

[Unreleased]: https://github.com/erdoganfurkan/HelioAI/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/erdoganfurkan/HelioAI/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/erdoganfurkan/HelioAI/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/erdoganfurkan/HelioAI/releases/tag/v0.2.0
