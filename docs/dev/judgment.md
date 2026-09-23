# The judgment layer

HelioAI verifies itself thoroughly — claims against the ledger, ids against the index, code
against the recipes, tools against the plan — and, until 0.4.0, nothing verified it against
the *question it was asked*. The judgment layer is where that comparison lives. It is one
module, `helioai.core.judgment`, a handful of sites that ask it, and one rule that governs
all of them: **it observes, it never corrects.**

## Why a separate judge, and why System One

Every check HelioAI already had is deterministic: a regex, a set membership, a tolerance on
a number. They are fast, reproducible and blind to meaning. The failures the benches kept
producing were failures of meaning — a plot in GSE for a question that said GSM, a confident
answer built on no product that measured the quantity asked for, a θ_Bn computed on a window
that ended at the shock. Catching those needs a reader.

The reader is not the agent's own model. TypeSafe's *System One* (the `jev` backend) answers
closed questions — yes/no (`Noul`) or one option from a fixed set (`Choice`) — in about
250 ms, with a calibrated confidence, and it cannot write free text into the system. That
shape is the point: the judge never rewrites an answer, never picks a product id, never
produces a date. It says *yes*, *no*, *this one of these*, or nothing.

## The safety contract

1. **Off by default.** `HELIOAI_JUDGMENT_BACKEND=null` abstains on everything; the loop is
   exactly the one that shipped before. The `jev` backend needs the optional `judgment` extra
   (`pip install "helioai-agent[judgment]"`) and `TYPESAFE_API_KEY`.
2. **Two axes.** A backend alone asks nothing. Each site is also an experiment
   (`HELIOAI_EXPERIMENTS=judgment_intent`), so "the judge does not help here" and "the layer
   costs something" can be told apart, one site at a time.
3. **Abstention is `None`.** Below a site's confidence floor an answer is `None`, never a
   sentinel that could be summed into a plausible wrong value. Every reader keeps "the judge
   did not say" apart from "the request named nothing".
4. **Bounded.** Every call is under `HELIOAI_JUDGMENT_TIMEOUT_S` (default 2 s). A slow or
   failing judge costs a turn nothing: the contract is asked concurrently with the first
   model call and read only after the answer.
5. **Recorded.** Every call is one JSON line — state in full, questions, raw answers, model,
   latency, request id — under the session workspace (`judgment.jsonl`) or beside the index
   (`judgment_index.jsonl`). A disagreement that cannot be adjudicated later is not a
   measurement.
6. **Nothing reaches the exported notebook.** A standalone export recomputes every number
   without the extra; no scientific statement depends on a judgment.
7. **Nothing reaches HelioBench.** Its graders consult no model.

## The sites

### `intent` — the contract, and what the turn did about it

With `judgment_intent` on, the lead hands the user's question, verbatim and alone, to the
judge as the first model call starts. The questions are in `judgment.INTENT_QUESTIONS`, all
closed:

| Field | Kind | What it fixes |
|---|---|---|
| `wants_value`, `wants_figure`, `wants_catalogue`, `wants_procedure` | `Noul` × 4 | What is to be delivered. Four yes/no rather than one choice: a choice abstained on every request that wanted two things — "plot \|B\| and compute θ_Bn" is a figure *and* a value |
| `quantity` | `Choice` over the index's SPASE measurement types | The physical quantity, in the vocabulary the products carry, so the later join is a string comparison |
| `frame` | `Choice` (GSE, GSM, RTN, …) | The coordinate frame named |
| `year`, `month`, `day` | `Choice` × 3 | The date, as components; **the code assembles the datetime** and its precision |
| `event_named`, `uncertainty_required`, `method_named`, `two_spacecraft` | `Noul` | What the question commits the answer to |

When the turn is over the contract is placed against what the turn produced — the parameter
cards of every product loaded, the claims of `final_answer`, the figures — by
`helioai.core.joins`, four joins with no model in them:

| Join | Compares | The failure it names |
|---|---|---|
| **frame** | frame asked vs `coord_sys` of the cards the sandbox filled from the archive's metadata | GSM asked, `BGSE` plotted |
| **window** | date named vs the bounds each download *obtained*; cards whose series stopped short of the window asked | the 2026-09-18 θ_Bn of 12° for a 54° shock, window ending at the shock |
| **quantity** | measurement type named vs the indexed type of the products loaded (a key lookup) | a confident answer built on no product measuring the quantity |
| **responsiveness** | what was required vs what was delivered | an uncertainty asked and no claim about a spread; a figure asked and none produced |

The result is one `intent` event per turn, contract and `checks` together, rendered by the
CLI, the web client and the Jupyter magic — the mismatches after the contract, so a turn that
got it right reads as before. Nothing in the loop acts on it.

### `index_classify` — the metadata the archives left empty

`helioai index --classify` runs off the loop, with only a backend (no experiment: a job
someone asked for is not a conversation nobody asked to be judged). One request per product,
two `Choice`s, both sentences the text carried stripped first so the judge reads the product
and not the labels:

- **measurement type** where the archive published none — 84 % of the catalogue, all of
  CDA. Filled at confidence ≥ 0.9 (`measurement_type_source: "jev"`); a published label the
  judge contradicts is kept and flagged `measurement_type_jev`, never replaced.
- **region** where the indexer had only guessed from its substring table. Replaced or filled
  at ≥ 0.9 (`region_source: "jev"`); a published dataset target is never touched; below the
  floor the guess stays, marked `table`.

The answers ship with the package. A pass costs money (82 266 requests, US$ 2.4 on
2026-09-22 — metered at 2.9 ¢ per 1 000 requests) and is the one part of the index code
cannot rebuild, so `helioai/data/judged_products.jsonl.gz` (0.66 MB) carries every question
asked so far with its raw answer, abstentions included, behind a line of provenance that
`helioai doctor` prints. `helioai index` applies it without a key; a rebuild from the file
reproduces the paid index to the byte. `--classify` asks only what no record answers and
appends the new answers to a local copy beside the data (`data/judged_products.jsonl.gz`),
which survives `--rebuild` and overrides the shipped file. The loop for a new provider is
therefore: anyone rebuilds and gets the new products untyped, exactly as the whole of CDA
was until 0.4.0; a maintainer with a key runs `--classify`, pays for those products alone,
and the next release ships the extended file. Because the file keeps answers rather than
decisions, the floor can be changed without asking again — and a record whose product
name no longer matches the id is not applied.

## What was measured before each site shipped

Every site had a stop rule, measured before it was built, offline where possible.

| Gate | Question | Result (2026-09-22) |
|---|---|---|
| **G2** | Round trip France → TypeSafe | median 266 ms, p95 373 ms over 20 calls (plan threshold 800) |
| **U13** | Can the judge recover the 12 850 measurement types AMDA/CSA publish, label stripped? | 76.5 % raw, **89 % at confidence ≥ 0.9** (72 % of items); the confident disagreements were the archive's own label errors (MMS FPI moments labelled MagneticField) |
| **U14** | Can it recover AMDA's 8 435 published regions, and beat the table? | table: 26.9 % exact, 41 % silent, 32 % wrong. Judge: 70 % exact, **97.1 % same body at ≥ 0.9**; where judge and table differ, judge right 75 : 1 |
| **U7** | Do the six HelioBench questions with an `expects` key yield contracts that agree with the key? | every decided field agrees; the three mixed requests read `value+figure` at 0.96–0.99 once the deliverable became four yes/no questions |

The three offline harnesses read the index and the inventory and write nothing but their
own output; none of them touches HelioBench.

## What is deliberately not done

- **No correction path.** A verdict that changed what the model sees would need a two-sided
  live replay at n ≥ 3 for each site — the 2026-09-15 lesson, where a well-meant runtime
  change made the agent worse on an ordinary question and only a replay showed it. Until
  then every site annotates; a correction, when one is built, lives behind its own
  experiment name, off.
- **No judge in a tool result during the turn.** The `vision.maybe_review` pattern (a verdict
  in the tool payload, bounded, unable to fail the turn) is the intended shape for the
  remaining semantic checks — catalogue relevance, delegation-brief adequacy — and none is
  built yet.
- **No `local` backend.** A local model would be a third thing to validate; `null` and `jev`
  are the two the tests cover.

## Reading the records

```bash
# every judgment of a session, one JSON line each
cat data/users/<user>/sessions/<label>/judgment.jsonl | jq -c '{site, latency_ms, answers}'
# what the indexing pass decided, and how confident it was
jq -c 'select(.answers.region.confidence < 0.9) | {state, answers}' data/chroma/judgment_index.jsonl
```

The configuration reference lists every variable; `helioai doctor` reports the backend, the
key and the timeout it will use.
