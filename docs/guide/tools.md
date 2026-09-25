# Agent tools

The agent has 18 tools. You never call them directly — you ask a question and it picks —
but knowing what exists tells you what HelioAI can be asked for.

Not all of them are shown to the model on every turn. The six plasma-physics tools and
the four catalog tools are *deferred*: their definitions are withheld until the agent
asks for them (`search_tools`) or calls one by name, which cuts the tool schemas re-sent
with every model call from 22 to 12. Nothing changes for you — the agent still finds them
when a question needs them.

## Data access

| Tool | What it does |
|---|---|
| `search_parameters` | Hybrid search over ~83 000 speasy products. Accepts a batch of queries. |
| `get_timeseries` | Downloads a parameter, persists it, returns cadence, units and components. |
| `list_missions` | Live catalogue of providers and missions. |

`search_parameters` fuses two channels with Reciprocal Rank Fusion: dense embeddings
(MiniLM) for descriptions, and BM25 over `id + text` for exact tokens. That is why both
"magnetic field in GSE coordinates" and `BGSEc` land on the same product — dense retrieval
alone never sees instrument codes it was not trained on.

Results can be filtered by `provider`, SPASE `region`, or measurement type.

## Event catalogs

| Tool | What it does |
|---|---|
| `list_catalogs` | Browse 217 AMDA catalogs and timetables, filtered by type or region. |
| `get_catalog` | Inspect one: event count, columns, time filter, `where`, `sort_by`, pagination. |
| `get_events_timeseries` | Download a parameter across **every** event in one call. |
| `save_catalog` | Persist your own event set in standard speasy format. |

`get_catalog` does the filtering agent-side on purpose. Handing a raw AMDA catalog to the
model would burn millions of tokens on data it should never read; instead the tool applies
the time window, `where` clause, sort and column selection, and returns a capped sample
with a warning when results were truncated.

`get_events_timeseries` is what makes statistical work practical: one call gives you the
same parameter across hundreds of events, which is the input to superposed epoch analysis.

Catalogs saved with `save_catalog` use the standard speasy format, so they open directly in
SciQLop for visual checking.

## Analysis

| Tool | What it does |
|---|---|
| `run_python` | Sandboxed Python — speasy, PlasmaPy, NumPy, SciPy, Matplotlib, Astropy, geopack. |
| `plasma_beta` | β = nkT / (B²/2μ₀) |
| `gyrofrequency` | Ion and electron gyrofrequency |
| `debye_length` | Debye screening length |
| `alfven_speed` | V_A = B / √(μ₀ρ) |
| `inertial_length` | Ion and electron inertial length |
| `power_spectrum` | Welch PSD |

The PlasmaPy wrappers exist so simple quantities do not require a sandbox round-trip. Real
analysis happens in `run_python`, which provides helpers the agent relies on:

- `load_data(key)` — load a previously downloaded dataset without re-fetching
- `clean(array)` — mask CDF fill values (`|x| ≥ 1e30`, `±inf`) as NaN
- `export(name, array)` — surface compact statistics to the model instead of raw arrays
- `param_card(...)`, `document_method(...)` — render provenance in the UI

`export()` matters more than it looks: returning a raw array or a base64 PNG to the model
costs tens of thousands of tokens; returning min/max/mean/std/shape costs about two
hundred and is what the model actually reasons over.

## Literature

| Tool | What it does |
|---|---|
| `find_papers` | Search NASA ADS for papers on an event, parameter or method. |

Needs `ADS_API_TOKEN` (free). Used heavily by the `librarian` sub-agent.

ADS requires every bare word of a query, and a model writes many: a query that returns
fewer papers than asked is sent once more with its bare words made optional — fielded
terms (`author:`, `title:`), quoted phrases and negations stay required — restricted to
refereed astronomy papers and ranked by relevance. The exact hits come first; the result
says `relaxed: true` and gives the `relaxed_query`. A query with explicit `AND`/`OR`/`NOT`
is sent as written.

## Recipes

| Tool | What it does |
|---|---|
| `list_recipes` | Catalogue of the shipped scientific recipes. |
| `load_recipe` | Load a recipe's source and its citation. |
| `run_recipe` | Run a recipe as shipped on the session's data: inputs bound first, source verbatim, one optional `call`. |

`run_recipe` is the one that keeps a recipe's calibration intact: the agent binds the
inputs (`{"B_up": "load_data('b3gsm').values[m_up]"}`), the recipe's own `export()` calls
produce the numbers, and the run is recorded as a use of the recipe with its reference.
`load_recipe` remains the way to *read* one. See [Recipes and provenance](recipes.md).

## Delegation

`task` spawns a sub-agent with its own context window and a restricted tool set. It is the
one tool that is not in the registry — the agent loop intercepts it directly.

| Role | Purpose | Tools | Max turns |
|---|---|---|---|
| `parameter_hunter` | vague description → speasy IDs | search only | 4 |
| `data_analyst` | download, analyse, plot, detect | data + sandbox | 12 |
| `plasma_physicist` | PlasmaPy calculations, sanity checks | sandbox | 4 |
| `librarian` | multi-round ADS literature search | `find_papers` | 5 |

Delegation exists for context isolation, not parallelism: a parameter search that reads
forty candidate descriptions should not leave forty descriptions sitting in the main
conversation for the rest of the session.

The whitelist is enforced, not advisory — a role calling a tool outside its set gets an
error back naming what it may use, and the tool is never dispatched.
