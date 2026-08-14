# Examples

Runnable notebooks. Outputs are stripped on purpose — run them and produce your own.

| Notebook | What it shows |
|---|---|
| [`01_jupyter_tour.ipynb`](01_jupyter_tour.ipynb) | Guided tour: asking questions, inline figures, direct PlasmaPy calls, event catalogs, session history, export |
| [`02_stpatrick_storm_2015.ipynb`](02_stpatrick_storm_2015.ipynb) | A worked scientific case — the 17 March 2015 storm from parameter discovery to plasma regimes |
| [`verify_reference_values.py`](verify_reference_values.py) | Re-measures every reference value quoted in notebook 02, straight from the archive. No LLM, no API key |

## Prerequisites

1. **Install** — `pip install helioai-agent`, or `uv sync` from a clone.
2. **One LLM provider key** in `.env`. See the
   [installation guide](https://erdoganfurkan.github.io/HelioAI/installation/).
   Data access itself needs no credentials.
3. **Build the parameter index once** — `helioai index` (~10 minutes, 83 000 products).

Then:

```bash
uv run jupyter lab examples/
```

## What to expect

These notebooks call a language model and download real data, so they are not
deterministic and they are not fast. The agent may resolve a parameter differently between
runs, and wall time is dominated by downloads — expect a few minutes for the storm
notebook.

That is also why they carry no committed outputs: an output cell would be a snapshot of one
particular run, and stale snapshots are worse than none.

Notebook 02 quotes reference values for its event, and those *are* deterministic — they come
from the archive, not from the model. `verify_reference_values.py` re-measures all of them
and needs neither an API key nor the index, so you can check the notebook's numbers before
spending a single token on it.

## Why these are not in the documentation site

They need an API key to execute, so CI cannot run them, and a notebook rendered without
outputs makes poor reading. The [documentation](https://erdoganfurkan.github.io/HelioAI)
carries worked transcripts in prose instead; these files are here to be *run*.
