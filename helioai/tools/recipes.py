"""Derived-recipe tools — list and load scientific Python recipes.

Recipes live in data/recipes/ as .py files with a YAML comment header:
    # name: theta_bn
    # description: Compute the shock normal angle theta_Bn from upstream/downstream B.
    # inputs: B_up (nT vec), B_dn (nT vec)
    # outputs: theta_bn_deg

list_recipes() returns the catalogue; load_recipe(name) returns the source code.
"""

from __future__ import annotations

import logging

from helioai.config import settings

log = logging.getLogger(__name__)

# Curated, not parsed from headers: a hand-written computation names its export()
# keys freely (a Rankine-Hugoniot run on the St Patrick 2015 shock exported
# "density_compression_ratio" / "predicted_RH_compression", not the recipe's own
# "r" / "r_predicted") — matching the recipe's literal export names would miss the
# exact case this table exists to catch. Substrings instead, chosen from what a
# calculation in this domain is actually called, whoever writes it. False positives
# cost nothing here (see sub_agents._flag_recipe_bypass — it annotates, never blocks);
# false negatives just mean one more recipe worth adding a substring for.
RECIPE_SIGNATURES: dict[str, tuple[str, ...]] = {
    "theta_bn": ("theta_bn",),
    "mvab": ("mvab", "minimum_variance", "lambda_min", "ratio_int_min"),
    "rankine_hugoniot": (
        "compression_ratio",
        "alfven_mach",
        "predicted_rh",
        "predicted_compression",
        "rh_compression",
        "spacecraft_frame",
    ),
    "shock_timing_2sc": (
        "shock_speed_timing",
        "along_normal",
        "transverse_separation",
        "lag_s",
    ),
    "walen_test": ("walen",),
    "pressure_balance": ("standoff", "p_dyn", "magnetopause", "r_mp"),
    "pitch_angle_dist": ("pitch_angle",),
    "superposed_epoch": ("epoch_median", "epoch_q25", "epoch_q75", "superposed_epoch"),
    "sep_onset_poisson_cusum": ("sep_onset", "cusum", "poisson_cusum"),
}


def _parse_header(text: str) -> dict[str, str]:
    """Extract key: value pairs from leading `# key: value` comment lines."""
    meta: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            break
        content = stripped[1:].strip()
        if ":" in content:
            key, _, val = content.partition(":")
            meta[key.strip()] = val.strip()
    return meta


async def list_recipes() -> dict:
    """List all available derived recipes with their name and description.

    Returns dict with 'recipes' list (sorted by name). Each entry has
    'name', 'description', 'inputs', 'outputs' (when present in header).
    Returns {"recipes": []} when the recipes directory does not exist.
    """
    try:
        recipes_dir = settings.recipes.recipes_dir
        if not recipes_dir.exists():
            return {"recipes": []}
        entries = []
        for path in sorted(recipes_dir.glob("*.py")):
            try:
                text = path.read_text(encoding="utf-8")
                meta = _parse_header(text)
                entry = {"name": meta.get("name", path.stem)}
                for field in ("description", "inputs", "outputs"):
                    if field in meta:
                        entry[field] = meta[field]
                entries.append(entry)
            except OSError as exc:
                log.warning("recipe_read_error", path=str(path), error=str(exc))
        return {"recipes": entries}
    except Exception as e:
        return {"error": str(e)}


async def load_recipe(name: str) -> dict:
    """Load the source code of a named recipe.

    Args:
        name: Recipe name without .py extension (e.g. 'theta_bn').

    Returns dict with 'name' and 'code'. Returns {'error': ...} when not found
    or when the name contains path-traversal characters.
    """
    try:
        if not name or any(c in name for c in ("/", "\\", "..")):
            return {"error": f"invalid recipe name: {name!r}"}
        recipes_dir = settings.recipes.recipes_dir.resolve()
        candidate = (recipes_dir / f"{name}.py").resolve()
        if not candidate.is_relative_to(recipes_dir):
            return {"error": f"recipe {name!r} not found"}
        if not candidate.is_file():
            return {"error": f"recipe {name!r} not found"}
        code = candidate.read_text(encoding="utf-8")
        meta = _parse_header(code)
        return {
            "name": meta.get("name", name),
            "code": code,
            "metadata": meta,
        }
    except Exception as e:
        return {"error": str(e)}
