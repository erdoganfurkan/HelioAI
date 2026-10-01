"""Derived-recipe tools — list, load and run scientific Python recipes.

Recipes live in data/recipes/ as .py files with a YAML comment header:
    # name: theta_bn
    # description: Compute the shock normal angle theta_Bn from upstream/downstream B.
    # inputs: B_up (nT vec), B_dn (nT vec)
    # outputs: theta_bn_deg

list_recipes() returns the catalogue; load_recipe(name) returns the source code;
run_recipe(name, inputs) executes it verbatim in the sandbox on the caller's inputs.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

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
# Sandbox helpers that ARE a published model of the quantity a recipe also computes.
# A run that called `mp_shue1998(...)` for the magnetopause did not bypass the
# pressure-balance recipe: it chose the empirical Shue model — with its reference — over
# the Chapman-Ferraro balance, which is a legitimate choice, not a hand-written copy.
# On the MMS1 live run the export `magnetopause_r_at_mms_Re` matched the recipe's
# "magnetopause" signature and the answer was flagged for a recipe it had no use for.
HELPER_ALTERNATIVES: dict[str, tuple[str, ...]] = {
    "pressure_balance": ("mp_shue1998(", "bs_jelinek2012("),
}

RECIPE_SIGNATURES: dict[str, tuple[str, ...]] = {
    "theta_bn": ("theta_bn",),
    "mvab": ("mvab", "minimum_variance", "lambda_min", "ratio_int_min"),
    "rankine_hugoniot": (
        # Not `compression_ratio`: |B_dn|/|B_up| is an ordinary output of the coplanarity
        # analysis too, and it accused a theta_bn run that had loaded and called its own
        # recipe. The names below belong to the jump conditions and to nothing else.
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
    """List all available derived recipes with their name, description and the call that runs them.

    Each entry carries `run_with`, the `run_recipe(...)` call with the recipe's own input
    names or functions, so a model can go from the catalogue straight to running one:
    loading a recipe first cost one LLM call per recipe, and the call is where a session
    pays — every one re-sends the whole context.

    Returns dict with 'recipes' list (sorted by name). Each entry has
    'name', 'description', 'inputs', 'outputs' (when present in header) and 'run_with'.
    Returns {"recipes": []} when the recipes directory does not exist.

    Example:
        >>> await list_recipes()
        {'recipes': [{'name': 'fill_values', 'description': '...', 'run_with': '...'},
                     {'name': 'mvab', ...}, {'name': 'rankine_hugoniot', ...}, ...]}
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
                entry["run_with"] = run_with(entry["name"], text)
                entries.append(entry)
            except OSError as exc:
                log.warning("recipe_read_error", path=str(path), error=str(exc))
        return {"recipes": entries}
    except Exception as e:
        return {"error": str(e)}


def _recipe_path(name: str) -> Path | None:
    """The file behind a recipe name, or None for a name that is not a shipped recipe —
    including one that tries to walk out of the recipes directory."""
    if not name or any(c in name for c in ("/", "\\", "..")):
        return None
    recipes_dir = settings.recipes.recipes_dir.resolve()
    candidate = (recipes_dir / f"{name}.py").resolve()
    if not candidate.is_relative_to(recipes_dir) or not candidate.is_file():
        return None
    return candidate


async def load_recipe(name: str) -> dict:
    """Load the source code of a named recipe.

    Args:
        name: Recipe name without .py extension (e.g. 'theta_bn').

    Returns dict with 'name' and 'code'. Returns {'error': ...} when not found
    or when the name contains path-traversal characters.

    Example:
        >>> await load_recipe("theta_bn")
        {'name': 'theta_bn', 'code': '# name: theta_bn\\n# description: Compute the shock...'}
    """
    try:
        path = _recipe_path(name)
        if path is None:
            return {"error": f"recipe {name!r} not found"}
        code = path.read_text(encoding="utf-8")
        meta = _parse_header(code)
        return {
            "name": meta.get("name", name),
            "code": code,
            "metadata": meta,
            "run_with": run_with(meta.get("name", name), code),
        }
    except Exception as e:
        return {"error": str(e)}


def _describe(name: str, code: str) -> dict:
    """What calling a recipe needs: its usage notes, its public signatures, its call."""
    tree = ast.parse(code)
    return {
        "usage": ast.get_docstring(tree) or "",
        "functions": _public_functions(tree),
        "run_with": run_with(name, code),
    }


def _public_functions(tree: ast.Module) -> list[dict]:
    """Signature and first docstring paragraph of each top-level public function.

    The first paragraph only: the rest of a recipe's docstrings is its calibration record
    (which shock, which database entry, which window moved the angle by how much), which
    matters to whoever edits the recipe, not to whoever calls it.
    """
    functions = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name.startswith("_"):
            continue
        signature = f"{node.name}({ast.unparse(node.args)})"
        if node.returns is not None:
            signature += f" -> {ast.unparse(node.returns)}"
        doc = (ast.get_docstring(node) or "").split("\n\n")[0].strip()
        functions.append({"signature": signature, "doc": doc})
    return functions


def _globals_read(code: str) -> list[str]:
    """The names a recipe reads with `globals().get`, literal or looped over.

    A regex on `globals().get("name")` missed `{k: globals().get(k) for k in ("t1", ...)}`,
    and `shock_timing_2sc` was announced as needing only `V_shock_rh`, the one optional
    input it reads by name — its five required ones were in the loop.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(code)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "globals"
            and node.args
        ):
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            names.add(arg.value)
        elif isinstance(arg, ast.Name):
            names |= _loop_constants(code, arg.id)
    return sorted(names)


def _loop_constants(code: str, var: str) -> set[str]:
    """String constants a comprehension or `for` binds to `var` from a literal tuple/list."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.comprehension | ast.For):
            if isinstance(node.target, ast.Name) and node.target.id == var:
                if isinstance(node.iter, ast.Tuple | ast.List):
                    found |= {
                        e.value
                        for e in node.iter.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    }
    return found


_PUBLIC_DEF = re.compile(r"^def\s+([A-Za-z]\w*)\s*\(", re.MULTILINE)


def run_with(name: str, code: str) -> str:
    """The one line that runs a recipe as shipped, read off its own source.

    A model that has just read a recipe's code is one paste away from running a copy of
    it in `run_python` — which is how a 54.85° θ_Bn came out of a 12-minute window the
    recipe would never have chosen. The line names the tool and, exactly, what to bind.

    A recipe declares its usual call in its header (`# run:`), because the names it reads
    are not a call: `rankine_hugoniot` reads eighteen, in three alternative bindings,
    and listed flat, alphabetically, they do not say which to bind. The other names it
    reads follow it. Without a declaration the line is derived: the
    variables read with `globals().get` for a script, the public functions for a library.

    Args:
        name: The recipe.
        code: Its source.

    Returns:
        The declared call, or a `run_recipe(...)` template with the recipe's own input
        names or functions.
    """
    inputs = _globals_read(code)
    declared = _parse_header(code).get("run")
    if declared:
        named = _declared_inputs(declared)
        others = [i for i in inputs if i not in named]
        if not named:
            return declared
        line = f"{declared} — replace each <...> with yours"
        if others:
            line += f"; other inputs it reads: {', '.join(others)} — its inputs say what each is"
        return line
    if inputs:
        bound = ", ".join(f"{i!r}: ..." for i in inputs)
        return (
            f"run_recipe({name!r}, inputs={{{bound}}}) — bind the inputs you have (each a "
            f"Python expression such as \"load_data('name')\" or a literal); the recipe's "
            f"source then runs verbatim on the session's data"
        )
    functions = [f for f in _PUBLIC_DEF.findall(code) if f != "export"]
    if functions:
        example = f"{functions[-1]}(...)"
        return (
            f"run_recipe({name!r}, inputs={{...}}, call={example!r}) — a library of functions "
            f"({', '.join(functions[:6])}); bind their arguments as inputs and name the call"
        )
    return (
        f"run_recipe({name!r}, inputs={{...}}) runs the recipe's source verbatim on the "
        f"session's data"
    )


def _declared_inputs(call: str) -> set[str]:
    """The input names of a declared `run_recipe(..., inputs={...})` call; empty when the
    declaration is not such a call (`fill_values` says to copy it instead)."""
    try:
        node = ast.parse(call, mode="eval").body
    except SyntaxError:
        return set()
    if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "run_recipe"):
        return set()
    for kw in node.keywords:
        if kw.arg == "inputs" and isinstance(kw.value, ast.Dict):
            return {k.value for k in kw.value.keys if isinstance(k, ast.Constant)}
    return set()


_INPUT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _bindings(inputs: dict) -> list[str]:
    """One assignment per input. A string is a Python expression evaluated in the
    sandbox — `load_data('b').values[:20]` — because a recipe's inputs are arrays
    the model cannot pass by value; anything else is a JSON literal, spelled as the
    Python literal it already is. A string that cannot be an expression — it does not
    parse, or it is a single bare name — is bound as the literal string it obviously
    is: a live run bound `{"units": "nT", "param_label": "|B| OMNI 1-min"}` as
    `param_label = (|B| OMNI 1-min)`, a SyntaxError and a lost turn, and nothing in a
    fresh script can ever be named `nT`."""
    lines = []
    for key, value in inputs.items():
        if not _INPUT_NAME.match(str(key)):
            raise ValueError(f"input {key!r} is not a valid Python name")
        rhs = value.strip() if isinstance(value, str) else repr(value)
        if not rhs:
            raise ValueError(f"input {key!r} has no value")
        if isinstance(value, str) and not _is_expression(rhs):
            rhs = repr(rhs)
        lines.append(f"{key} = ({rhs})")
    return lines


def _is_expression(text: str) -> bool:
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return False
    return not isinstance(tree.body, ast.Name)


def recipe_script(name: str, code: str, inputs: dict, call: str | None) -> str:
    """Assemble the script `run_recipe` executes: inputs, the recipe, the call.

    The recipe's source is inserted verbatim — not a function copied out of it, not a
    formula rewritten from memory — so its calibrated constants, its checks and its own
    `export()` calls run as shipped. `__name__` is set first so a recipe that guards a
    demo behind `if __name__ == "__main__":` runs its functions and not its demo, the way
    it would if imported. The call comes last, for the recipes that are a library of
    functions rather than a script.

    Args:
        name: The recipe.
        code: Its source, as `load_recipe` returns it.
        inputs: `{variable: expression | literal}` bound before the recipe.
        call: An expression evaluated after it, or None.

    Returns:
        The Python source, as it is written to the session's `code_N.py`.
    """
    parts = [
        f"# run_recipe: {name} — inputs, then the recipe verbatim; not __main__, so a demo",
        '# the recipe guards behind `if __name__ == "__main__":` stays off',
        '__name__ = "recipe"',
    ]
    parts += _bindings(inputs)
    parts += ["", f"# ── recipe {name} ──", code.rstrip("\n"), ""]
    if call and call.strip():
        parts += [
            "# ── call ──",
            f"_recipe_result = ({call.strip()})",
            "print(repr(_recipe_result))",
        ]
    return "\n".join(parts) + "\n"


async def run_recipe(
    name: str,
    inputs: dict | None = None,
    call: str | None = None,
    timeout: float = 60.0,
    _plot_dir: str | None = None,
    _run_idx: int | None = None,
    _no_net: bool = False,
) -> dict:
    """Run a shipped recipe on the session's data, without the model rewriting it.

    `load_recipe` hands the model the source, and the model then pastes a part of it
    into `run_python` — or reads the constants and rewrites the computation by hand,
    which is what the recipe check keeps catching. Here the recipe runs as shipped: the
    inputs are bound first, the source follows verbatim, and an optional call applies
    one of its functions. The exports are the recipe's own, the script written to the
    workspace is the one the export reproduces, and the run is recorded as a use of the
    recipe (`method_used`) with its reference.

    Args:
        name: Recipe name, as `list_recipes` lists it.
        inputs: `{variable: value}` bound before the recipe runs. A string is a Python
            expression evaluated in the sandbox (`"load_data('b').values[:20]"`); a
            number or list is used as the literal it is.
        call: An expression evaluated after the recipe, for a recipe that is a library
            of functions — `"rh_jump(n_u, n_d, V_u, V_d, B_u, B_d)"`; its value is
            printed. Not needed for a recipe whose run block reads its inputs itself.
        timeout: Sandbox time budget in seconds.
        _plot_dir: Session workspace, injected by the runtime.
        _run_idx: Script index in that workspace, injected by the runtime.
        _no_net: Deny the sandbox a network namespace, injected by the runtime.

    Returns:
        The `run_python` result — stdout, exports, figures, `code_path` — plus
        `recipe` (`name`, `reference`, `description`), `inputs` as bound, and a
        `method_used` card. When the run failed, or produced nothing at all — six of
        the recipes read their inputs with `globals().get` and do nothing, silently,
        when a name is bound wrong — it also carries `recipe_notice`: the recipe's
        usage, public signatures and `run_with`, how it is called without its source.
        `{"error": ...}` for an unknown recipe (with the names there are) or an input
        that is not a Python name (with the notice).
    """
    from helioai.tools.sandbox import run_python

    path = _recipe_path(name)
    if path is None:
        names = sorted(p.stem for p in settings.recipes.recipes_dir.glob("*.py"))
        return {
            "error": f"recipe {name!r} not found; call list_recipes for the names",
            "recipes": names,
        }
    code = path.read_text(encoding="utf-8")
    meta = _parse_header(code)
    bound = dict(inputs or {})
    try:
        script = recipe_script(name, code, bound, call)
    except ValueError as e:
        return {"error": str(e), "recipe_notice": _describe(meta.get("name", name), code)}

    result = await run_python(
        script, timeout=timeout, _plot_dir=_plot_dir, _run_idx=_run_idx, _no_net=_no_net
    )
    recipe = {
        "name": meta.get("name", name),
        "reference": meta.get("reference", ""),
        "description": meta.get("description", ""),
    }
    result["recipe"] = recipe
    result["inputs"] = bound
    produced = (
        result.get("exports") or result.get("figure_paths") or (result.get("stdout") or "").strip()
    )
    if "error" in result or not produced:
        result["recipe_notice"] = _describe(recipe["name"], code)
    if "error" not in result:
        result.setdefault("cards", []).append(
            {
                "kind": "method_used",
                "name": recipe["name"],
                "reference": recipe["reference"],
                "method": recipe["description"],
            }
        )
    return result
