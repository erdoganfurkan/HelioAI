"""The call a recipe declares in its header (`# run:`) runs as written.

`list_recipes`, `load_recipe` and `recipe_notice` all hand the model that line as the
way to run the recipe, and the model copies it, replacing each `<...>`. A declared call
that binds the wrong names, or an array where a series is expected, would make every
session start with a failed run — so each one is executed here: its placeholders are
replaced with stand-ins, each input is evaluated as `run_recipe` binds it (a Python
expression, in order, in the sandbox namespace, `load_data` included), and the recipe
must export. The stand-in series carry gaps, as `load_data` series do: mvab and
walen_test refuse a NaN by design, so their declared calls must drop the gaps themselves.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace

import numpy as np
import pytest

from helioai.tools.recipes import _parse_header
from tests.recipes.conftest import RECIPE_NAMES, RECIPES_DIR, _sandbox_like_namespace

pytestmark = pytest.mark.recipes

SHOCK = "2015-03-17T04:00:00"


def _step(before, after, cadence_s, cols=None, minutes=60):
    t0 = np.datetime64(SHOCK, "s")
    n = int(minutes * 60 / cadence_s) + 1
    t = t0 + (np.arange(n) - n // 2) * np.timedelta64(cadence_s, "s")
    shape = (n, cols) if cols else (n, 1)
    values = np.where((t > t0)[:, None], after, before) * np.ones(shape)
    values[3] = np.nan
    return SimpleNamespace(time=t, values=values, units="")


def _alfvenic(n=200):
    b = np.column_stack(
        [5 + 3 * np.cos(np.linspace(0, np.pi, n)), np.sin(np.linspace(0, np.pi, n)), np.ones(n)]
    )
    t = np.datetime64(SHOCK, "s") + np.arange(n) * np.timedelta64(3, "s")
    v_a = b * 1e-9 / np.sqrt(4e-7 * np.pi * 5.0 * 1e6 * 1.6726e-27) / 1e3
    v = v_a + np.array([-400.0, 30.0, -10.0])
    density = np.full((n, 1), 5.0)
    v[7], b[40], density[90] = np.nan, np.nan, np.nan
    return (
        SimpleNamespace(time=t, values=v),
        SimpleNamespace(time=t, values=b),
        SimpleNamespace(time=t, values=density),
    )


def _events(n=6):
    out = []
    for i in range(n):
        t = np.arange(0, 3600, 60).astype("timedelta64[s]") + np.datetime64(
            f"2015-03-{17 + i:02d}T00:00:00"
        )
        y = 5.0 + np.sin(np.linspace(0, np.pi, t.size)) * (i + 1)
        out.append(SimpleNamespace(time=t, values=y, start=str(t[0]), stop=str(t[-1]), units="nT"))
    return out


def _flux():
    rng = np.random.default_rng(1)
    t = np.datetime64("2021-10-28T12:00:00") + np.arange(900).astype("timedelta64[m]")
    x = rng.poisson(20, 900).astype(float)
    x[400:] += np.linspace(0, 300, 500)
    return SimpleNamespace(time=t, values=x, units="1/(cm2 s sr MeV)")


_v, _b, _n = _alfvenic()
_rng = np.random.default_rng(0)
_particles = SimpleNamespace(time=_b.time, values=_rng.normal(0, 1, (200, 3)))
_particles.values[11] = np.nan
_cloud = _rng.normal(0, 1, (300, 3)) * [10, 3, 0.3]
_cloud[20] = np.nan

# `<placeholder>` -> the text that replaces it, and the datasets load_data answers.
STAND_INS: dict[str, tuple[dict[str, str], dict[str, object]]] = {
    "mvab": (
        {"<b>": "b"},
        {"b": SimpleNamespace(values=_cloud)},
    ),
    "pitch_angle_dist": ({"<v>": "v", "<b>": "b"}, {"v": _particles, "b": _b}),
    "pressure_balance": (
        {"<density, cm-3>": "5.0", "<speed, km/s>": "400.0", "<|B|, nT>": "5.0"},
        {},
    ),
    "rankine_hugoniot": (
        {"<n>": "n", "<v>": "v", "<b>": "b", "<crossing time>": SHOCK},
        {
            "n": _step(17.43, 45.12, 60),
            "v": _step(411.3, 514.1, 60),
            "b": _step(np.array([6.0, 0.0, 8.0]), np.array([6.0, 0.0, 8.0]) * 2.527, 3, cols=3),
        },
    ),
    "sep_onset_poisson_cusum": ({"<flux>": "flux"}, {"flux": _flux()}),
    "shock_timing_2sc": (
        {
            "<crossing at 1>": SHOCK,
            "<crossing at 2>": "2015-03-17T04:33:20",
            "<position of 1 at t1, km>": "np.array([1.5e6, 2e5, 0.0])",
            "<position of 2 at t2, km>": "np.array([2.5e6, 2e5, 3e4])",
            "<shock normal from theta_bn or mvab>": "np.array([1.0, 0.0, 0.0])",
        },
        {},
    ),
    "superposed_epoch": ({"<param>_events": "b_events"}, {"b_events": _events()}),
    "theta_bn": (
        {"<b>": "b", "<crossing time>": SHOCK},
        {"b": _step(np.array([5.0, 0.0, 8.66]), np.array([5.0, 0.0, 21.65]), 3, cols=3)},
    ),
    "walen_test": ({"<v>": "v", "<b>": "b", "<n>": "n"}, {"v": _v, "b": _b, "n": _n}),
}
# solar_mach asks the network for ephemerides; fill_values declares a copy, not a call.
NOT_EXECUTED = {"solar_mach", "fill_values"}


def _declared(name: str) -> ast.Call | None:
    run = _parse_header((RECIPES_DIR / f"{name}.py").read_text(encoding="utf-8")).get("run")
    if not run:
        return None
    try:
        node = ast.parse(run, mode="eval").body
    except SyntaxError:
        return None
    return node if isinstance(node, ast.Call) and node.func.id == "run_recipe" else None


def test_every_recipe_declares_how_it_is_run():
    for name in RECIPE_NAMES:
        header = _parse_header((RECIPES_DIR / f"{name}.py").read_text(encoding="utf-8"))
        assert header.get("run"), f"{name}: no `# run:` line in its header"
        if name != "fill_values":
            call = _declared(name)
            assert call is not None and ast.literal_eval(call.args[0]) == name, name
    assert set(STAND_INS) | NOT_EXECUTED == set(RECIPE_NAMES), "a recipe without a stand-in"


@pytest.mark.parametrize("name", sorted(STAND_INS))
def test_the_declared_call_runs_and_exports(name, recipe, tmp_path):
    call = _declared(name)
    inputs = ast.literal_eval(next(kw.value for kw in call.keywords if kw.arg == "inputs"))
    replace, datasets = STAND_INS[name]
    ns = _sandbox_like_namespace({"datasets": datasets, "_tmp": str(tmp_path)}, {}, [])
    bound = {}
    for key, value in inputs.items():
        if isinstance(value, str):
            for placeholder, text in replace.items():
                value = value.replace(placeholder, text)
            assert "<" not in value, f"{name}.{key}: placeholder without a stand-in: {value}"
            try:
                value = eval(value, ns)  # noqa: S307 — the way run_recipe binds an input
            except (NameError, SyntaxError):
                pass
        bound[key] = ns[key] = value

    run = recipe(name, **bound)

    assert run.exports, f"{name}: the declared call exported nothing"
