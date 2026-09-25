"""rankine_hugoniot runs through `run_recipe` like every other script-shaped recipe.

It used to be a library only: nothing ran unless the caller named a `call`, and the
prompts told the model to paste the source and write the calls itself — the path the
recipe checks were built to police. Its run block now reads the series and the shock
time (or given windows, or the means), picks the windows the recipe calibrated, and
applies `rh_jump`. The numbers are those of the 2015-03-17 reference means the recipe's
own self-check asserts.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.recipes

SHOCK = np.datetime64("2015-03-17T04:45:00", "s")


def _series(before, after, cols=1):
    t = SHOCK + (np.arange(61) - 30) * np.timedelta64(60, "s")
    values = np.where((t > SHOCK)[:, None], after, before) * np.ones((61, cols))
    return SimpleNamespace(time=t, values=values)


def _swe_shaped():
    field = _series(np.array([6.0, 0.0, 8.0]), np.array([6.0, 0.0, 8.0]) * 2.527, cols=3)
    return {
        "density": _series(17.43, 45.12),
        "speed": _series(411.3, 514.1),
        "B": field,
        "temperature": _series(8.34, 45.0),
    }


def test_series_and_a_shock_time_run_the_jump_conditions(recipe):
    run = recipe("rankine_hugoniot", **_swe_shaped(), shock_time=SHOCK)
    assert run.value("r").item() == pytest.approx(2.59, abs=0.01)
    assert run.value("V_shock").item() == pytest.approx(579, abs=5)
    assert run.value("M_A").item() == pytest.approx(3.20, abs=0.15)
    assert run.value("n_upstream").item() == pytest.approx(17.43)
    assert run.exports["n_downstream"]["units"] == "cm-3"
    assert run.value("B_downstream").item() == pytest.approx(25.27, abs=0.01)
    assert run.exports["T_upstream"]["units"] == "eV"


def test_given_windows_replace_the_derived_ones(recipe):
    inputs = _swe_shaped()
    del inputs["temperature"]
    run = recipe(
        "rankine_hugoniot",
        **inputs,
        upstream=("2015-03-17T04:10:00", "2015-03-17T04:40:00"),
        downstream=("2015-03-17T04:50:00", "2015-03-17T05:10:00"),
    )
    assert run.value("r").item() == pytest.approx(2.59, abs=0.01)
    assert "T_upstream" not in run.exports, "no temperature bound, none exported"


def test_the_means_alone_are_enough(recipe):
    run = recipe(
        "rankine_hugoniot",
        n_u=17.43,
        n_d=45.12,
        V_u=411.3,
        V_d=514.1,
        B_u=10.0,
        B_d=25.27,
        T_u=8.34,
        T_d=45.0,
    )
    assert run.value("V_shock").item() == pytest.approx(579, abs=5)
    assert "n_upstream" not in run.exports, "no windows were taken"


def test_a_partial_binding_says_what_is_missing(recipe):
    """Six recipes did nothing, silently, on a wrong binding; this one refuses by name."""
    with pytest.raises(ValueError, match=r"bound \['density', 'speed'\].*shock_time"):
        recipe("rankine_hugoniot", density=_series(17.43, 45.12), speed=_series(411.3, 514.1))


def test_with_nothing_bound_nothing_is_exported(recipe):
    assert recipe("rankine_hugoniot").exports == {}


def test_run_with_gives_the_usual_call_then_the_other_inputs():
    """The header's `# run:` call comes first — the series and the shock time — and the
    alternatives (given windows, the means) and options are named after it, not mixed in
    alphabetically with it: eighteen flat names sent the model to read the source."""
    from helioai.config import _PKG_RECIPES
    from helioai.tools.recipes import run_with

    line = run_with("rankine_hugoniot", (_PKG_RECIPES / "rankine_hugoniot.py").read_text("utf-8"))
    call, other = line.split(" — ", 1)
    for name in ('"density":', '"speed":', '"B":', '"shock_time":'):
        assert name in call, name
    for name in ("upstream", "downstream", "normal", "temperature", "n_u", "T_d"):
        assert name in other.split("other inputs it reads:")[1], name
    timing = run_with("shock_timing_2sc", (_PKG_RECIPES / "shock_timing_2sc.py").read_text("utf-8"))
    for name in ('"t1":', '"t2":', '"r1":', '"r2":', '"n_hat":'):
        assert name in timing, f"read in a loop, still announced: {name}"
    assert "V_shock_rh" in timing.split("other inputs it reads:")[1]
