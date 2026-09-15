"""Every number a recipe exports carries its unit, or is declared dimensionless.

The provenance ledger is compared unit-aware to the numbers the answer states: a
speed recorded without "km/s" cannot vouch for "V_shock = 579 km/s", and a ratio
recorded as "nT" would be contradicted by the same ratio quoted bare. So each
`export(name, data, units)` call in the shelf is held to two rules: the unit string is
one astropy parses, and a blank one is on the list of quantities that are genuinely
dimensionless — a ratio, a count, a correlation, a unit vector.
"""

from __future__ import annotations

import re

import astropy.units as u
import numpy as np
import pytest

from tests.recipes.conftest import RECIPE_NAMES, RECIPES_DIR

pytestmark = pytest.mark.recipes

# Exports that are dimensionless by nature. A new blank export must be added here on
# purpose, with the reason it has no unit.
DIMENSIONLESS: dict[str, str] = {
    "r": "density compression ratio",
    "B_ratio": "|B| compression ratio",
    "mom_residual": "momentum-flux residual, a fraction",
    "M_A": "Alfvén Mach number",
    "M_ms": "magnetosonic Mach number",
    "r_predicted": "compression ratio predicted from M_ms",
    "r_mismatch": "relative gap between r and r_predicted, a fraction",
    "compression_ratio": "|B_dn|/|B_up|",
    "shock_normal": "unit vector",
    "mvab_ratio_int_min": "eigenvalue ratio",
    "mvab_normal": "unit vector",
    "walen_slope": "regression slope of V on V_A, both km/s",
    "walen_R2": "coefficient of determination",
    "ht_residual": "normalised residual electric field",
    "epoch_n": "count of contributing events",
    "onset_index": "sample index",
    "n_particles": "count",
    "pad_counts": "counts per bin",
    "pad_anisotropy": "count ratio",
}

_EXPORT = re.compile(r'export\(\s*(?P<name>"[^"]+"|[A-Za-z_]\w*)\s*,\s*(?P<rest>[^\n]*)\)')
_UNIT_LITERAL = re.compile(r'"([^"]*)"\s*\)?\s*(#.*)?$')


def _export_calls(name: str) -> list[tuple[str, str]]:
    src = (RECIPES_DIR / f"{name}.py").read_text(encoding="utf-8")
    calls = []
    for line in src.splitlines():
        m = _EXPORT.search(line)
        if not m or line.lstrip().startswith("#") or "def export" in line:
            continue
        calls.append((m.group("name"), line.strip()))
    return calls


@pytest.mark.parametrize("name", RECIPE_NAMES)
def test_every_export_call_passes_a_unit_argument(name):
    """The third argument is present on every call — blank on purpose, never omitted."""
    for export_name, line in _export_calls(name):
        args = line[line.index("(") + 1 :]
        depth, commas = 0, 0
        for ch in args:
            depth += {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}.get(ch, 0)
            if ch == "," and depth == 0:
                commas += 1
        assert commas >= 2, f"{name}: {line!r} exports {export_name} without a unit argument"


def _events(n=6):
    from types import SimpleNamespace

    out = []
    for i in range(n):
        t = np.arange(0, 3600, 60).astype("timedelta64[s]") + np.datetime64(
            f"2015-03-{17 + i:02d}T00:00:00"
        )
        y = 5.0 + np.sin(np.linspace(0, np.pi, t.size)) * (i + 1)
        out.append(SimpleNamespace(time=t, values=y, start=str(t[0]), stop=str(t[-1]), units="nT"))
    return out


def _sep_flux(onset_index=400, n=900):
    from types import SimpleNamespace

    rng = np.random.default_rng(1)
    t = np.datetime64("2021-10-28T12:00:00") + np.arange(n).astype("timedelta64[m]")
    x = rng.poisson(20, n).astype(float)
    x[onset_index:] += np.linspace(0, 300, n - onset_index)
    return SimpleNamespace(time=t, values=x, units="1/(cm2 s sr MeV)")


def _shock_series(n=40):
    from types import SimpleNamespace

    bx, bz = 10 * np.cos(np.radians(60)), 10 * np.sin(np.radians(60))
    t = np.datetime64("2015-03-17T03:50:00", "s") + np.arange(n) * np.timedelta64(30, "s")
    values = np.tile([bx, 0.0, bz], (n, 1))
    values[n // 2 :, 2] *= 2.5
    return SimpleNamespace(time=t, values=values, units="nT")


def _exporting_runs(recipe) -> dict[str, dict]:
    """Every recipe driven down the path that exports, with inputs that have units."""
    rng = np.random.default_rng(0)
    cloud = rng.normal(0, 1, (300, 3)) * np.array([10.0, 3.0, 0.3])
    shock = _shock_series()
    b_rot = np.column_stack(
        [
            5 + 3 * np.cos(np.linspace(0, np.pi, 200)),
            np.sin(np.linspace(0, np.pi, 200)),
            1 + 0 * np.linspace(0, 1, 200),
        ]
    )
    n_cm3 = np.full(200, 5.0)
    v_a = b_rot * 1e-9 / np.sqrt(4e-7 * np.pi * n_cm3[:, None] * 1e6 * 1.6726e-27) / 1e3
    runs = {
        "theta_bn": recipe("theta_bn", B_up=shock.values[:20], B_dn=shock.values[20:]),
        "mvab": recipe("mvab", B=cloud),
        "walen_test": recipe(
            "walen_test", V=v_a + np.array([-400.0, 30.0, -10.0]), B=b_rot, n_cm3=n_cm3
        ),
        "superposed_epoch": recipe("superposed_epoch", events=_events(), units="nT"),
        "sep_onset_poisson_cusum": recipe(
            "sep_onset_poisson_cusum", flux=_sep_flux(), units="1/(cm2 s sr MeV)"
        ),
        "pressure_balance": recipe("pressure_balance", n_sw=5.0, V_sw=400.0, B_sw=5.0),
        "pitch_angle_dist": recipe(
            "pitch_angle_dist", V=rng.normal(0, 1, (2000, 3)), B=np.array([0.0, 0.0, 10.0])
        ),
    }
    rh = recipe("rankine_hugoniot")
    rh.namespace["rh_jump"](17.43, 45.12, 411.3, 514.1, 10.0, 25.27, 8.34, 45.0)
    runs["rankine_hugoniot"] = rh
    timing = recipe("shock_timing_2sc")
    t1 = np.datetime64("2015-03-17T04:00:00")
    out = timing.namespace["timing_2sc"](
        t1,
        t1 + np.timedelta64(2000, "s"),
        np.array([1.5e6, 2e5, 0.0]),
        np.array([2.5e6, 2e5, 3e4]),
        np.array([1.0, 0.0, 0.0]),
    )
    timing.namespace["export"]("lag_s", np.array([out["lag_s"]]), units="s")
    timing.namespace["export"](
        "shock_speed_timing", np.array([out["shock_speed_km_s"]]), units="km/s"
    )
    runs["shock_timing_2sc"] = timing
    return runs


def test_every_recorded_unit_parses_and_every_blank_one_is_declared_dimensionless(recipe):
    """What the ledger will hold, recipe by recipe: a unit astropy understands, or a
    blank the shelf declares dimensionless by name. Every exporting recipe must appear."""
    runs = _exporting_runs(recipe)
    silent = [name for name, run in runs.items() if not run.exports]
    assert silent == [], f"these recipes exported nothing on their exporting path: {silent}"
    for name, run in runs.items():
        for key, entry in run.exports.items():
            unit = entry["units"]
            if unit == "":
                assert key in DIMENSIONLESS, (
                    f"{name} exports {key} without a unit; add it to DIMENSIONLESS with a "
                    "reason, or pass one"
                )
            else:
                u.Unit(unit)


def test_a_bare_unit_string_on_the_shelf_would_be_caught():
    with pytest.raises(ValueError):
        u.Unit("not-a-unit!")
