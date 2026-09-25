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
        "superposed_epoch": recipe("superposed_epoch", events=_events()),
        "sep_onset_poisson_cusum": recipe("sep_onset_poisson_cusum", flux=_sep_flux()),
        "pressure_balance": recipe("pressure_balance", n_sw=5.0, V_sw=400.0, B_sw=5.0),
        "pitch_angle_dist": recipe(
            "pitch_angle_dist", V=rng.normal(0, 1, (2000, 3)), B=np.array([0.0, 0.0, 10.0])
        ),
    }
    rh = recipe("rankine_hugoniot")
    rh.namespace["rh_jump"](17.43, 45.12, 411.3, 514.1, 10.0, 25.27, 8.34, 45.0)
    runs["rankine_hugoniot"] = rh
    t1 = np.datetime64("2015-03-17T04:00:00")
    runs["shock_timing_2sc"] = recipe(
        "shock_timing_2sc",
        t1=t1,
        t2=t1 + np.timedelta64(2000, "s"),
        r1=np.array([1.5e6, 2e5, 0.0]),
        r2=np.array([2.5e6, 2e5, 3e4]),
        n_hat=np.array([1.0, 0.0, 0.0]),
    )
    return runs


# The physical dimension each dimensioned export must carry. `walen_slope` in "nT" or
# `P_dyn_nPa` in "km/s" would parse; they would still be wrong provenance.
EXPECTED_DIMENSION: dict[str, str] = {
    "theta_bn": "angle",
    "theta_bn_sampling_std_deg": "angle",
    "theta_bn_window_spread_deg": "angle",
    "normal_spread_deg": "angle",
    "B_up_mean_nT": "magnetic flux density",
    "B_dn_mean_nT": "magnetic flux density",
    "B_up_mag_nT": "magnetic flux density",
    "B_dn_mag_nT": "magnetic flux density",
    "Bn_std_nT": "magnetic flux density",
    "mvab_lambda_min": "nT2",
    "mvab_dphi_min_int": "angle",
    "mvab_dphi_min_max": "angle",
    "mvab_dBn": "magnetic flux density",
    "V_HT": "speed",
    "V_shock": "speed",
    "V_A": "speed",
    "c_s": "speed",
    "U_upstream": "speed",
    "r_mp_RE": "length",
    "P_dyn_nPa": "pressure",
    "P_applied_nPa": "pressure",
    "P_mag_sw_nPa": "pressure",
    "P_total_sw_nPa": "pressure",
    "B_msp_ref_nT": "magnetic flux density",
    "lag_s": "time",
    "shock_speed_timing": "speed",
    "along_normal_separation": "length",
    "transverse_separation": "length",
    "pitch_angles_median_deg": "angle",
    "pitch_angles_mean_deg": "angle",
    "pad_bin_edges_deg": "angle",
}

# Exports whose unit is whatever the input data carried: checked against the input.
INHERITED_FROM_INPUT: dict[str, str] = {
    "epoch_median": "nT",
    "epoch_q25": "nT",
    "epoch_q75": "nT",
    "epoch_ci_low": "nT",
    "epoch_ci_high": "nT",
    "cusum": "1/(cm2 s sr MeV)",
}


def _dimension_matches(unit: str, expected: str) -> bool:
    if expected == "nT2":
        return u.Unit(unit).is_equivalent(u.nT**2)
    return u.get_physical_type(u.Unit(unit)) == expected


def test_every_recorded_unit_has_the_dimension_of_the_quantity_or_is_declared_dimensionless(
    recipe,
):
    """What the ledger will hold, recipe by recipe: the dimension of the quantity named,
    the unit of the data it was computed from, or a blank the shelf declares
    dimensionless by name. Every exporting recipe must appear, and every export must be
    accounted for by one of the three tables."""
    runs = _exporting_runs(recipe)
    silent = [name for name, run in runs.items() if not run.exports]
    assert silent == [], f"these recipes exported nothing on their exporting path: {silent}"
    for name, run in runs.items():
        for key, entry in run.exports.items():
            unit = entry["units"]
            if key in DIMENSIONLESS:
                assert unit == "", f"{name}.{key} is dimensionless but recorded as {unit!r}"
            elif key in EXPECTED_DIMENSION:
                assert _dimension_matches(unit, EXPECTED_DIMENSION[key]), (
                    f"{name}.{key} recorded as {unit!r}, expected a {EXPECTED_DIMENSION[key]}"
                )
            elif key in INHERITED_FROM_INPUT:
                assert u.Unit(unit) == u.Unit(INHERITED_FROM_INPUT[key]), (
                    f"{name}.{key} recorded as {unit!r}, the data carried "
                    f"{INHERITED_FROM_INPUT[key]!r}"
                )
            else:
                raise AssertionError(
                    f"{name} exports {key}; say what it is in DIMENSIONLESS, "
                    "EXPECTED_DIMENSION or INHERITED_FROM_INPUT"
                )


def test_the_zscore_cusum_is_recorded_dimensionless_whatever_the_flux_carries(recipe):
    run = recipe("sep_onset_poisson_cusum", flux=_sep_flux(), method="zscore")
    assert run.exports["cusum"]["units"] == ""


def test_a_bare_unit_string_on_the_shelf_would_be_caught():
    with pytest.raises(ValueError):
        u.Unit("not-a-unit!")


def test_the_recipe_check_accepts_every_recipes_own_exports(recipe):
    """The lead's recipe check reads a recipe's `# outputs:` header and its public
    functions off the `load_recipe` result, and what the run exported off the ledger.
    Each corrected recipe renamed or added exports; a session that loaded a recipe,
    pasted it whole into run_python and exported exactly what the recipe exports must
    not be accused of using it shallowly or of never calling it. Through run_recipe the
    run is exempt altogether."""
    import json

    from helioai.core.llm.base import Message, ToolCall
    from helioai.core.tool_exec import _flag_recipe_bypass
    from helioai.tools.recipes import _parse_header

    for name, run in _exporting_runs(recipe).items():
        code = (RECIPES_DIR / f"{name}.py").read_text(encoding="utf-8")
        loaded = ToolCall(id="l", name="load_recipe", arguments={"name": name})
        pasted = ToolCall(id="p", name="run_python", arguments={"code": "B = 1\n" + code})
        history = [
            Message(role="user", content="q"),
            Message(role="assistant", content="", tool_calls=[loaded]),
            Message(
                role="tool",
                tool_call_id="l",
                name="load_recipe",
                content=json.dumps({"name": name, "code": code, "metadata": _parse_header(code)}),
            ),
            Message(role="assistant", content="", tool_calls=[pasted]),
            Message(role="tool", tool_call_id="p", name="run_python", content="{}"),
        ]
        exported = [
            {"kind": "exports", "values": {k: {"mean": 1.0} for k in run.exports}},
        ]
        _, flags = _flag_recipe_bypass("done", history, exported)
        assert flags == [], f"{name}: {flags}"

        ran = ToolCall(id="r", name="run_recipe", arguments={"name": name})
        direct = [
            Message(role="user", content="q"),
            Message(role="assistant", content="", tool_calls=[ran]),
            Message(role="tool", tool_call_id="r", name="run_recipe", content="{}"),
        ]
        _, flags = _flag_recipe_bypass("done", direct, exported)
        assert flags == [], f"{name} via run_recipe: {flags}"
