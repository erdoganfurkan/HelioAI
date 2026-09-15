from __future__ import annotations

import pytest

from helioai.tools.sandbox_helpers import mp_shue1998

pytestmark = pytest.mark.recipes

OLD_QUIET_STANDOFF_RE = 9.709130547235814


def _pressure_balance_namespace(recipe):
    return recipe("pressure_balance", **{"__name__": "recipe"}).namespace


def test_derived_reference_field_uses_dipole_and_chapman_ferraro_factor(recipe):
    mp_standoff = _pressure_balance_namespace(recipe)["mp_standoff"]

    result = mp_standoff(5.0, 400.0, 5.0)

    assert result["B_msp_ref_nT"] == pytest.approx(61.6, abs=0.5)


def test_quiet_standoff_is_larger_than_the_old_round_reference(recipe):
    mp_standoff = _pressure_balance_namespace(recipe)["mp_standoff"]

    derived = mp_standoff(5.0, 400.0, 5.0)
    old_reference = mp_standoff(5.0, 400.0, 5.0, B_msp_ref=50.0)

    assert 9.5 <= derived["r_mp_RE"] <= 11.5
    assert derived["r_mp_RE"] / old_reference["r_mp_RE"] == pytest.approx(
        (61.6 / 50.0) ** (1 / 3), rel=0.01
    )


def test_explicit_reference_field_preserves_the_old_quiet_value(recipe):
    mp_standoff = _pressure_balance_namespace(recipe)["mp_standoff"]

    result = mp_standoff(5.0, 400.0, 5.0, B_msp_ref=50.0)

    assert result["r_mp_RE"] == pytest.approx(OLD_QUIET_STANDOFF_RE, rel=1e-12)
    assert result["B_msp_ref_nT"] == pytest.approx(50.0)


def test_result_flags_that_imf_bz_is_not_in_the_pressure_balance(recipe):
    mp_standoff = _pressure_balance_namespace(recipe)["mp_standoff"]

    result = mp_standoff(5.0, 400.0, 5.0)

    assert result["bz_ignored"] is True
    assert "mp_shue1998" in result["note"]


def test_run_block_exports_bound_inputs_with_units(recipe):
    run = recipe(
        "pressure_balance",
        n_sw=5.0,
        V_sw=400.0,
        B_sw=5.0,
        **{"__name__": "recipe"},
    )

    assert {name: export["units"] for name, export in run.exports.items()} == {
        "r_mp_RE": "R_earth",
        "P_dyn_nPa": "nPa",
        "P_mag_sw_nPa": "nPa",
        "P_total_sw_nPa": "nPa",
        "B_msp_ref_nT": "nT",
    }


def test_run_block_does_not_export_without_bound_inputs(recipe):
    run = recipe("pressure_balance", **{"__name__": "recipe"})

    assert run.exports == {}


def test_pressure_balance_standoff_is_close_to_shue_for_north_south_neutral_imf(recipe):
    mp_standoff = _pressure_balance_namespace(recipe)["mp_standoff"]

    result = mp_standoff(5.0, 400.0, 5.0)
    _, shue_r = mp_shue1998(result["P_dyn_nPa"], 0.0, theta_deg=0.0)

    # The 15% band is a judgment-call sanity check, not a calibration.
    assert result["r_mp_RE"] == pytest.approx(float(shue_r[0]), rel=0.15)
