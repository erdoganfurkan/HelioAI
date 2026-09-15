from __future__ import annotations

import pytest

from helioai.tools.sandbox_helpers import mp_shue1998

pytestmark = pytest.mark.recipes

OLD_QUIET_STANDOFF_RE = 9.709130547235814
QUIET_STANDOFF_RE = 10.295202738406955
IGRF13_B0_NT = 29_806.0
DERIVED_B_MSP_REF_NT = 59.612


def _pressure_balance_namespace(recipe):
    return recipe("pressure_balance", **{"__name__": "recipe"}).namespace


def test_derived_reference_field_uses_dipole_and_chapman_ferraro_factor(recipe):
    mp_standoff = _pressure_balance_namespace(recipe)["mp_standoff"]

    result = mp_standoff(5.0, 400.0, 5.0)

    assert result["B_msp_ref_nT"] == pytest.approx(DERIVED_B_MSP_REF_NT, abs=0.5)


def test_quiet_standoff_is_larger_than_the_old_round_reference(recipe):
    mp_standoff = _pressure_balance_namespace(recipe)["mp_standoff"]

    derived = mp_standoff(5.0, 400.0, 5.0)
    old_reference = mp_standoff(5.0, 400.0, 5.0, B_msp_ref=50.0)

    assert 9.5 <= derived["r_mp_RE"] <= 11.5
    assert derived["r_mp_RE"] == pytest.approx(QUIET_STANDOFF_RE)
    assert derived["r_mp_RE"] / old_reference["r_mp_RE"] == pytest.approx(
        (DERIVED_B_MSP_REF_NT / 50.0) ** (1 / 3), rel=0.01
    )


def test_dynamic_pressure_is_standard_and_applied_pressure_carries_stagnation_factor(recipe):
    mp_standoff = _pressure_balance_namespace(recipe)["mp_standoff"]

    result = mp_standoff(5.0, 400.0, 5.0)

    assert result["P_dyn_nPa"] == pytest.approx(1.33808, abs=1e-4)
    assert result["P_applied_nPa"] == pytest.approx(0.88 * result["P_dyn_nPa"])


def test_derived_reference_field_is_independent_of_reference_radius(recipe):
    mp_standoff = _pressure_balance_namespace(recipe)["mp_standoff"]

    at_10_re = mp_standoff(5.0, 400.0, 5.0)
    at_20_re = mp_standoff(5.0, 400.0, 5.0, r_ref=20.0)

    assert at_20_re["B_msp_ref_nT"] == pytest.approx(DERIVED_B_MSP_REF_NT / 8)
    assert at_20_re["r_mp_RE"] == pytest.approx(at_10_re["r_mp_RE"])


def test_standoff_scales_with_chapman_ferraro_factor(recipe):
    mp_standoff = _pressure_balance_namespace(recipe)["mp_standoff"]

    nominal = mp_standoff(5.0, 400.0, 5.0)
    doubled = mp_standoff(5.0, 400.0, 5.0, f_cf=4.0)

    assert doubled["r_mp_RE"] / nominal["r_mp_RE"] == pytest.approx(2 ** (1 / 3))


def test_standoff_scales_with_surface_dipole_field(recipe):
    mp_standoff = _pressure_balance_namespace(recipe)["mp_standoff"]

    nominal = mp_standoff(5.0, 400.0, 5.0)
    doubled = mp_standoff(5.0, 400.0, 5.0, B0_nT=2 * IGRF13_B0_NT)

    assert doubled["r_mp_RE"] / nominal["r_mp_RE"] == pytest.approx(2 ** (1 / 3))


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
        "P_applied_nPa": "nPa",
        "P_mag_sw_nPa": "nPa",
        "P_total_sw_nPa": "nPa",
        "B_msp_ref_nT": "nT",
    }
    assert run.value("P_dyn_nPa")[0] == pytest.approx(1.33808, abs=1e-4)
    assert run.value("P_applied_nPa")[0] == pytest.approx(0.88 * 1.33808)


def test_run_block_does_not_export_without_bound_inputs(recipe):
    run = recipe("pressure_balance", **{"__name__": "recipe"})

    assert run.exports == {}


def test_pressure_balance_standoff_is_close_to_shue_for_north_south_neutral_imf(recipe):
    """With standard P_dyn and Bz=0, pressure balance is 4.5% below Shue's r0.

    The 15% band is a judgment-call sanity check, not a calibration.
    """
    mp_standoff = _pressure_balance_namespace(recipe)["mp_standoff"]

    result = mp_standoff(5.0, 400.0, 5.0)
    _, shue_r = mp_shue1998(result["P_dyn_nPa"], 0.0, theta_deg=0.0)

    assert float(shue_r[0]) == pytest.approx(10.895567869889632)
    assert result["r_mp_RE"] == pytest.approx(float(shue_r[0]), rel=0.15)
