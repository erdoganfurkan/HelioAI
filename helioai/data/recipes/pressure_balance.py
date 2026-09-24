# name: pressure_balance
# description: Determine the magnetopause standoff distance from Chapman-Ferraro pressure balance.
# inputs: n_sw (solar wind density cm-3), V_sw (solar wind speed km/s), B_sw (IMF magnitude nT), B0_nT (optional surface equatorial dipole field nT), f_cf (optional Chapman-Ferraro field factor), r_ref (optional reference distance R_E), B_msp_ref (optional magnetosphere field nT at r_ref)
# outputs: r_mp_RE (R_earth), P_dyn_nPa (nPa), P_applied_nPa (nPa), P_mag_sw_nPa (nPa), P_total_sw_nPa (nPa), B_msp_ref_nT (nT)
# reference: Chapman & Ferraro (1931); Spreiter, Summers & Alksne (1966), Planet. Space Sci. 14, 223; Alken et al. (2021), Earth Planets Space 73, 49; Schield (1969), JGR 74, 1275; Shue et al. (1998), JGR 103, 17691.

"""Magnetopause pressure balance.

The subsolar magnetopause standoff distance r_mp is estimated from the
Chapman-Ferraro pressure balance between the solar wind and the compressed
dayside geomagnetic field:

    P_applied + P_mag_sw ≈ P_mag_msp(r_mp)

where:
    P_dyn       = ρ V²                   standard solar wind dynamic pressure
    P_applied   = K P_dyn                stagnation pressure used in the balance
    P_mag_sw    = B_sw² / (2 μ₀)          solar wind magnetic pressure
    P_mag_msp   = B_msp(r)² / (2 μ₀)      magnetospheric magnetic pressure

For a dipole, the field scales as B_dip(r) = B0 (R_E / r)^3, so the magnetic
pressure scales as r^-6. At the subsolar magnetopause the Chapman-Ferraro
current approximately doubles the field. With B0 = 29 806 nT from the IGRF-13
epoch-2020 dipole coefficients and f_CF = 2.0, the derived reference field is
59.6 nT at 10 R_E. The older 30 800 nT value is a common textbook round number
(Kivelson & Russell, 1995, Table 6.1); pass it as B0_nT if that convention is
wanted. Schield (1969) discusses a larger effective factor, about 2.44, when
tail and ring-current contributions are included; keep f_cf explicit if that
assumption is wanted.

Usage through run_recipe:
    run_recipe("pressure_balance", inputs={"n_sw": 5, "V_sw": 400, "B_sw": 5})

The run block reads n_sw, V_sw and B_sw from globals and exports results only
when all three are bound. Optional globals B0_nT, f_cf, r_ref and B_msp_ref
override the default reference field. Direct use is also supported:
    result = mp_standoff(n_sw=5, V_sw=400, B_sw=5)

The pressure balance ignores IMF Bz. For an empirical standoff that includes
southward-Bz erosion, compare with mp_shue1998(P_dyn_nPa, Bz_nT) in the
sandbox.
"""

import numpy as np

MU0 = 4e-7 * np.pi     # H/m
MP = 1.6726e-27        # proton mass kg
CM3_M3 = 1e6           # 1/cm³ = 1e6 /m³
NT_T = 1e-9            # nT → T
KMS_MS = 1e3           # km/s → m/s
PA_NPA = 1e9           # Pa → nPa

# Spreiter, Summers & Alksne (1966): stagnation-pressure coefficient.
K_STAGNATION = 0.88
# Alken et al. (2021), IGRF-13 epoch 2020 dipole coefficients: |g10,g11,h11|.
B0_NT = 29_806.0
# Chapman & Ferraro (1931) dayside current doubles the dipole field at the nose.
F_CF = 2.0
BZ_NOTE = (
    "pressure balance ignores IMF Bz; for an empirical standoff that includes "
    "Bz erosion compare with mp_shue1998(P_dyn_nPa, Bz_nT) available in the sandbox"
)


def mp_standoff(
    n_sw: float,
    V_sw: float,
    B_sw: float,
    *,
    B0_nT: float = B0_NT,
    f_cf: float = F_CF,
    r_ref: float = 10.0,
    B_msp_ref: float | None = None,
) -> dict:
    """Estimate subsolar magnetopause standoff from pressure balance.

    Parameters
    ----------
    n_sw : float
        Solar wind proton number density in cm⁻³.
    V_sw : float
        Solar wind bulk speed in km/s.
    B_sw : float
        IMF magnitude in nT. This pressure-balance recipe uses the magnetic
        pressure from |B| but deliberately does not use IMF Bz orientation.
    B0_nT : float, optional
        Equatorial dipole field at Earth's surface in nT. The default,
        29 806 nT, is computed from the IGRF-13 epoch-2020 dipole coefficients
        reported by Alken et al. (2021).
    f_cf : float, optional
        Chapman-Ferraro enhancement of the dayside field. The default 2.0 is
        the classical current-sheet doubling; Schield (1969) discusses about
        2.44 when tail and ring-current contributions are folded in.
    r_ref : float, optional
        Reference distance in Earth radii for B_msp_ref or for the derived
        dipole reference field.
    B_msp_ref : float, optional
        Magnetospheric field in nT at r_ref. When supplied, this explicit value
        is used to reproduce earlier calculations or a scientist's preferred
        frame; when omitted it is derived as f_cf * B0_nT * (R_E / r_ref)^3.

    Returns
    -------
    dict
        r_mp_RE, standard and applied solar-wind pressure terms in nPa, the
        B_msp_ref_nT actually used, bz_ignored=True, and a note pointing to
        mp_shue1998 for Bz-aware empirical comparisons.

        Changed: returns a dict (was a float); result["r_mp_RE"] is the former
        return value.
    """
    rho = n_sw * CM3_M3 * MP
    P_dyn = rho * (V_sw * KMS_MS) ** 2
    P_applied = K_STAGNATION * P_dyn
    P_mag_sw = (B_sw * NT_T) ** 2 / (2 * MU0)
    P_sw_total = P_applied + P_mag_sw

    if B_msp_ref is None:
        B_msp_ref_nT = f_cf * B0_nT * (1.0 / r_ref) ** 3
    else:
        B_msp_ref_nT = B_msp_ref

    P_msp_ref = (B_msp_ref_nT * NT_T) ** 2 / (2 * MU0)
    r_mp = r_ref * (P_msp_ref / P_sw_total) ** (1.0 / 6.0)

    result = {
        "r_mp_RE": float(r_mp),
        "P_dyn_nPa": float(P_dyn * PA_NPA),
        "P_applied_nPa": float(P_applied * PA_NPA),
        "P_mag_sw_nPa": float(P_mag_sw * PA_NPA),
        "P_total_sw_nPa": float(P_sw_total * PA_NPA),
        "B_msp_ref_nT": float(B_msp_ref_nT),
        "bz_ignored": True,
        "note": BZ_NOTE,
    }

    if "export" in globals():
        export("r_mp_RE", np.array([result["r_mp_RE"]]), "R_earth")
        export("P_dyn_nPa", np.array([result["P_dyn_nPa"]]), "nPa")
        export("P_applied_nPa", np.array([result["P_applied_nPa"]]), "nPa")
        export("P_mag_sw_nPa", np.array([result["P_mag_sw_nPa"]]), "nPa")
        export("P_total_sw_nPa", np.array([result["P_total_sw_nPa"]]), "nPa")
        export("B_msp_ref_nT", np.array([result["B_msp_ref_nT"]]), "nT")

    print(f"Dynamic pressure      : {result['P_dyn_nPa']:.3f} nPa")
    print(f"Applied pressure      : {result['P_applied_nPa']:.3f} nPa")
    print(f"SW magnetic pressure  : {result['P_mag_sw_nPa']:.3f} nPa")
    print(f"Total applied pressure: {result['P_total_sw_nPa']:.3f} nPa")
    print(f"Reference MSP field   : {result['B_msp_ref_nT']:.1f} nT at {r_ref:g} R_E")
    print(f"Magnetopause standoff : {result['r_mp_RE']:.2f} R_E")
    print(BZ_NOTE)

    return result


# ── Run ────────────────────────────────────────────────────────────────────────
# Define n_sw / V_sw / B_sw before this script. Optional globals B0_nT, f_cf,
# r_ref and B_msp_ref override the reference-field derivation.

n_sw = globals().get("n_sw")
V_sw = globals().get("V_sw")
B_sw = globals().get("B_sw")

if n_sw is not None and V_sw is not None and B_sw is not None:
    result = mp_standoff(
        n_sw,
        V_sw,
        B_sw,
        B0_nT=globals().get("B0_nT", B0_NT),
        f_cf=globals().get("f_cf", F_CF),
        r_ref=globals().get("r_ref", 10.0),
        B_msp_ref=globals().get("B_msp_ref"),
    )


# ── Example (typical slow solar wind) ─────────────────────────────────────────
if __name__ == "__main__":
    mp_standoff(n_sw=5, V_sw=400, B_sw=5)
