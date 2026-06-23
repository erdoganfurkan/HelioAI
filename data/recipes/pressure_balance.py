# name: pressure_balance
# description: Determine the magnetopause standoff distance using solar wind dynamic pressure and IMF balance with Earth's dipole magnetic pressure.
# inputs: n_sw (solar wind density cm-3), V_sw (solar wind speed km/s), B_sw (IMF magnitude nT), B_msp (magnetosphere field nT at subsolar nose, default 50 nT)
# outputs: r_mp (magnetopause standoff distance in Earth radii), P_dyn (dynamic pressure nPa), P_mag_sw (solar wind magnetic pressure nPa), P_mag_msp (magnetospheric pressure nPa)
# reference: Chapman-Ferraro pressure balance (Chapman & Ferraro 1931); standoff scaling, see Shue et al. (1997), JGR 102, 9497.

"""Magnetopause pressure balance.

The magnetopause standoff distance r_mp (in Earth radii from geocenter) is found
from the pressure balance between the solar wind and the magnetospheric field:

    P_dyn + P_mag_sw ≈ P_mag_msp(r_mp)

where:
    P_dyn       = (1/2) ρ V²              solar wind dynamic pressure
    P_mag_sw    = B_sw² / (2 μ₀)          solar wind magnetic pressure
    P_mag_msp   = B_msp(r)² / (2 μ₀)     magnetospheric field pressure

For Earth's dipole, B_msp(r) ≈ B_equator (R_E/r)^6  (equatorial dipole, ×2 at subsolar)
Typical subsolar dipole ~30-60 nT at 10 R_E → scale from input B_msp.

Usage inside run_python:
    load_recipe("pressure_balance")
    r_mp = mp_standoff(n_sw=5, V_sw=400, B_sw=5)
"""

import numpy as np

MU0     = 4e-7 * np.pi   # H/m
MP      = 1.6726e-27      # proton mass kg
RE      = 6.371e6         # Earth radius m
CM3_M3  = 1e6             # 1/cm³ = 1e6 /m³
NT_T    = 1e-9            # nT → T
KMS_MS  = 1e3             # km/s → m/s
PA_NPA  = 1e9             # Pa → nPa


def mp_standoff(
    n_sw: float,
    V_sw: float,
    B_sw: float,
    B_msp_ref: float = 50.0,
    r_ref: float = 10.0,
) -> float:
    """Estimate magnetopause standoff distance from pressure balance.

    Parameters
    ----------
    n_sw      : solar wind proton number density (cm⁻³)
    V_sw      : solar wind speed (km/s)
    B_sw      : IMF magnitude (nT)
    B_msp_ref : magnetospheric field at r_ref (nT); default 50 nT at 10 RE
    r_ref     : reference distance for B_msp_ref (RE)

    Returns
    -------
    r_mp : standoff distance in Earth radii
    """
    # Solar wind pressures (SI)
    rho     = n_sw * CM3_M3 * MP
    P_dyn   = 0.5 * rho * (V_sw * KMS_MS)**2
    P_mag_sw = (B_sw * NT_T)**2 / (2 * MU0)
    P_sw_total = P_dyn + P_mag_sw

    # Dipole scaling: B(r) = B_ref * (r_ref / r)^3  (equatorial dipole component)
    # Subsolar field ~ 2× equatorial (field compression factor at nose ≈ 1.3-2)
    # Pressure balance: B(r)^2 / (2 mu0) = P_sw_total
    # → r_mp = r_ref * (B_ref^2 / (2 mu0 * P_sw_total))^(1/6)
    B_ref_si = B_msp_ref * NT_T
    P_msp_ref = B_ref_si**2 / (2 * MU0)
    r_mp = r_ref * (P_msp_ref / P_sw_total) ** (1.0 / 6.0)

    # nPa conversions for readability
    P_dyn_nPa    = P_dyn * PA_NPA
    P_mag_sw_nPa = P_mag_sw * PA_NPA
    P_sw_nPa     = P_sw_total * PA_NPA

    export("r_mp_RE", np.array([r_mp]))
    export("P_dyn_nPa", np.array([P_dyn_nPa]))
    export("P_mag_sw_nPa", np.array([P_mag_sw_nPa]))
    export("P_total_sw_nPa", np.array([P_sw_nPa]))

    print(f"Dynamic pressure      : {P_dyn_nPa:.3f} nPa")
    print(f"SW magnetic pressure  : {P_mag_sw_nPa:.3f} nPa")
    print(f"Total SW pressure     : {P_sw_nPa:.3f} nPa")
    print(f"Magnetopause standoff : {r_mp:.2f} R_E")

    return r_mp


# ── Example (typical slow solar wind) ─────────────────────────────────────────
if __name__ == "__main__":
    mp_standoff(n_sw=5, V_sw=400, B_sw=5)
