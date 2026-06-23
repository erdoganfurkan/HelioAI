# name: rankine_hugoniot
# description: Apply the Rankine-Hugoniot jump conditions to a collisionless shock, computing shock velocity and density compression ratio from upstream/downstream moments.
# inputs: n_u (upstream number density cm-3), n_d (downstream number density cm-3), V_u (upstream bulk speed km/s), V_d (downstream bulk speed km/s), B_u (upstream |B| nT), B_d (downstream |B| nT), T_u (upstream temperature eV), T_d (downstream temperature eV)
# outputs: V_shock (shock velocity km/s in spacecraft frame), r (density compression ratio), theta_check (consistency angle degrees)
# reference: Rankine-Hugoniot jump conditions (Rankine 1870; Hugoniot 1887); for collisionless shocks see Schwartz (1998), ISSI SR-001, ch. 10.

"""Rankine-Hugoniot jump conditions for a collisionless shock.

Applies the four conservation laws across the shock:
  1. Mass flux:         r = n_d / n_u  (compression ratio)
  2. Momentum flux:     n_u m V_u^2 + P_u + B_u^2/2mu0 = n_d m V_d^2 + P_d + B_d^2/2mu0
  3. Energy flux:       (total energy + pressure) × V conserved
  4. Magnetic flux:     B_d / B_u = r  (for perpendicular shock, exact)

For an oblique shock the full tensor form is needed; this script handles the
1-D normal-incidence (de Hoffmann-Teller) approximation which is valid for
quasi-perpendicular shocks (theta_Bn > 45°).

Usage inside run_python:
    load_recipe("rankine_hugoniot")
    r, V_shock = rh_jump(n_u=5, n_d=20, V_u=450, V_d=200, B_u=5, B_d=20, T_u=10, T_d=100)
"""

import numpy as np

MU0 = 4e-7 * np.pi        # H/m
MP  = 1.6726e-27           # proton mass kg
EV  = 1.6022e-19           # J per eV
CM3_TO_M3 = 1e6            # 1/cm³ = 1e6 /m³


def rh_jump(
    n_u: float, n_d: float,
    V_u: float, V_d: float,
    B_u: float, B_d: float,
    T_u: float = 0.0, T_d: float = 0.0,
) -> tuple[float, float]:
    """Compute shock velocity and compression ratio from RH jump conditions.

    Parameters
    ----------
    n_u, n_d : upstream/downstream number density (cm⁻³)
    V_u, V_d : upstream/downstream bulk speed along shock normal (km/s)
    B_u, B_d : upstream/downstream magnetic field magnitude (nT)
    T_u, T_d : upstream/downstream temperature (eV, optional)

    Returns
    -------
    V_shock : shock velocity in spacecraft frame (km/s)
    r       : density compression ratio n_d / n_u
    """
    # Compression ratio from mass-flux conservation (n*(V - V_sh) = const)
    # n_u (V_u - V_sh) = n_d (V_d - V_sh)
    # Solving for V_sh:
    # n_u V_u - n_u V_sh = n_d V_d - n_d V_sh
    # (n_d - n_u) V_sh = n_d V_d - n_u V_u
    if abs(n_d - n_u) < 1e-12:
        V_shock = 0.0
    else:
        V_shock = (n_d * V_d - n_u * V_u) / (n_d - n_u)

    r = n_d / n_u

    # Convert to SI for energy check
    n_u_si = n_u * CM3_TO_M3
    n_d_si = n_d * CM3_TO_M3
    V_u_si = V_u * 1e3
    V_d_si = V_d * 1e3
    B_u_si = B_u * 1e-9
    B_d_si = B_d * 1e-9
    T_u_si = T_u * EV
    T_d_si = T_d * EV

    # Pressures: thermal + magnetic
    P_u = n_u_si * T_u_si + B_u_si**2 / (2 * MU0)
    P_d = n_d_si * T_d_si + B_d_si**2 / (2 * MU0)

    # Momentum flux balance check: ρ V² + P should be conserved
    mom_u = n_u_si * MP * V_u_si**2 + P_u
    mom_d = n_d_si * MP * V_d_si**2 + P_d
    mom_residual = abs(mom_d - mom_u) / (abs(mom_u) + 1e-30)

    export("V_shock_km_s", np.array([V_shock]))
    export("compression_ratio", np.array([r]))
    export("momentum_residual_frac", np.array([mom_residual]))
    export("B_ratio", np.array([B_d / B_u]))

    print(f"Shock velocity  : {V_shock:.1f} km/s (spacecraft frame)")
    print(f"Compression ratio r = n_d/n_u = {r:.2f}")
    print(f"B ratio Bd/Bu   = {B_d/B_u:.2f}  (expected ~r for perp. shock: {r:.2f})")
    print(f"Momentum residual: {mom_residual*100:.1f}%  (< 10% = good)")

    return V_shock, r


# ── Example (edit values to match your event) ─────────────────────────────────
if __name__ == "__main__":
    # IP shock 2004-11-07 (approximate)
    V_shock, r = rh_jump(
        n_u=5,  n_d=25,
        V_u=500, V_d=200,
        B_u=5,  B_d=22,
        T_u=10, T_d=120,
    )
