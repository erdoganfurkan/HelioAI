# name: walen_test
# description: Walén test — check whether a discontinuity is a rotational discontinuity (|slope| ~ 1) or a tangential discontinuity (|slope| << 1).
# inputs: V (ion velocity array shape (N,3) in km/s), B (magnetic field array shape (N,3) in nT), n_cm3 (ion density array shape (N,) in cm⁻³)
# outputs: slope, R2, interpretation

"""Walén test for rotational vs tangential discontinuities.

The test computes the Alfvén velocity from the measured B and n, then compares
ΔV (observed ion velocity change across the discontinuity) with ΔV_A (Alfvén
velocity change predicted for a rotational discontinuity).

If the regression slope ΔV ~ slope * ΔV_A satisfies |slope| ≈ 0.9–1.1, the
structure is consistent with a rotational discontinuity (e.g. reconnection jet).
|slope| << 1 → tangential discontinuity.

The regression is done separately on Vx, Vy, Vz vs V_Ax, V_Ay, V_Az.

Usage (inside run_python):
    # V, B arrays must be aligned in time (same timestamps).
    # Split the interval into two halves: before and after the discontinuity,
    # or use the full interval for a scatter-plot slope estimate.
"""

import numpy as np
from scipy import stats

MU0 = 4 * np.pi * 1e-7       # H/m
MP = 1.6726e-27               # kg (proton mass)
CM3_TO_M3 = 1e6               # 1 cm⁻³ = 1e6 m⁻³


def alfven_velocity(B_nT, n_cm3):
    """Alfvén velocity in km/s. B shape (N,3), n shape (N,)."""
    B_T = B_nT * 1e-9                          # nT → T
    n_m3 = n_cm3 * CM3_TO_M3
    rho = n_m3 * MP                            # kg/m³
    VA = B_T / np.sqrt(MU0 * rho[:, None])    # m/s vector
    return VA * 1e-3                           # km/s


def walen_test(V, B, n_cm3):
    """Run the Walén test.

    Args:
        V:     Ion bulk velocity (N, 3) in km/s
        B:     Magnetic field (N, 3) in nT
        n_cm3: Ion number density (N,) in cm⁻³

    Returns dict with slope, R², and physical interpretation.
    """
    V = np.asarray(V, dtype=float)
    B = np.asarray(B, dtype=float)
    n = np.asarray(n_cm3, dtype=float)

    VA = alfven_velocity(B, n)  # (N, 3) in km/s

    # Detrend both V and VA (reference frame deHoffmann-Teller approximation)
    dV = V - V.mean(axis=0)
    dVA = VA - VA.mean(axis=0)

    # Flatten all components for a single global regression
    dV_flat = dV.ravel()
    dVA_flat = dVA.ravel()

    slope, intercept, r_value, p_value, _ = stats.linregress(dVA_flat, dV_flat)
    R2 = r_value ** 2

    if abs(slope) >= 0.9:
        interpretation = "rotational discontinuity / reconnection (|slope| ≈ 1)"
    elif abs(slope) >= 0.5:
        interpretation = "mixed — possible weakly rotational structure"
    else:
        interpretation = "tangential discontinuity (|slope| ≪ 1)"

    return {
        "slope": round(float(slope), 4),
        "R2": round(float(R2), 4),
        "intercept_km_s": round(float(intercept), 4),
        "interpretation": interpretation,
        "VA_mean_km_s": VA.mean(axis=0).tolist(),
    }


# ── Run ────────────────────────────────────────────────────────────────────────
# Replace V, B, n_cm3 with your actual speasy variables.
# Example:
#   var_B = spz.get_data("cdaweb/MMS1_FGM_SRVY_L2/mms1_fgm_b_gse_srvy_l2", t0, t1)
#   var_V = spz.get_data("cdaweb/MMS1_FPI_FAST_L2_DIS-MOMS/mms1_dis_bulkv_gse_fast", t0, t1)
#   var_n = spz.get_data("cdaweb/MMS1_FPI_FAST_L2_DIS-MOMS/mms1_dis_numberdensity_fast", t0, t1)
#   # Interpolate to common timestamps before passing here.

np.random.seed(42)
N = 200
t = np.linspace(0, np.pi, N)
B = np.column_stack([5 + 3 * np.cos(t), -2 + np.sin(t), 1 + 0.5 * np.sin(2 * t)])
n_cm3 = 5 + np.random.normal(0, 0.3, N)
VA = alfven_velocity(B, n_cm3)
V = VA + np.random.normal(0, 5, (N, 3))   # noisy rotational discontinuity

result = walen_test(V, B, n_cm3)
export("walen_slope", np.array([result["slope"]]))
export("walen_R2", np.array([result["R2"]]))
print(result)
