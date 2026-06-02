# name: theta_bn
# description: Compute the shock normal angle theta_Bn from upstream and downstream magnetic field vectors.
# inputs: B_up (array of shape (N,3) or (3,) in nT, upstream), B_dn (array of shape (N,3) or (3,), downstream)
# outputs: theta_bn_deg — angle in degrees between the upstream B and the shock normal

"""Shock normal angle theta_Bn.

theta_Bn is the angle between the upstream magnetic field and the shock normal vector n.
The normal n is estimated from the coplanarity theorem:
    n = (B_dn × B_up) × (B_dn - B_up)
    n = n / |n|

theta_Bn < 45° → quasi-parallel shock (field-aligned)
theta_Bn > 45° → quasi-perpendicular shock

Usage (inside run_python after downloading B upstream and downstream):
    B_up = np.array([Bx_up_mean, By_up_mean, Bz_up_mean])   # average over upstream interval
    B_dn = np.array([Bx_dn_mean, By_dn_mean, Bz_dn_mean])   # average over downstream interval
    # Then run this script.
"""

import numpy as np


def _to_vec(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        return arr.mean(axis=0)
    return arr.ravel()[:3]


def theta_bn(B_up, B_dn):
    u = _to_vec(B_up)
    d = _to_vec(B_dn)

    dB = d - u
    n = np.cross(np.cross(d, u), dB)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return {"error": "degenerate: B_up and B_dn are collinear or identical"}

    n_hat = n / norm
    u_hat = u / (np.linalg.norm(u) + 1e-30)

    cos_angle = np.clip(np.dot(u_hat, n_hat), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(np.abs(cos_angle)))

    geometry = "quasi-parallel" if angle_deg < 45 else "quasi-perpendicular"
    return {
        "theta_bn_deg": round(float(angle_deg), 2),
        "geometry": geometry,
        "shock_normal": n_hat.tolist(),
        "B_up_mean_nT": u.tolist(),
        "B_dn_mean_nT": d.tolist(),
    }


# ── Run ────────────────────────────────────────────────────────────────────────
# Replace B_up / B_dn with your actual upstream and downstream B arrays.
# Example using previously downloaded speasy variables:
#   var_up = spz.get_data("cdaweb/MMS1_FGM_SRVY_L2/mms1_fgm_b_gse_srvy_l2",
#                          "2017-07-11T22:30", "2017-07-11T22:32")
#   var_dn = spz.get_data("cdaweb/MMS1_FGM_SRVY_L2/mms1_fgm_b_gse_srvy_l2",
#                          "2017-07-11T22:34", "2017-07-11T22:36")
#   B_up = var_up.values[:, :3]   # exclude |B| column if present
#   B_dn = var_dn.values[:, :3]

B_up = np.array([5.0, -2.0, 1.0])   # placeholder upstream B (nT)
B_dn = np.array([15.0, -8.0, 4.0])  # placeholder downstream B (nT)

result = theta_bn(B_up, B_dn)
export("theta_bn", np.array([result.get("theta_bn_deg", float("nan"))]))
print(result)
