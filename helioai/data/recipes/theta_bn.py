# name: theta_bn
# description: Compute the shock normal angle theta_Bn from upstream and downstream magnetic field vectors.
# inputs: B_up (array of shape (N,3) or (3,) in nT, upstream), B_dn (array of shape (N,3) or (3,), downstream)
# outputs: theta_bn_deg — angle in degrees between the upstream B and the shock normal
# reference: Coplanarity theorem (Colburn & Sonett 1966); Schwartz (1998), "Shock and Discontinuity Normals, Mach Numbers and Related Parameters", ISSI SR-001, ch. 10.

"""Shock normal angle theta_Bn.

theta_Bn is the angle between the upstream magnetic field and the shock normal vector n.
The normal n is estimated from the coplanarity theorem:
    n = (B_dn × B_up) × (B_dn - B_up)
    n = n / |n|

theta_Bn < 45° → quasi-parallel shock (field-aligned)
theta_Bn > 45° → quasi-perpendicular shock

Usage (inside run_python after downloading B upstream and downstream):
    B_up = var.values[mask_up]     # (N,3) over the upstream interval, or its mean 3-vector
    B_dn = var.values[mask_dn]     # (N,3) over the downstream interval, or its mean 3-vector
    # Then run this script: it uses B_up / B_dn when they are already defined.

An (N, 3) input is averaged over its finite rows — fill values are NaN by the time the
data reaches you, and a plain mean of a gapped interval is NaN. A NaN, zero or collinear
input returns {"error": ...} and nothing else: no angle, no geometry.
"""

import numpy as np


def _to_vec(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        finite_rows = np.isfinite(arr).all(axis=1)
        if not finite_rows.any():
            return np.full(3, np.nan)
        return arr[finite_rows].mean(axis=0)
    return arr.ravel()[:3]


def theta_bn(B_up, B_dn):
    """Shock normal angle between the upstream field and the coplanarity normal.

    Parameters
    ----------
    B_up : (N, 3) or (3,) array of upstream field vectors in nT. An (N, 3)
           series is averaged over its finite rows, so an upstream window can be
           passed as-is without pre-averaging — fill values are NaN by then, and
           a plain mean of a gapped interval would be NaN.
    B_dn : (N, 3) or (3,) array of downstream field vectors in nT, same handling.

    Returns
    -------
    dict with theta_bn_deg (degrees, 0-90), geometry ("quasi-parallel" below
    45 deg, "quasi-perpendicular" above), shock_normal (unit 3-vector),
    B_up_mean_nT and B_dn_mean_nT (the vectors actually used).
    On a NaN or zero input, or on collinear or identical vectors, the normal is
    undefined and the dict carries an "error" key and nothing else: a NaN used
    to flow through to theta_bn_deg and, `nan < 45` being False, come back
    labelled quasi-perpendicular — a physical classification of an angle that
    does not exist.

    The angle is taken through |cos| because the coplanarity normal has an
    arbitrary sign: n and -n describe the same shock, so theta_Bn is folded into
    0-90 deg. Anything downstream that needs the normal's direction (a shock
    speed, a Mach number) must fix the sign itself from the flow.
    """
    u = _to_vec(B_up)
    d = _to_vec(B_dn)

    if u.shape != (3,) or d.shape != (3,) or not (np.isfinite(u).all() and np.isfinite(d).all()):
        return {"error": "invalid input: B_up and B_dn must be finite 3-vectors, or (N,3) arrays with at least one finite row"}
    if np.linalg.norm(u) < 1e-12 or np.linalg.norm(d) < 1e-12:
        return {"error": "invalid input: B_up or B_dn is a zero vector"}

    dB = d - u
    n = np.cross(np.cross(d, u), dB)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return {"error": "degenerate: B_up and B_dn are collinear or identical"}

    n_hat = n / norm
    u_hat = u / np.linalg.norm(u)

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
# Define B_up / B_dn before this script — from a downloaded dataset:
#   var = load_data("b3gsm")                       # (N, 3) GSM vector, fill already NaN
#   B_up = var.values[(var.time >= t_up0) & (var.time <= t_up1)]
#   B_dn = var.values[(var.time >= t_dn0) & (var.time <= t_dn1)]
# The placeholders below are used only when neither is defined.

B_up = globals().get("B_up", np.array([5.0, -2.0, 1.0]))    # placeholder upstream B (nT)
B_dn = globals().get("B_dn", np.array([15.0, -8.0, 4.0]))   # placeholder downstream B (nT)

result = theta_bn(B_up, B_dn)
if "error" in result:
    print("theta_bn:", result["error"])
else:
    export("theta_bn", np.array([result["theta_bn_deg"]]), "deg")
    print(result)
