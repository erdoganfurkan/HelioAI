# name: mvab
# description: Minimum Variance Analysis of B (MVAB) — finds the coordinate system where Bn variance is minimum, giving the current-sheet or discontinuity normal.
# inputs: B (magnetic field array shape (N,3) in nT)
# outputs: mvab_ratio_int_min, mvab_lambda_min, mvab_dphi_min_int, mvab_dphi_min_max, mvab_normal
# reference: Sonnerup & Scheible (1998), "Minimum and Maximum Variance Analysis", in Analysis Methods for Multi-Spacecraft Data, ISSI SR-001, ch. 8, eq. 8.23-8.24 for the uncertainties.

"""Minimum Variance Analysis of the magnetic field (MVAB).

MVAB finds the eigenvectors of the magnetic variance matrix M:
    M_ij = <Bi * Bj> - <Bi> * <Bj>

Eigenvalues λ_min < λ_int < λ_max:
- λ_min → minimum variance direction (shock/current-sheet normal n)
- λ_int → intermediate variance direction
- λ_max → maximum variance direction (along field)

Quality indicators and uncertainty:
- λ_int / λ_min > 5      → well-determined normal
- λ_max / λ_int > 5      → clean rotation
- Eigenvalue ratios < 2  → degenerate — normal poorly constrained
- The Sonnerup & Scheible (1998) statistical angular errors are reported for
  rotations of the minimum-variance normal toward the intermediate and maximum
  directions. N < 30 is allowed but warned: the eigenvalue ratios are then too
  sample-limited to trust by themselves.

Reference: Sonnerup & Scheible, ISSI SR-001, 1998, ch. 8.

Usage (inside run_python):
    B = var.values[:, :3]   # (N, 3) array, columns = Bx, By, Bz in nT
"""

import numpy as np


def _angle_error(lam_i, lam_j, lam_min, n_samples, floor):
    numerator = max(lam_i + lam_j - lam_min, 0.0)
    denominator = (lam_i - lam_j) ** 2
    if denominator <= floor**2:
        return float("inf")
    variance = max(lam_min, 0.0) / (n_samples - 1) * numerator / denominator
    return float(np.sqrt(max(variance, 0.0)))


def _angle_projection_term(angle_rad, mean_component, floor):
    if np.isfinite(angle_rad):
        return (angle_rad * mean_component) ** 2
    return float("inf") if abs(mean_component) > floor else 0.0


def mvab(B):
    """Estimate the magnetic minimum-variance normal and its statistical error.

    Parameters
    ----------
    B : array-like, shape (N, 3)
        Magnetic field vectors in nT.

    Returns
    -------
    dict
        Eigenvalues and eigenvectors of the sample covariance matrix, the raw
        eigenvalue ratios, a quality label, the Sonnerup-Scheible angular
        uncertainty of the normal in degrees, and Δ<B·n> in nT.
    """
    B = np.asarray(B, dtype=float)
    if B.ndim != 2 or B.shape[1] < 3:
        return {"error": "B must be shape (N, 3)"}

    B = B[:, :3]
    N = B.shape[0]
    if N < 2:
        return {"error": "B must contain at least two samples"}

    M = np.cov(B.T)   # 3×3 variance matrix

    eigenvalues, eigenvectors = np.linalg.eigh(M)   # ascending order

    lam_min, lam_int, lam_max = eigenvalues
    n_min  = eigenvectors[:, 0]   # minimum variance → normal
    n_int  = eigenvectors[:, 1]
    n_max  = eigenvectors[:, 2]

    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    eigenvalue_floor = 100.0 * np.finfo(float).eps * scale

    ratio_int_min = float(lam_int / lam_min) if lam_min > 0 else float("inf")
    ratio_max_int = float(lam_max / lam_int) if lam_int > 0 else float("inf")

    dphi_min_int = _angle_error(lam_int, lam_min, lam_min, N, eigenvalue_floor)
    dphi_min_max = _angle_error(lam_max, lam_min, lam_min, N, eigenvalue_floor)
    B_mean = B.mean(axis=0)
    B_int_mean = float(np.dot(B_mean, n_int))
    B_max_mean = float(np.dot(B_mean, n_max))
    dBn = np.sqrt(
        max(lam_min, 0.0) / (N - 1)
        + _angle_projection_term(dphi_min_int, B_int_mean, eigenvalue_floor)
        + _angle_projection_term(dphi_min_max, B_max_mean, eigenvalue_floor)
    )

    if lam_min <= eigenvalue_floor:
        quality = "degenerate — λ_min is numerically zero/non-positive, normal poorly constrained"
    elif ratio_int_min > 5:
        quality = "well-determined normal (λ_int/λ_min > 5)"
    elif ratio_int_min > 2:
        quality = "moderate — normal may be uncertain"
    else:
        quality = "degenerate — λ_int/λ_min < 2, normal poorly constrained"

    result = {
        "normal_n_min": n_min.tolist(),
        "n_int": n_int.tolist(),
        "n_max": n_max.tolist(),
        "lambda_min": float(lam_min),
        "lambda_int": float(lam_int),
        "lambda_max": float(lam_max),
        "ratio_int_min": ratio_int_min,
        "ratio_max_int": ratio_max_int,
        "dphi_min_int_deg": float(np.degrees(dphi_min_int)),
        "dphi_min_max_deg": float(np.degrees(dphi_min_max)),
        "dBn_nT": float(dBn),
        "quality": quality,
    }
    if N < 30:
        result["warning"] = "fewer than 30 samples — eigenvalue ratios unreliable"
    return result


# ── Run ────────────────────────────────────────────────────────────────────────
# Replace B with your actual speasy variable values.
# Example:
#   var = spz.get_data("cdaweb/MMS1_FGM_SRVY_L2/mms1_fgm_b_gse_srvy_l2",
#                      "2017-07-11T22:30", "2017-07-11T22:40")
#   B = var.values[:, :3]

if __name__ == "__main__" and "export" not in globals():

    def export(name, data, units=""):
        return None


if __name__ == "__main__" and "B" not in globals():
    np.random.seed(0)
    N = 300
    t = np.linspace(0, 2 * np.pi, N)
    # Synthetic current sheet crossing: Bx rotates, Bz is the normal component (small)
    B = np.column_stack([
        10 * np.tanh(t - np.pi),          # Bx: Harris sheet
        5  * np.sin(t),                   # By: guide field variation
        0.5 * np.random.normal(0, 1, N),  # Bz: small normal component
    ])


B = globals().get("B")
if B is not None:
    result = mvab(B)
    if "error" not in result:
        export("mvab_ratio_int_min", np.array([result["ratio_int_min"]]), "")
        export("mvab_lambda_min", np.array([result["lambda_min"]]), "nT2")
        export("mvab_dphi_min_int", np.array([result["dphi_min_int_deg"]]), "deg")
        export("mvab_dphi_min_max", np.array([result["dphi_min_max_deg"]]), "deg")
        export("mvab_normal", np.array(result["normal_n_min"]), "")
        if "warning" in result:
            print(result["warning"])
    print(result)
