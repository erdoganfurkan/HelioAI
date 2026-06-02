# name: mvab
# description: Minimum Variance Analysis of B (MVAB) — finds the coordinate system where Bn variance is minimum, giving the current-sheet or discontinuity normal.
# inputs: B (magnetic field array shape (N,3) in nT)
# outputs: eigenvalues (lambda_min, lambda_int, lambda_max), eigenvectors (n_min, n_int, n_max), normal direction

"""Minimum Variance Analysis of the magnetic field (MVAB).

MVAB finds the eigenvectors of the magnetic variance matrix M:
    M_ij = <Bi * Bj> - <Bi> * <Bj>

Eigenvalues λ_min < λ_int < λ_max:
- λ_min → minimum variance direction (shock/current-sheet normal n)
- λ_int → intermediate variance direction
- λ_max → maximum variance direction (along field)

Quality indicators:
- λ_int / λ_min > 5      → well-determined normal
- λ_max / λ_int > 5      → clean rotation
- Eigenvalue ratios < 2  → degenerate — normal poorly constrained

Reference: Sonnerup & Scheible, ISSI SR-001, 1998.

Usage (inside run_python):
    B = var.values[:, :3]   # (N, 3) array, columns = Bx, By, Bz in nT
"""

import numpy as np


def mvab(B):
    """Minimum Variance Analysis.

    Args:
        B: Magnetic field array (N, 3) in nT.

    Returns dict with eigenvalues, eigenvectors, ratios, and interpretation.
    """
    B = np.asarray(B, dtype=float)
    if B.ndim != 2 or B.shape[1] < 3:
        return {"error": "B must be shape (N, 3)"}

    B = B[:, :3]
    M = np.cov(B.T)   # 3×3 variance matrix

    eigenvalues, eigenvectors = np.linalg.eigh(M)   # ascending order

    lam_min, lam_int, lam_max = eigenvalues
    n_min  = eigenvectors[:, 0]   # minimum variance → normal
    n_int  = eigenvectors[:, 1]
    n_max  = eigenvectors[:, 2]

    ratio_int_min = float(lam_int / lam_min) if lam_min > 0 else float("inf")
    ratio_max_int = float(lam_max / lam_int) if lam_int > 0 else float("inf")

    if ratio_int_min > 5:
        quality = "well-determined normal (λ_int/λ_min > 5)"
    elif ratio_int_min > 2:
        quality = "moderate — normal may be uncertain"
    else:
        quality = "degenerate — λ_int/λ_min < 2, normal poorly constrained"

    return {
        "normal_n_min": n_min.tolist(),
        "n_int": n_int.tolist(),
        "n_max": n_max.tolist(),
        "lambda_min": round(float(lam_min), 4),
        "lambda_int": round(float(lam_int), 4),
        "lambda_max": round(float(lam_max), 4),
        "ratio_int_min": round(ratio_int_min, 2),
        "ratio_max_int": round(ratio_max_int, 2),
        "quality": quality,
    }


# ── Run ────────────────────────────────────────────────────────────────────────
# Replace B with your actual speasy variable values.
# Example:
#   var = spz.get_data("cdaweb/MMS1_FGM_SRVY_L2/mms1_fgm_b_gse_srvy_l2",
#                      "2017-07-11T22:30", "2017-07-11T22:40")
#   B = var.values[:, :3]

np.random.seed(0)
N = 300
t = np.linspace(0, 2 * np.pi, N)
# Synthetic current sheet crossing: Bx rotates, Bz is the normal component (small)
B = np.column_stack([
    10 * np.tanh(t - np.pi),           # Bx: Harris sheet
    5  * np.sin(t),                     # By: guide field variation
    0.5 * np.random.normal(0, 1, N),   # Bz: small normal component
])

result = mvab(B)
export("mvab_ratio_int_min", np.array([result["ratio_int_min"]]))
export("mvab_lambda_min",    np.array([result["lambda_min"]]))
print(result)
