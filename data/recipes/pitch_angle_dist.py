# name: pitch_angle_dist
# description: Compute the pitch angle of a particle population given particle velocity vectors and local magnetic field vectors, then plot the pitch angle distribution (PAD).
# inputs: V_particles (array (N,3) km/s), B_field (array (N,3) or (3,) nT), n_bins (int, default 18)
# outputs: pa_deg (pitch angles in degrees, array N), counts (histogram counts per bin), pad_plot (figure)
# reference: Pitch angle α = arccos(V·B / |V||B|); see Baumjohann & Treumann (1996), Basic Space Plasma Physics, ch. 2.

"""Pitch angle distribution (PAD).

The pitch angle α is the angle between the particle velocity vector and the
local magnetic field direction:

    cos(α) = (V · B) / (|V| |B|)

This recipe computes PADs for particle data obtained from MMS-FPI, Cluster-PEACE,
Van Allen Probes-MagEIS, or similar instruments.

Requirements:
- V_particles : shape (N, 3), particle velocity in instrument or GSE frame (km/s)
- B_field     : shape (N, 3) or (3,), magnetic field in the same frame (nT)

Usage inside run_python:
    load_recipe("pitch_angle_dist")
    pa, counts, edges = compute_pad(V_particles, B_field)
"""

import matplotlib.pyplot as plt
import numpy as np


def compute_pad(
    V_particles: "np.ndarray",
    B_field: "np.ndarray",
    n_bins: int = 18,
    label: str = "PAD",
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Compute and plot a pitch angle distribution.

    Parameters
    ----------
    V_particles : (N, 3) array of particle velocities (km/s)
    B_field     : (N, 3) or (3,) array of magnetic field vectors (nT)
    n_bins      : number of pitch angle bins (default 18 → 10° bins)
    label       : plot title suffix

    Returns
    -------
    pa_deg : (N,) pitch angles in degrees
    counts : (n_bins,) histogram counts
    edges  : (n_bins+1,) bin edges in degrees
    """
    V = np.asarray(V_particles, dtype=float)
    B = np.asarray(B_field, dtype=float)

    if V.ndim == 1:
        V = V[np.newaxis, :]
    if B.ndim == 1:
        B = np.broadcast_to(B, V.shape).copy()

    # Unit vectors
    V_norm = np.linalg.norm(V, axis=1, keepdims=True)
    B_norm = np.linalg.norm(B, axis=1, keepdims=True)

    # Mask zero-magnitude vectors
    mask = (V_norm[:, 0] > 0) & (B_norm[:, 0] > 0)
    V_hat = np.where(V_norm > 0, V / (V_norm + 1e-30), 0.0)
    B_hat = np.where(B_norm > 0, B / (B_norm + 1e-30), 0.0)

    cos_alpha = np.clip(np.sum(V_hat * B_hat, axis=1), -1.0, 1.0)
    pa_deg = np.degrees(np.arccos(cos_alpha))
    pa_deg[~mask] = np.nan

    valid = pa_deg[~np.isnan(pa_deg)]
    counts, edges = np.histogram(valid, bins=n_bins, range=(0, 180))
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    # Plot
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(bin_centers, counts, width=edges[1] - edges[0], color="#4fc3f7",
           edgecolor="#30363d", linewidth=0.5, alpha=0.85)
    ax.set_xlabel("Pitch angle (°)", fontsize=9)
    ax.set_ylabel("Counts", fontsize=9)
    ax.set_xlim(0, 180)
    ax.set_xticks(range(0, 181, 30))
    ax.set_title(label, fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    export("pitch_angles_median_deg", np.array([float(np.nanmedian(pa_deg))]))
    export("pitch_angles_mean_deg", np.array([float(np.nanmean(pa_deg))]))
    export("n_particles", np.array([float(len(valid))]))

    print(f"Median pitch angle : {np.nanmedian(pa_deg):.1f}°")
    print(f"Mean pitch angle   : {np.nanmean(pa_deg):.1f}°")
    print(f"Valid particles    : {len(valid)} / {len(pa_deg)}")

    return pa_deg, counts, edges


# ── Example (isotropic distribution) ─────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N = 500
    V_test = rng.normal(0, 1, (N, 3))
    B_test = np.array([0.0, 0.0, 10.0])
    compute_pad(V_test, B_test, label="Synthetic isotropic PAD")
