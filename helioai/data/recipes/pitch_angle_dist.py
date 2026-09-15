# name: pitch_angle_dist
# description: Compute the pitch angle of a particle population given particle velocity vectors and local magnetic field vectors, then plot the pitch angle distribution (PAD).
# inputs: V (array (N,3) km/s), B (array (N,3) or (3,) nT), n_bins (int, default 18), bins ("cos" or "deg", default "cos"), label (plot title)
# outputs: pitch_angles_median_deg, pitch_angles_mean_deg, n_particles, pad_counts, pad_bin_edges_deg, pad_anisotropy
# reference: Pitch angle α = arccos(V·B / |V||B|); see Baumjohann & Treumann (1996), Basic Space Plasma Physics, ch. 2.

"""Pitch angle distribution (PAD).

The pitch angle α is the angle between the particle velocity vector and the
local magnetic field direction:

    cos(α) = (V · B) / (|V| |B|)

This recipe computes PADs for particle data obtained from MMS-FPI, Cluster-PEACE,
Van Allen Probes-MagEIS, or similar instruments.

Equal-width bins in cos(α) have equal solid angle:

    dΩ = 2π sin(α) dα = −2π d(cos(α))

An isotropic population therefore has equal expected counts in every bin, without
a 1/sin(α) correction. The plot shows particle counts per steradian, not flux or
phase-space density. Its x-axis remains α in degrees: arccos-transformed edges
give wider bins near the poles and narrower bins near 90°, as required for equal
solid angle. The angular widths must not be mistaken for unequal acceptance.

For backward comparison, bins="deg" retains uniform degree bins and the old
division by sin(α) at each bin centre. That correction magnifies the absolute
Poisson fluctuations of sparsely populated polar bins; it cannot reduce their
large relative uncertainty. Polar spikes in that legacy plot need not be beams.

Requirements:
- V : shape (N, 3), particle velocity in instrument or GSE frame (km/s)
- B : shape (N, 3) or (3,), magnetic field in the same frame (nT)

Usage: run_recipe("pitch_angle_dist", inputs={"V": ..., "B": ...}) applies the
bound inputs. Alternatively, load the source and call:
    pa, counts, edges = compute_pad(V_particles, B_field)

The result still unpacks as that three-element tuple. It also exposes all six
exported quantities as named attributes, e.g. result.pad_anisotropy.
"""

import matplotlib.pyplot as plt
import numpy as np


class PADResult(tuple):
    """Preserve three-value unpacking while retaining every provenance quantity."""

    def __new__(cls, pa_deg, counts, edges, values):
        result = super().__new__(cls, (pa_deg, counts, edges))
        for name, value in values.items():
            setattr(result, name, value)
        return result


def compute_pad(
    V_particles: "np.ndarray",
    B_field: "np.ndarray",
    n_bins: int = 18,
    label: str = "PAD",
    *,
    bins: str = "cos",
) -> PADResult:
    """Compute and plot a pitch angle distribution.

    Parameters
    ----------
    V_particles : (N, 3) array of particle velocities (km/s)
    B_field     : (N, 3) or (3,) array of magnetic field vectors (nT)
    n_bins      : number of pitch angle bins (default 18)
    label       : plot title suffix
    bins        : "cos" (default), equal solid angle, or "deg", legacy 10° bins
                  for n_bins=18 with the pole-noise-amplifying 1/sin correction

    Returns
    -------
    pa_deg : (N,) pitch angles in degrees
    counts : (n_bins,) raw counts for "cos"; sine-corrected counts for "deg"
    edges  : (n_bins+1,) increasing bin edges in degrees

    Raises
    ------
    ValueError
        If n_bins is not a positive integer or bins is not "cos" or "deg".

    Notes
    -----
    The returned PADResult also carries pitch_angles_median_deg and
    pitch_angles_mean_deg (degrees), n_particles (valid particles), pad_counts
    (raw counts in either scheme), pad_bin_edges_deg (degrees), and pad_anisotropy.
    These quantities are exported with units when the sandbox export helper exists.

    Anisotropy is mean(counts at the two poles) / mean(counts at the equator).
    The equator uses the central two bins for even n_bins, or the central bin
    for odd n_bins. In "deg" mode it uses the legacy sine-corrected counts,
    not the unequal-solid-angle raw counts. Values >1 indicate field-aligned
    populations and <1 indicate pancakes; the chosen angular acceptance depends
    on n_bins. A populated pole with an empty equator gives infinity; both empty,
    or fewer than three bins, gives NaN. No pseudocount is added.
    """
    if bins not in ("cos", "deg"):
        raise ValueError('bins must be "cos" or "deg"')
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")

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
    if bins == "cos":
        cos_edges = np.linspace(-1.0, 1.0, n_bins + 1)
        # Negating cos(α) orders bins by increasing α, including the 90° boundary.
        counts_raw, _ = np.histogram(-cos_alpha[~np.isnan(pa_deg)], bins=cos_edges)
        edges = np.degrees(np.arccos(-cos_edges))
        counts = counts_raw
        plot_counts = counts / (4 * np.pi / n_bins)
    else:
        counts_raw, edges = np.histogram(valid, bins=n_bins, range=(0, 180))
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        sin_alpha = np.sin(np.radians(bin_centers))
        sin_alpha = np.where(sin_alpha < 1e-10, 1e-10, sin_alpha)
        counts = counts_raw / sin_alpha
        plot_counts = counts
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    polar = np.mean(counts[[0, -1]])
    equatorial = np.mean(counts[(n_bins - 1) // 2:n_bins // 2 + 1])
    if n_bins < 3 or (polar == 0 and equatorial == 0):
        anisotropy = float("nan")
    elif equatorial == 0:
        anisotropy = float("inf")
    else:
        anisotropy = float(polar / equatorial)
    median = float(np.median(valid)) if len(valid) else float("nan")
    mean = float(np.mean(valid)) if len(valid) else float("nan")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(bin_centers, plot_counts, width=np.diff(edges), color="#1976b9",
           edgecolor="#30363d", linewidth=0.5, alpha=0.85)
    ax.set_xlabel("Pitch angle (°)", fontsize=9)
    ax.set_ylabel("Particles sr⁻¹" if bins == "cos" else "Counts / sin(α) (legacy)", fontsize=9)
    ax.set_xlim(0, 180)
    ax.set_xticks(range(0, 181, 30))
    ax.set_title(label, fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    values = {
        "pitch_angles_median_deg": median,
        "pitch_angles_mean_deg": mean,
        "n_particles": len(valid),
        "pad_counts": counts_raw,
        "pad_bin_edges_deg": edges,
        "pad_anisotropy": anisotropy,
    }
    if "export" in globals():
        for name, value in values.items():
            export(name, np.atleast_1d(value), "deg" if name.endswith("_deg") else "")

    print(f"Median pitch angle : {median:.1f}°")
    print(f"Mean pitch angle   : {mean:.1f}°")
    print(f"Valid particles    : {len(valid)} / {len(pa_deg)}")

    return PADResult(pa_deg, counts, edges, values)


# ── Run on bound inputs only ─────────────────────────────────────────────────
V = globals().get("V")
B = globals().get("B")
if V is not None and B is not None:
    result = compute_pad(
        V, B, n_bins=globals().get("n_bins", 18),
        bins=globals().get("bins", "cos"),
        label=globals().get("label", "Pitch angle distribution"),
    )


# ── Example (isotropic distribution) ─────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N = 500
    V_test = rng.normal(0, 1, (N, 3))
    B_test = np.array([0.0, 0.0, 10.0])
    compute_pad(V_test, B_test, label="Synthetic isotropic PAD")
