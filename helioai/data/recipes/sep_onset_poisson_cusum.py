# name: sep_onset_poisson_cusum
# description: SEP onset with the Poisson CUSUM of Huttunen-Heikinmaa et al. (2005), on intensities; the z-score option is the same detector in normalized units.
# inputs: flux — SimpleNamespace(time, values) from load_data("<param>"), e.g. an energetic proton/electron intensity channel; method ("poisson" default, or "zscore"); component (int, default 0); bg_hours (background window length in hours from the start, default 2.0); robust (median/MAD background, default True); n_sigma (out-of-control level, default 2.0); h_sigma (nominal decision threshold, default 2.0); h (explicit decision threshold in the selected CUSUM units, default None); m_consecutive (finite samples that must stay above threshold, default 30); gap_reset (consecutive NaNs that reset the finite-sample run counter, default 5); units (flux units for exported raw CUSUM; otherwise flux.units or "")
# outputs: cusum, onset_index
# reference: Huttunen-Heikinmaa, Valtonen & Laitinen (2005), A&A 442, 673, doi:10.1051/0004-6361:20042620; Page (1954), Biometrika 41, 100, doi:10.1093/biomet/41.1-2.100; Rousseeuw & Croux (1993), J. Am. Stat. Assoc. 88, 1273, doi:10.1080/01621459.1993.10476408

"""SEP onset detection by Poisson CUSUM, with normalized output as an option.

Huttunen-Heikinmaa, Valtonen & Laitinen (2005) use a Poisson CUSUM on the
measured intensities, not on standardized residuals. The background gives the
in-control level ``mu`` and the out-of-control level is
``mu_d = mu + n_sigma*sigma``. The Poisson reference value is
``k = (mu_d - mu)/(ln(mu_d) - ln(mu))``, and the raw cumulative sum
``S_i = max(0, S_{i-1} + x_i - k)`` is compared with a threshold in the same
raw units. When ``h`` is not supplied, this recipe uses ``h_sigma*sigma`` so
the Poisson and optional z-score Page CUSUM have the same nominal sensitivity.
For a positive Poisson background, the z-score form with
``k_z = (k - mu)/sigma`` is exactly the same detector at another scale:
``S_raw = sigma*S_z``. The substantive choices in this recipe are therefore
the robust background estimate and starting detection only after the
background window.

The default background estimate is robust: ``mu`` is the median and ``sigma``
is ``1.4826*MAD`` (Hampel-style scaling, as discussed by Rousseeuw & Croux,
1993). Set ``robust=False`` to recover the mean/std background used by many
simple demonstrations. Detection begins after the background window so spikes
used to estimate the background cannot themselves seed the CUSUM run.
Missing samples do not count toward the required run of finite above-threshold
samples; more than ``gap_reset`` consecutive missing samples reset that run
counter and the CUSUM state. After a gap longer than ``gap_reset`` samples the
detector restarts; an onset that straddles such a gap is reported as
indeterminate through ``onset_indeterminate``.

Usage inside run_python:
    flux = load_data("erne_protons")
    method = "poisson"
    # then run this script
"""

import numpy as np
import matplotlib.pyplot as plt


def _poisson_reference_value(mu, mu_d):
    if not np.isfinite(mu) or not np.isfinite(mu_d) or mu_d <= mu:
        raise ValueError("Poisson CUSUM requires finite mu_d > mu")
    if mu <= 0.0:
        raise ValueError(
            "Poisson reference needs a positive background mean; subtract nothing, or use method='zscore'"
        )
    return float((mu_d - mu) / (np.log(mu_d) - np.log(mu)))


def background_stats(x, t, bg_hours, robust):
    """Estimate the pre-event background level and scatter.

    Parameters
    ----------
    x : array-like
        Intensity samples.
    t : array-like
        Sample times, convertible to ``datetime64[s]``.
    bg_hours : float
        Length of the background interval measured from the first sample.
    robust : bool
        Use median and scaled MAD when true; otherwise use mean and standard
        deviation.

    Returns
    -------
    tuple
        ``(mu, sigma, bg_mask)`` where ``bg_mask`` marks the finite samples used
        for the estimate.
    """

    x = np.asarray(x, dtype=float)
    t = np.asarray(t)
    if x.size != t.size:
        raise ValueError(f"flux.time and flux.values lengths differ: {t.size} != {x.size}")
    if x.size == 0:
        raise ValueError("empty flux time series")

    finite = np.isfinite(x)
    t_sec = t.astype("datetime64[s]").astype("int64").astype(float)
    bg_mask = finite & (t_sec <= t_sec[0] + float(bg_hours) * 3600.0)
    if bg_mask.sum() < 5:
        raise ValueError(f"background window too short: {bg_mask.sum()} finite samples")

    bg = x[bg_mask]
    bg_std = float(np.std(bg))
    if robust:
        mu = float(np.median(bg))
        sigma = float(1.4826 * np.median(np.abs(bg - mu)))
        if sigma <= 0.0 and bg_std > 0.0:
            sigma = bg_std
            print("background MAD is zero; falling back to standard deviation for sigma")
    else:
        mu = float(np.mean(bg))
        sigma = bg_std
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("flat background (sigma = 0) — pick another background window")
    return mu, sigma, bg_mask


def cusum_poisson(x, mu, mu_d, gap_reset=5):
    """Compute the raw Poisson CUSUM used for SEP onset timing.

    The returned series is in the same units as ``x`` because the
    Huttunen-Heikinmaa et al. recurrence uses ``x_i - k`` directly. Non-finite
    samples keep the previous state. A gap longer than ``gap_reset`` samples
    restarts the detector from zero at the next finite sample.
    """

    x = np.asarray(x, dtype=float)
    k = _poisson_reference_value(float(mu), float(mu_d))
    cusum = np.zeros_like(x, dtype=float)
    gap = 0
    for i in range(x.size):
        previous = cusum[i - 1] if i else 0.0
        if np.isfinite(x[i]):
            if gap > int(gap_reset):
                previous = 0.0
            gap = 0
            cusum[i] = max(0.0, previous + x[i] - k)
        else:
            gap += 1
            cusum[i] = previous
    return cusum


def cusum_zscore(x, mu, sigma, k, gap_reset=5):
    """Compute the optional Page CUSUM on standardized intensity residuals.

    This preserves the previous recipe behaviour: values are converted to
    ``(x - mu)/sigma`` and the Poisson reference value, shifted into z-units by
    the caller, is subtracted from each finite sample. Non-finite samples keep
    the previous state. A long gap restarts the detector from zero at the next
    finite sample.
    """

    x = np.asarray(x, dtype=float)
    z = (x - float(mu)) / float(sigma)
    cusum = np.zeros_like(x, dtype=float)
    gap = 0
    for i in range(x.size):
        previous = cusum[i - 1] if i else 0.0
        if np.isfinite(z[i]):
            if gap > int(gap_reset):
                previous = 0.0
            gap = 0
            cusum[i] = max(0.0, previous + z[i] - float(k))
        else:
            gap += 1
            cusum[i] = previous
    return cusum


def evidence_straddles_long_gap(cusum, h, finite, gap_reset=5):
    """Return true when above-threshold evidence is cut off by a long data gap.

    A CUSUM excursion followed by a long hole is not a confirmed onset and must
    not be joined to quiet samples on the far side. Marking it indeterminate is
    more honest than silently dropping the interrupted evidence.
    """

    cusum = np.asarray(cusum)
    finite = np.asarray(finite, dtype=bool)
    if finite.size != cusum.size:
        raise ValueError(f"finite mask length differs from CUSUM length: {finite.size} != {cusum.size}")

    gap = 0
    above_before_gap = False
    for i, is_finite in enumerate(finite):
        if is_finite:
            if gap > int(gap_reset) and above_before_gap:
                return True
            gap = 0
            above_before_gap = False
            continue
        if gap == 0:
            above_before_gap = i > 0 and cusum[i - 1] > float(h)
        gap += 1
    return gap > int(gap_reset) and above_before_gap


def first_run_above(cusum, h, m_consecutive, finite=None, gap_reset=5):
    """Return the first index of a sustained CUSUM excursion.

    Huttunen-Heikinmaa et al. require a 30-sample run for their 1-minute SEP
    application. Returning the first sample of that run keeps the reported
    onset at the beginning of the statistically sustained rise. Short gaps do
    not count as confirming samples; long gaps reset the run.
    """

    cusum = np.asarray(cusum)
    if finite is None:
        finite = np.ones(cusum.size, dtype=bool)
    else:
        finite = np.asarray(finite, dtype=bool)
        if finite.size != cusum.size:
            raise ValueError(f"finite mask length differs from CUSUM length: {finite.size} != {cusum.size}")

    run = 0
    run_start = None
    gap = 0
    for i, value in enumerate(cusum):
        if not finite[i]:
            gap += 1
            if gap > int(gap_reset):
                run = 0
                run_start = None
            continue
        gap = 0
        if value > float(h):
            if run == 0:
                run_start = i
            run += 1
        else:
            run = 0
            run_start = None
        if run >= int(m_consecutive):
            return run_start
    return None


flux = globals().get("flux")
if flux is None:
    raise ValueError("bind `flux` (load_data(...)) before this recipe")

component = int(globals().get("component", 0))
bg_hours = float(globals().get("bg_hours", 2.0))
method = str(globals().get("method", "poisson")).lower()
robust = bool(globals().get("robust", True))
n_sigma = float(globals().get("n_sigma", 2.0))
h_sigma = float(globals().get("h_sigma", 2.0))
h = globals().get("h", None)
m_consecutive = int(globals().get("m_consecutive", 30))
gap_reset = int(globals().get("gap_reset", 5))
flux_units = globals().get("units") or getattr(flux, "units", "") or ""

if method not in {"poisson", "zscore"}:
    raise ValueError(f"unknown SEP onset method {method!r}; use 'poisson' or 'zscore'")
if n_sigma <= 0.0:
    raise ValueError("n_sigma must be positive")
if h_sigma <= 0.0:
    raise ValueError("h_sigma must be positive")
if m_consecutive < 1:
    raise ValueError("m_consecutive must be at least one sample")
if gap_reset < 0:
    raise ValueError("gap_reset must be non-negative")

method_used = method

t = np.asarray(flux.time)
x = np.asarray(flux.values, dtype=float)
if x.ndim > 1:
    x = x[:, component]
finite = np.isfinite(x)

mu, sigma, bg_mask = background_stats(x, t, bg_hours, robust)
mu_d = mu + n_sigma * sigma
detect_start = int(np.where(bg_mask)[0][-1]) + 1

cusum = np.zeros_like(x, dtype=float)
if detect_start < x.size:
    if method_used == "poisson":
        cusum[detect_start:] = cusum_poisson(x[detect_start:], mu, mu_d, gap_reset=gap_reset)
        threshold = float(h) if h is not None else h_sigma * sigma
        threshold_label = f"h = {threshold:g}"
    else:
        k_raw = _poisson_reference_value(mu, mu_d) if mu > 0.0 else mu_d / 2.0
        k = (k_raw - mu) / sigma
        cusum[detect_start:] = cusum_zscore(x[detect_start:], mu, sigma, k, gap_reset=gap_reset)
        threshold = float(h) if h is not None else h_sigma
        threshold_label = f"h = {threshold:g}"
else:
    threshold = float(h) if h is not None else (h_sigma * sigma if method_used == "poisson" else h_sigma)
    threshold_label = f"h = {threshold:g}"

onset_indeterminate = evidence_straddles_long_gap(cusum, threshold, finite, gap_reset=gap_reset)
onset_idx = first_run_above(cusum, threshold, m_consecutive, finite=finite, gap_reset=gap_reset)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
ax1.plot(t, x, lw=0.8)
ax1.axvspan(t[0], t[np.where(bg_mask)[0][-1]], alpha=0.15, label="background")
ax1.set_ylabel("intensity")
if finite.any() and np.nanmin(x[finite]) > 0:
    ax1.set_yscale("log")
ax2.plot(t, cusum, lw=0.8)
ax2.axhline(threshold, ls="--", lw=0.8, label=threshold_label)
ax2.set_ylabel("Poisson CUSUM" if method_used == "poisson" else "z-score CUSUM")
ax2.set_xlabel("time (UTC)")

if onset_idx is not None:
    onset_time = str(t[onset_idx].astype("datetime64[s]")).replace(" ", "T")
    for ax in (ax1, ax2):
        ax.axvline(t[onset_idx], color="tab:red", lw=1.2)
    ax1.set_title(f"SEP onset ({method_used} CUSUM): {onset_time}")
    print(f"onset_time = {onset_time}")
else:
    onset_time = None
    ax1.set_title(f"SEP onset ({method_used} CUSUM): no onset found")
    print("no onset found — lower h/h_sigma/n_sigma or check the background window")
if onset_indeterminate:
    print("onset_indeterminate = True — CUSUM evidence crossed a gap longer than gap_reset")

ax1.legend(loc="upper left")
ax2.legend(loc="upper left")
plt.tight_layout()
plt.show()

export("cusum", cusum, flux_units if method_used == "poisson" else "")  # noqa: F821 — provided by the sandbox preamble
export("onset_index", np.nan if onset_idx is None else onset_idx, "")  # noqa: F821
