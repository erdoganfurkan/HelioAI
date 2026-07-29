# name: sep_onset_poisson_cusum
# description: Solar energetic particle (SEP) event onset time determination with the Poisson-CUSUM method — robust against gradual rises and noisy backgrounds.
# inputs: flux — SimpleNamespace(time, values) from load_data("<param>"), e.g. an energetic proton/electron intensity channel; component (int, default 0); bg_hours (background window length in hours from the start, default 2.0); n_sigma (out-of-control level, default 2.0); h_sigma (decision threshold on the normalized CUSUM, default 2.0); m_consecutive (samples that must stay above threshold, default 30)
# outputs: onset_time (ISO string), cusum curve exported, figure flux + CUSUM + onset marker
# reference: Huttunen-Heikinmaa, Valtonen & Laitinen (2005), A&A 442, 673, doi:10.1051/0004-6361:20042620

"""SEP onset with Poisson-CUSUM.

The pre-event background (first `bg_hours` of the interval) gives the
in-control mean mu and std sigma; the intensity is normalized to
z = (x - mu)/sigma. Following the Poisson-CUSUM scheme, the out-of-control
level is mu_d = mu + n_sigma*sigma and the (normalized) reference value
k = (mu_d - mu)/(ln(mu_d) - ln(mu)), shifted to z-units. The cumulative sum
    S_0 = 0,  S_i = max(0, S_{i-1} + z_i - k)
signals the event when it stays above h_sigma for `m_consecutive` samples
(30 by default, as in the reference for 1-min data — this long run is what
rejects background noise excursions); the onset is the first sample of that
run. Thresholds are tunable — see the reference for parameter discussion.

Usage inside run_python:
    flux = load_data("erne_protons")
    component = 0
    # then run this script
"""

import numpy as np
import matplotlib.pyplot as plt

component = int(globals().get("component", 0))
bg_hours = float(globals().get("bg_hours", 2.0))
n_sigma = float(globals().get("n_sigma", 2.0))
h_sigma = float(globals().get("h_sigma", 2.0))
m_consecutive = int(globals().get("m_consecutive", 30))

t = np.asarray(flux.time)  # noqa: F821 — set via load_data before running
x = np.asarray(flux.values, dtype=float)  # noqa: F821
if x.ndim > 1:
    x = x[:, component]
finite = np.isfinite(x)

t_sec = t.astype("datetime64[s]").astype("int64").astype(float)
bg_mask = finite & (t_sec <= t_sec[0] + bg_hours * 3600.0)
if bg_mask.sum() < 5:
    raise ValueError(f"background window too short: {bg_mask.sum()} finite samples")

mu = float(np.mean(x[bg_mask]))
sigma = float(np.std(x[bg_mask]))
if sigma <= 0:
    raise ValueError("flat background (sigma = 0) — pick another background window")

mu_d = mu + n_sigma * sigma
k_raw = (mu_d - mu) / (np.log(mu_d) - np.log(mu)) if mu > 0 else mu_d / 2.0
k = (k_raw - mu) / sigma
threshold = h_sigma

z = (x - mu) / sigma
cusum = np.zeros_like(x)
for i in range(1, x.size):
    zi = z[i] if finite[i] else 0.0
    cusum[i] = max(0.0, cusum[i - 1] + zi - k)

above = cusum > threshold
onset_idx = None
run = 0
for i, flag in enumerate(above):
    run = run + 1 if flag else 0
    if run >= m_consecutive:
        onset_idx = i - m_consecutive + 1
        break

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
ax1.plot(t, x, lw=0.8)
ax1.axvspan(t[0], t[np.where(bg_mask)[0][-1]], alpha=0.15, label="background")
ax1.set_ylabel("intensity")
ax1.set_yscale("log" if np.nanmin(x[finite]) > 0 else "linear")
ax2.plot(t, cusum, lw=0.8)
ax2.axhline(threshold, ls="--", lw=0.8, label=f"h = {h_sigma}·σ")
ax2.set_ylabel("CUSUM")
ax2.set_xlabel("time (UTC)")

if onset_idx is not None:
    onset_time = str(t[onset_idx].astype("datetime64[s]")).replace(" ", "T")
    for ax in (ax1, ax2):
        ax.axvline(t[onset_idx], color="tab:red", lw=1.2)
    ax1.set_title(f"SEP onset (Poisson-CUSUM): {onset_time}")
    print(f"onset_time = {onset_time}")
else:
    onset_time = None
    ax1.set_title("SEP onset (Poisson-CUSUM): no onset found")
    print("no onset found — lower h_sigma/n_sigma or check the background window")

ax1.legend(loc="upper left")
ax2.legend(loc="upper left")
plt.tight_layout()
plt.show()

export("cusum", cusum)  # noqa: F821 — provided by the sandbox preamble
