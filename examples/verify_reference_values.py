#!/usr/bin/env python3
"""Re-measure every reference value quoted in 02_stpatrick_storm_2015.ipynb.

The notebook states its reference values are measured rather than quoted. This is
the script that measures them, so the claim is checkable and so a change in the
upstream archives shows up as a diff rather than as a notebook that silently
disagrees with itself.

Needs only speasy and numpy. Run it and compare against the tables in the
notebook:

    python examples/verify_reference_values.py
"""

from __future__ import annotations

import numpy as np
import speasy as spz

WINDOW = ("2015-03-16T18:00:00", "2015-03-18T12:00:00")
SHOCK_SEARCH = ("2015-03-17T03:30:00", "2015-03-17T05:00:00")

MU0 = 4 * np.pi * 1e-7
M_P = 1.67262192e-27
K_B = 1.380649e-23
EV_PER_K = 8.617333e-5
R_E_KM = 6371.0

B_WIND = "cda/WI_H0_MFI/BGSM"
B_ACE = "cda/AC_H0_MFI/BGSM"
N_WIND = "cda/WI_H1_SWE/Proton_Np_moment"
V_WIND = "cda/WI_H1_SWE/Proton_V_moment"
W_WIND = "cda/WI_H1_SWE/Proton_W_moment"


def fetch(param_id: str, start: str, stop: str) -> tuple[np.ndarray, np.ndarray]:
    """Download a parameter and blank its fill values out to NaN.

    Uses the FILLVAL the CDF declares rather than a magnitude cutoff: Wind/SWE
    fills with 99999.9 and ACE with -1e31, so any single hard-coded threshold
    gets one of the two wrong. rtol suits a float32 round-trip of the sentinel.
    """
    var = spz.get_data(param_id, start, stop)
    if var is None or len(var) == 0:
        raise SystemExit(f"no data for {param_id} over {start} → {stop}")
    values = np.asarray(var.values, dtype="float64")
    bad = ~np.isfinite(values) | (np.abs(values) >= 1e30)
    fillval = (getattr(var, "meta", {}) or {}).get("FILLVAL")
    if fillval is not None:
        try:
            fv = float(np.asarray(fillval).ravel()[0])
            if np.isfinite(fv):
                bad |= np.isclose(values, fv, rtol=1e-6)
        except (TypeError, ValueError, IndexError):
            pass
    values[bad] = np.nan
    return np.asarray(var.time), values


def magnitude(vectors: np.ndarray) -> np.ndarray:
    mag = np.sqrt((vectors**2).sum(axis=1))
    mag[np.isnan(vectors).any(axis=1)] = np.nan
    return mag


def invalid_pct(values: np.ndarray) -> float:
    return 100.0 * float(np.isnan(values).mean())


def window_mean(times: np.ndarray, values: np.ndarray, start, stop) -> float:
    sel = (times >= start) & (times < stop) & ~np.isnan(values)
    return float(np.nanmean(values[sel])) if sel.any() else float("nan")


def find_fast_forward_shock(times, n, b_on_n, v, half_width: int = 6):
    """Locate the shock as a SIMULTANEOUS jump in density, field and speed.

    A density jump alone is not a shock: this window contains a larger density
    rise at 12:51 UT with no speed change, which is a compression structure
    inside the driver. Requiring all three to increase is what separates them.
    """
    best_score, best_index = -np.inf, None
    for i in range(half_width, len(times) - half_width):
        before = slice(i - half_width, i)
        after = slice(i, i + half_width)
        n1, n2 = np.nanmean(n[before]), np.nanmean(n[after])
        b1, b2 = np.nanmean(b_on_n[before]), np.nanmean(b_on_n[after])
        v1, v2 = np.nanmean(v[before]), np.nanmean(v[after])
        if not (n2 > n1 and b2 > b1 and v2 > v1):
            continue
        score = (n2 / n1 - 1) + (b2 / b1 - 1) + (v2 / v1 - 1)
        if score > best_score:
            best_score, best_index = score, i
    if best_index is None:
        raise SystemExit("no fast forward shock found in the window")
    return times[best_index], best_score


def arrival_from_field(param_id: str) -> np.datetime64:
    """Shock arrival at one spacecraft, from the steepest smoothed |B| rise."""
    times, vectors = fetch(param_id, *SHOCK_SEARCH)
    mag = magnitude(vectors)
    ok = ~np.isnan(mag)
    times, mag = times[ok], mag[ok]
    smoothed = np.convolve(mag, np.ones(5) / 5, mode="same")
    return times[int(np.nanargmax(np.diff(smoothed))) + 1]


def main() -> None:
    tb, bvec = fetch(B_WIND, *WINDOW)
    bmag = magnitude(bvec)
    tn, n = fetch(N_WIND, *WINDOW)
    tv, v = fetch(V_WIND, *WINDOW)
    tw, w = fetch(W_WIND, *WINDOW)
    n, v, w = n.ravel(), v.ravel(), w.ravel()
    temp_k = M_P * (w * 1e3) ** 2 / (2 * K_B)

    ta, avec = fetch(B_ACE, *WINDOW)
    amag = magnitude(avec)

    print(f"window {WINDOW[0]} → {WINDOW[1]}\n")
    print("coverage (invalid %)")
    for label, values in [
        ("Wind MFI  BGSM", bmag),
        ("Wind SWE  Np_moment", n),
        ("Wind SWE  V_moment", v),
        ("Wind SWE  W_moment", w),
        ("ACE  MAG  BGSM", amag),
    ]:
        print(f"  {label:<22} {invalid_pct(values):5.1f}%")

    ok = ~np.isnan(n)
    b_on_n = np.interp(
        tn.astype("datetime64[s]").astype(float),
        tb[~np.isnan(bmag)].astype("datetime64[s]").astype(float),
        bmag[~np.isnan(bmag)],
    )
    shock, score = find_fast_forward_shock(tn[ok], n[ok], b_on_n[ok], v[ok])
    print(f"\nfast forward shock at {str(shock)[:19]}  (jump score {score:.2f})")

    hour, half_hour = np.timedelta64(1, "h"), np.timedelta64(30, "m")
    up = (shock - hour, shock)
    down = (shock, shock + half_hour)
    state = {}
    for label, (t0, t1) in [("upstream (1 h before)", up), ("downstream (30 min after)", down)]:
        state[label] = {
            "B": window_mean(tb, bmag, t0, t1),
            "n": window_mean(tn, n, t0, t1),
            "V": window_mean(tv, v, t0, t1),
            "T": window_mean(tw, temp_k, t0, t1),
        }
        s = state[label]
        print(
            f"  {label:<26} |B|={s['B']:6.2f} nT  n={s['n']:6.2f} cm-3  "
            f"V={s['V']:7.1f} km/s  T={s['T'] / 1e3:6.1f} kK"
        )

    a, b = state["upstream (1 h before)"], state["downstream (30 min after)"]
    r_n, r_b = b["n"] / a["n"], b["B"] / a["B"]
    v_a = a["B"] * 1e-9 / np.sqrt(MU0 * M_P * a["n"] * 1e6) / 1e3
    c_s = np.sqrt(5 / 3 * K_B * a["T"] / M_P) / 1e3

    # Shock speed from mass-flux conservation, n_u(V_u - V_sh) = n_d(V_d - V_sh).
    # NOT V_d * r/(r-1): that form assumes the upstream plasma is at rest, and here
    # it is flowing at ~410 km/s, which inflates the answer from 579 to 838 km/s.
    v_shock = (b["n"] * b["V"] - a["n"] * a["V"]) / (b["n"] - a["n"])
    # The Mach number needs the shock speed IN THE UPSTREAM FRAME, not in the
    # spacecraft frame — dividing the latter by V_A gave 16 instead of 3.2.
    m_a = (v_shock - a["V"]) / v_a

    # Independent check: MHD predicts the compression a given M_A should produce.
    # r = 3.94 at M_A = 16 against a measured 2.59 is how the wrong M_A shows up.
    gamma = 5 / 3
    r_predicted = (gamma + 1) * m_a**2 / ((gamma - 1) * m_a**2 + 2)

    print(f"\n  r_n = {r_n:.2f}   r_B = {r_b:.2f}   (agreement is the sanity check)")
    print(f"  V_A = {v_a:.1f} km/s   c_s = {c_s:.1f} km/s")
    print(f"  V_shock = {v_shock:.0f} km/s in the spacecraft frame,")
    print(f"           {v_shock - a['V']:.0f} km/s relative to the upstream flow")
    print(f"  M_A = {m_a:.2f}  →  MHD predicts r = {r_predicted:.2f}, measured {r_n:.2f}")

    day = (tb >= np.datetime64("2015-03-17T00:00")) & (tb < np.datetime64("2015-03-18T00:00"))
    j = int(np.nanargmin(bvec[day][:, 2]))
    print(f"\n  Wind: |B| max = {np.nanmax(bmag[day]):.2f} nT")
    print(f"        Bz min  = {np.nanmin(bvec[day][:, 2]):.2f} nT at {str(tb[day][j])[:19]}")
    print(f"        V max = {np.nanmax(v):.1f} km/s    n max = {np.nanmax(n):.1f} cm-3")

    wind_arrival, ace_arrival = arrival_from_field(B_WIND), arrival_from_field(B_ACE)
    lag = (ace_arrival - wind_arrival) / np.timedelta64(1, "s")
    print("\ntwo-spacecraft timing")
    print(f"  Wind sees the shock at {str(wind_arrival)[:19]}")
    print(f"  ACE  sees the shock at {str(ace_arrival)[:19]}")
    print(f"  lag (ACE - Wind) = {lag:+.0f} s")

    t_pos, ace_pos = fetch("amda/ace_xyz_gse", *SHOCK_SEARCH)
    _, wind_pos = fetch("amda/wnd_xyz_gse", *SHOCK_SEARCH)
    r_ace, r_wind = ace_pos[0], wind_pos[0]
    dr = (r_ace - r_wind) * R_E_KM
    print(f"  ACE  position = {np.round(r_ace, 1)} R_E")
    print(f"  Wind position = {np.round(r_wind, 1)} R_E")
    print(f"  dR (ACE - Wind) = {np.round(dr / R_E_KM, 1)} R_E")
    if lag > 0:
        apparent = abs(dr[0]) / lag
        print(f"  apparent speed along X = {apparent:.0f} km/s")
        print(f"  vs V_shock from jump conditions = {v_shock:.0f} km/s")
        print("  the excess is the front's tilt, not an error")


if __name__ == "__main__":
    main()
