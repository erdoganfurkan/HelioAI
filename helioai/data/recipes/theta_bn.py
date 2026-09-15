# name: theta_bn
# description: Compute the shock normal angle theta_Bn from upstream and downstream magnetic field vectors.
# inputs: B_up (array of shape (N,3) or (3,) in nT, upstream), B_dn (array of shape (N,3) or (3,), downstream); or B (series with .time and .values) plus shock_time; or B alone to list shock candidates; optional guard_min, span_min, n_candidates
# outputs: theta_bn (deg), shock_normal, compression_ratio (magnetic |<B_dn>|/|<B_up>|), B_up_mean_nT, B_dn_mean_nT, theta_bn_std_deg, normal_spread_deg, Bn_std_nT
# reference: Coplanarity theorem (Colburn & Sonett 1966); Schwartz (1998), "Shock and Discontinuity Normals, Mach Numbers and Related Parameters", ISSI SR-001, ch. 10.

"""Shock normal angle theta_Bn.

theta_Bn is the angle between the upstream magnetic field and the shock normal vector n.
The normal n is estimated from the magnetic coplanarity theorem:
    n = (B_dn × B_up) × (B_dn - B_up)
    n = n / |n|

theta_Bn < 45° → quasi-parallel shock (field-aligned)
theta_Bn > 45° → quasi-perpendicular shock

Usage with a measured crossing time, preferred because the recipe owns the windows:
    B = load_data("b3gsm")
    shock_time = np.datetime64("2015-03-17T04:00:00")
    # Then run this script. It uses 8-minute windows separated from the shock by
    # the 2-minute guard band, rejecting windows that still contain the ramp.

Usage when the crossing time is not known yet — find it first, do not hunt for it with
hand-written run_python cells (one live run spent nine of its twelve turns on that):
    B = load_data("b3gsm")
    # Run this script with B alone: it prints the n_candidates largest |B| jumps over
    # one-minute windows (time, jump in nT, downstream/upstream ratio) and stops. Look
    # at them against the plot, pick one, then run again with shock_time set to it.
    # The same list is available as a function: find_shock_candidates(B, n=5).

Usage with windows already chosen by the analyst:
    B_up = var.values[mask_up]     # (N,3) over the upstream interval, or its mean 3-vector
    B_dn = var.values[mask_dn]     # (N,3) over the downstream interval, or its mean 3-vector
    # Then run this script: it uses B_up / B_dn when both are already defined.

An (N, 3) input is averaged over its finite rows — fill values are NaN by the time the
data reaches you, and a plain mean of a gapped interval is NaN. A NaN, zero or collinear
input returns {"error": ...} and nothing else: no angle, no geometry.

A single-number "coplanarity residual" computed from the two mean vectors is not a real
diagnostic here: (B_dn × B_up) · n̂ and (B_dn - B_up) · n̂ are both exactly zero by
construction for the coplanarity normal. When full upstream/downstream series are passed,
the recipe reports repeatability diagnostics instead: bootstrap scatter in theta_Bn, the
68th-percentile angular spread of the bootstrap normals, and the standard deviation of
B · n̂ across both windows. Mean-vector inputs report these diagnostics as None and do not
export them.

The bootstrap resamples rows i.i.d.; that understates uncertainty for correlated plasma
data, and it does not measure the live window-choice spread (54.85-64.27° on one shock).
The shock_time path addresses that separate failure by fixing the windows and applying
abrupt-step and trend checks. WINDOW_TREND_MAX is the allowed first-half to second-half
|B| trend as a fraction of the between-window |B| jump. Those checks are heuristics,
not proof that a window is stationary.
"""

import numpy as np

GUARD_MIN = 2.0
SPAN_MIN = 8.0
WINDOW_TREND_MAX = 0.25

if __name__ == "__main__" and "export" not in globals():

    def export(name, data, units=""):
        """No-op outside the HelioAI sandbox so the recipe can be run standalone."""
        return None


def _finite_row_mask(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return np.zeros(arr.shape[0] if arr.ndim else 0, dtype=bool)
    return np.isfinite(arr).all(axis=1) & (np.abs(arr) < 1e30).all(axis=1)


def _to_vec(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        finite_rows = _finite_row_mask(arr)
        if not finite_rows.any():
            return np.full(3, np.nan)
        return arr[finite_rows].mean(axis=0)
    return arr.ravel()[:3]


def _solve_from_means(u, d):
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
        "theta_bn_deg": float(angle_deg),
        "geometry": geometry,
        "shock_normal": n_hat,
        "compression_ratio": float(np.linalg.norm(d) / np.linalg.norm(u)),
        "B_up_mean_nT": u,
        "B_dn_mean_nT": d,
    }


def _series_diagnostics(B_up, B_dn, n_hat):
    up = np.asarray(B_up, dtype=float)
    dn = np.asarray(B_dn, dtype=float)
    if up.ndim != 2 or dn.ndim != 2 or up.shape[1:] != (3,) or dn.shape[1:] != (3,):
        return {"theta_bn_std_deg": None, "normal_spread_deg": None, "Bn_std_nT": None}

    up_rows = up[_finite_row_mask(up)]
    dn_rows = dn[_finite_row_mask(dn)]
    if len(up_rows) < 5 or len(dn_rows) < 5:
        return {"theta_bn_std_deg": None, "normal_spread_deg": None, "Bn_std_nT": None}

    rng = np.random.default_rng(0)
    angles = []
    normals = []
    for _ in range(200):
        u_sample = up_rows[rng.integers(0, len(up_rows), len(up_rows))]
        d_sample = dn_rows[rng.integers(0, len(dn_rows), len(dn_rows))]
        sample = _solve_from_means(u_sample.mean(axis=0), d_sample.mean(axis=0))
        if "error" not in sample:
            angles.append(sample["theta_bn_deg"])
            normals.append(sample["shock_normal"])

    if not angles:
        return {"theta_bn_std_deg": None, "normal_spread_deg": None, "Bn_std_nT": None}

    normals = np.asarray(normals)
    dots = np.clip(np.abs(normals @ n_hat), -1.0, 1.0)
    spreads = np.degrees(np.arccos(dots))
    b_normal = np.concatenate([up_rows @ n_hat, dn_rows @ n_hat])
    return {
        "theta_bn_std_deg": float(np.std(angles)),
        "normal_spread_deg": float(np.percentile(spreads, 68)),
        "Bn_std_nT": float(np.std(b_normal)),
    }


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
    compression_ratio (the magnetic compression |<B_dn>| / |<B_up>|),
    B_up_mean_nT and B_dn_mean_nT (the vectors actually used), and series-only diagnostics theta_bn_std_deg,
    normal_spread_deg and Bn_std_nT. On a NaN or zero input, or on collinear or
    identical vectors, the normal is undefined and the dict carries an "error"
    key and nothing else: a NaN used to flow through to theta_bn_deg and,
    `nan < 45` being False, come back labelled quasi-perpendicular — a physical
    classification of an angle that does not exist.

    The angle is taken through |cos| because the coplanarity normal has an
    arbitrary sign: n and -n describe the same shock, so theta_Bn is folded into
    0-90 deg. Anything downstream that needs the normal's direction (a shock
    speed, a Mach number) must fix the sign itself from the flow.
    """
    u = _to_vec(B_up)
    d = _to_vec(B_dn)
    result = _solve_from_means(u, d)
    if "error" in result:
        return result

    diagnostics = _series_diagnostics(B_up, B_dn, result["shock_normal"])
    return {
        "theta_bn_deg": round(result["theta_bn_deg"], 2),
        "geometry": result["geometry"],
        "shock_normal": result["shock_normal"].tolist(),
        "compression_ratio": result["compression_ratio"],
        "B_up_mean_nT": result["B_up_mean_nT"].tolist(),
        "B_dn_mean_nT": result["B_dn_mean_nT"].tolist(),
        **diagnostics,
    }


CANDIDATE_STEP_MIN = 1.0


def find_shock_candidates(B, n=5, search=None, step_min=CANDIDATE_STEP_MIN):
    """The n largest |B| jumps of a series, as candidate shock crossings.

    A fast forward shock is a step in |B| of a few nT completed within seconds; on a
    day of 3-s data the biggest one-minute jumps are where to look. This is a screening
    list, not a detection: a discontinuity, a sheath structure or a data edge can jump
    too, and a reverse or slow shock steps down. Look at the candidates on the plot and
    choose; the recipe never chooses for you.

    Parameters
    ----------
    B : series with `.time` (datetime64) and `.values` ((N,) or (N, 3)), as `load_data`
        returns it, or a `(t, values)` pair.
    n : how many candidates to return, best first.
    search : optional (start, stop) datetime64 pair restricting the search.
    step_min : the window, in minutes, over which a jump is measured.

    Returns
    -------
    list of dicts {time, jump_nT, ratio, B_before_nT, B_after_nT}, largest jump first.
    A step spreads over every sample within `step_min` of it on either side, so
    candidates closer than 2·step_min to a larger one are the same step and are merged.
    """
    if isinstance(B, tuple):
        t, values = B
    else:
        t, values = B.time, B.values
    t = np.asarray(t)
    v = np.asarray(values, dtype=float)
    mag = np.linalg.norm(v, axis=1) if v.ndim > 1 else v
    if search is not None:
        keep = (t >= np.datetime64(search[0])) & (t <= np.datetime64(search[1]))
        t, mag = t[keep], mag[keep]
    if t.size < 3:
        return []
    step = np.timedelta64(int(step_min * 60), "s")
    t_s = t.astype("datetime64[s]")
    before = np.searchsorted(t_s, t_s - step, side="left")
    after = np.searchsorted(t_s, t_s + step, side="right") - 1
    finite = np.isfinite(mag)
    jumps = np.full(mag.shape, np.nan)
    ok = finite & finite[before] & finite[after] & (after > before)
    jumps[ok] = mag[after][ok] - mag[before][ok]
    order = np.argsort(-np.nan_to_num(jumps, nan=-np.inf))
    out = []
    for i in order:
        if not np.isfinite(jumps[i]) or jumps[i] <= 0:
            break
        if any(abs(t_s[i] - np.datetime64(c["time"], "s")) <= 2 * step for c in out):
            continue
        b0, b1 = float(mag[before[i]]), float(mag[after[i]])
        out.append(
            {
                "time": str(t_s[i]),
                "jump_nT": round(float(jumps[i]), 3),
                "ratio": round(b1 / b0, 3) if b0 > 0 else float("nan"),
                "B_before_nT": round(b0, 3),
                "B_after_nT": round(b1, 3),
            }
        )
        if len(out) >= n:
            break
    return out


def shock_windows(shock_time, guard_min: float = GUARD_MIN, span_min: float = SPAN_MIN):
    """The four magnetic-field averaging window edges, derived from the shock time alone.

    Parameters
    ----------
    shock_time : numpy datetime64 crossing time.
    guard_min  : minutes skipped either side of the crossing, so the ramp and its
                 foot/overshoot do not land in an average.
    span_min   : length in minutes of each averaging window.

    Returns
    -------
    (up_start, up_stop, dn_start, dn_stop) as datetime64, ordered in time.
    """
    t0 = np.datetime64(shock_time)
    guard = np.timedelta64(int(guard_min * 60), "s")
    span = np.timedelta64(int(span_min * 60), "s")
    return (t0 - guard - span, t0 - guard, t0 + guard, t0 + guard + span)


def _window_rows(t, values, t0, t1, label):
    inside = (t >= t0) & (t <= t1)
    rows = values[inside]
    finite = _finite_row_mask(rows)
    n = int(finite.sum())
    if n < 3:
        raise ValueError(f"{label} window has only {n} finite rows; need at least 3")
    return rows, rows[finite]


def _max_magnitude_step(rows):
    if len(rows) < 2:
        return 0.0
    mag = np.linalg.norm(rows, axis=1)
    valid = np.isfinite(mag) & (np.abs(mag) < 1e30)
    if not (valid[1:] & valid[:-1]).any():
        return 0.0
    jumps = np.abs(np.diff(mag))
    return float(np.max(jumps[valid[1:] & valid[:-1]]))


def _magnitude_trend(rows):
    mag = np.linalg.norm(rows, axis=1)
    valid_mag = mag[np.isfinite(mag) & (np.abs(mag) < 1e30)]
    if len(valid_mag) < 3:
        return 0.0
    split = len(valid_mag) // 2
    return float(abs(np.mean(valid_mag[split:]) - np.mean(valid_mag[:split])))


def windows_from_series(B, shock_time, guard_min=GUARD_MIN, span_min=SPAN_MIN):
    """Select upstream/downstream magnetic-field rows from a series and a shock time.

    The caller supplies only the crossing time. This function owns the averaging
    windows, rejects windows with too few finite vector samples, and refuses a
    window whose largest single-step |B| jump or first-half/second-half |B|
    trend looks like the shock ramp. These are abrupt-step and trend checks,
    not proof that a window is stationary.

    Parameters
    ----------
    B          : object with .time datetime64 array and .values shaped (N, 3), in nT.
    shock_time : numpy datetime64 crossing time.
    guard_min  : minutes skipped either side of the crossing.
    span_min   : length in minutes of each averaging window.

    Returns
    -------
    (B_up_rows, B_dn_rows, (up_start, up_stop, dn_start, dn_stop)) with finite rows only.
    """
    if not hasattr(B, "time") or not hasattr(B, "values"):
        raise ValueError("B must be a series object with .time and .values")
    t = np.asarray(B.time)
    values = np.asarray(B.values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"B.values must have shape (N, 3), got {values.shape}")
    if t.shape[0] != values.shape[0]:
        raise ValueError("B.time and B.values must have the same length")

    windows = shock_windows(shock_time, guard_min, span_min)
    u_raw, u_rows = _window_rows(t, values, windows[0], windows[1], "upstream")
    d_raw, d_rows = _window_rows(t, values, windows[2], windows[3], "downstream")

    total_jump = abs(np.linalg.norm(_to_vec(d_rows)) - np.linalg.norm(_to_vec(u_rows)))
    if total_jump > 1e-12:
        for label, rows in (("upstream", u_raw), ("downstream", d_raw)):
            if _max_magnitude_step(rows) > 0.5 * total_jump:
                raise ValueError(
                    f"{label} window overlaps the ramp — move shock_time or increase guard_min"
                )
            trend = _magnitude_trend(rows)
            if trend > WINDOW_TREND_MAX * total_jump:
                raise ValueError(
                    f"{label} window is not stationary (trend of {trend:.2f} nT across it) "
                    "— it likely contains the ramp; move shock_time or increase guard_min"
                )

    return u_rows, d_rows, windows


def _export_result(result):
    export("theta_bn", np.array([result["theta_bn_deg"]]), "deg")
    export("shock_normal", np.asarray(result["shock_normal"]), "")
    export("compression_ratio", np.array([result["compression_ratio"]]), "")
    export("B_up_mean_nT", np.asarray(result["B_up_mean_nT"]), "nT")
    export("B_dn_mean_nT", np.asarray(result["B_dn_mean_nT"]), "nT")
    if result.get("theta_bn_std_deg") is not None:
        export("theta_bn_std_deg", np.array([result["theta_bn_std_deg"]]), "deg")
    if result.get("normal_spread_deg") is not None:
        export("normal_spread_deg", np.array([result["normal_spread_deg"]]), "deg")
    if result.get("Bn_std_nT") is not None:
        export("Bn_std_nT", np.array([result["Bn_std_nT"]]), "nT")


def _fmt_time(t):
    return np.datetime_as_string(np.datetime64(t), unit="s")


# ── Run ────────────────────────────────────────────────────────────────────────
# Preferred live use derives the windows from a measured shock_time:
#   B = load_data("b3gsm")                       # .time, .values (N, 3), fill already NaN
#   shock_time = np.datetime64("2015-03-17T04:00:00")
# Analyst-chosen windows are still accepted for backwards compatibility:
#   B_up = var.values[(var.time >= t_up0) & (var.time <= t_up1)]
#   B_dn = var.values[(var.time >= t_dn0) & (var.time <= t_dn1)]
# The placeholder demo below is only for a bare `python theta_bn.py` run.

_B_up = globals().get("B_up")
_B_dn = globals().get("B_dn")
_B = globals().get("B")
_shock_time = globals().get("shock_time")
_guard_min = globals().get("guard_min", GUARD_MIN)
_span_min = globals().get("span_min", SPAN_MIN)
result = None

try:
    if _B_up is not None and _B_dn is not None:
        B_up = _B_up
        B_dn = _B_dn
    elif _B is not None and _shock_time is not None:
        B_up, B_dn, _windows = windows_from_series(_B, _shock_time, _guard_min, _span_min)
        print(f"upstream window: {_fmt_time(_windows[0])} to {_fmt_time(_windows[1])}")
        print(f"downstream window: {_fmt_time(_windows[2])} to {_fmt_time(_windows[3])}")
    elif _B is not None:
        shock_candidates = find_shock_candidates(_B, n=int(globals().get("n_candidates", 5)))
        if not shock_candidates:
            print("theta_bn: no |B| jump found in B — check the interval and the data")
        else:
            print("theta_bn: shock_time not set — the largest |B| jumps, to choose from:")
            for _c in shock_candidates:
                print(
                    f"  {_c['time']}  +{_c['jump_nT']} nT  "
                    f"({_c['B_before_nT']} → {_c['B_after_nT']} nT, ratio {_c['ratio']})"
                )
            print("pick one, then run again with shock_time = np.datetime64('<time>')")
    else:
        if __name__ == "__main__":
            B_up = np.array([5.0, -2.0, 1.0])
            B_dn = np.array([15.0, -8.0, 4.0])
        else:
            print("theta_bn: define B_up and B_dn, or B and shock_time, before running this recipe")

    if "B_up" in globals() and "B_dn" in globals():
        result = theta_bn(B_up, B_dn)
except ValueError as _e:
    result = {"error": str(_e)}

if result is None:
    pass
elif "error" in result:
    print("theta_bn:", result["error"])
else:
    _export_result(result)
    print(result)
