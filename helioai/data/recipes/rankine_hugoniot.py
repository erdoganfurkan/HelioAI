# name: rankine_hugoniot
# description: Rankine-Hugoniot analysis of a collisionless shock. Derives the upstream/downstream averaging windows from the shock time itself, then computes compression ratio, shock speed and Mach numbers, and checks them against each other.
# inputs: shock_time (numpy datetime64), and per quantity a (time, values) pair — magnetic field magnitude nT, proton density cm-3, bulk speed km/s, temperature eV (optional). Each may have its own cadence.
# outputs: V_shock (km/s, spacecraft frame), r (density compression ratio), plus the windows used and a consistency verdict
# reference: Rankine-Hugoniot jump conditions (Rankine 1870; Hugoniot 1887); for collisionless shocks see Schwartz (1998), ISSI SR-001, ch. 10 — including the averaging-window guidance this recipe implements.

"""Rankine-Hugoniot jump conditions for a collisionless shock.

Applies the conservation laws across the shock:
  1. Mass flux:         r = n_d / n_u  (compression ratio)
  2. Momentum flux:     n m (V - V_sh)^2 + P conserved
  3. Magnetic flux:     B_d / B_u = r  (perpendicular shock, exact)

**This recipe picks the averaging windows itself, and that is the point.** Choosing them
by hand is where the analysis actually goes wrong: one run averaged the downstream side
from shock+30 min out to shock+120 min, landing in the decaying sheath rather than the
shocked plasma. Density came out 32 cm-3 instead of 45, the compression ratio 1.89
instead of 2.59, and the Alfvenic Mach number 4.6 instead of 3.2 — every downstream
number wrong, from one plausible-sounding choice. Do not pass your own averages.

Usage: load_recipe("rankine_hugoniot") returns this source. Paste it into run_python and
call, with each quantity on its own time base (B is typically 3 s, plasma moments 92 s):

    n_u, n_d = upstream_downstream(t_n, n, shock_time)
    V_u, V_d = upstream_downstream(t_v, v, shock_time)
    B_u, B_d = upstream_downstream(t_b, bmag, shock_time)
    T_u, T_d = upstream_downstream(t_t, t_ev, shock_time)
    V_shock, r = rh_jump(n_u=n_u, n_d=n_d, V_u=V_u, V_d=V_d, B_u=B_u, B_d=B_d, T_u=T_u, T_d=T_d)

Recommended pipeline when normal vector is determined:
    theta_bn or mvab -> n_hat -> upstream_downstream(t_v, v, shock_time, normal=n_hat)

V_shock comes back FIRST — this example used to read `r, V_shock`, which silently
swapped a 579 km/s speed with a compression ratio of 2.59.

V_shock is in the spacecraft frame, derived from mass-flux conservation. Two things
follow. Do not use V_d * r/(r-1) instead: it assumes the upstream plasma is at rest,
and against a 400 km/s solar wind it overestimates by ~45%. And the Alfvenic Mach
number needs the shock speed in the UPSTREAM frame, (V_shock - V_u)/V_A — dividing
the spacecraft-frame speed by V_A gives a number several times too large.
"""

import numpy as np

MU0 = 4e-7 * np.pi  # H/m
MP = 1.6726e-27  # proton mass kg
EV = 1.6022e-19  # J per eV
CM3_TO_M3 = 1e6  # 1/cm³ = 1e6 /m³
GAMMA = 5.0 / 3.0

# Guard band and window length, in minutes, either side of the shock.
#
# GUARD skips the ramp and the downstream overshoot. At L1 the ramp is seconds and the
# overshoot a couple of minutes — not the quarter of an hour it is tempting to allow.
# SPAN must stay inside the plasma the shock just processed: the sheath starts evolving
# within the hour, and averaging into that decay is what drags the downstream state down.
#
# These are measured, not assumed. Sweeping both against the 2015-03-17 shock and scoring
# on the shock's own internal consistency — for a perpendicular shock the density and
# magnetic compressions must agree, which needs no reference values:
#
#     guard  span     r_n    r_B   apart
#         2    20    2.48   2.53      2%
#         5    20    2.53   2.57      2%   <- adopted
#         5    30    2.37   2.65     10%
#        10    30    2.24   2.66     16%
#        15    30    2.13   2.63     19%
#
# The published compression for this event is 2.59. A generous guard with a long span
# is the worst combination and the most plausible-sounding one, which is exactly how a
# run reported 1.89.
#
# HONEST LIMITATION: that sweep is one event. A second shock (2012-07-14) was tried and
# was inconclusive — Wind plasma moments there are ~3 min apart, so most windows caught
# fewer than three samples and `window_mean` correctly refused to average them. So these
# are CALIBRATED, not derived: they suit a fast forward shock at L1 sampled at 92 s.
# For a slower or weaker shock, or a coarser instrument, sweep them again on your own
# event and score on the r_n vs r_B agreement, which needs no published answer. They are
# arguments for a reason — pass your own rather than trusting these.
#
# VELOCITY NORMAL PROJECTION: When `normal=None`, `upstream_downstream` averages the Euclidean
# norm |V|. This is a solid approximation when the shock normal is nearly aligned with the flow
# direction (typical for fast forward shocks at L1) and degrades as 1/cos(theta_Vn) (~+40% error
# on V_shock at theta_Vn = 45 deg). When a shock normal is available from theta_bn or mvab,
# pass `normal=n` to project V·n̂.
GUARD_MIN = 5.0
SPAN_MIN = 20.0


def shock_windows(shock_time, guard_min: float = GUARD_MIN, span_min: float = SPAN_MIN):
    """The four window edges, derived from the shock time alone.

    Returns
    -------
    (up_start, up_stop, dn_start, dn_stop) as datetime64, ordered in time.
    """
    t0 = np.datetime64(shock_time)
    guard = np.timedelta64(int(guard_min * 60), "s")
    span = np.timedelta64(int(span_min * 60), "s")
    return (t0 - guard - span, t0 - guard, t0 + guard, t0 + guard + span)


def window_mean(t, values, t0, t1, min_samples: int = 3, normal=None) -> float:
    """NaN-aware mean of `values` over [t0, t1]. NaN when too few samples land inside.

    When `values` is a vector series (n, 3) and `normal` is provided, projects `values @ normal`
    (with normal normalized to unit length). If `normal` is None, averages the Euclidean norm |v|,
    which is exact for scalar quantities (|B|) and a good approximation for bulk speed when the
    shock normal is nearly aligned with the solar wind flow direction.
    """
    t = np.asarray(t)
    v = np.asarray(values, dtype=float)
    if v.ndim > 1:
        if normal is not None:
            n_hat = np.asarray(normal, dtype=float)
            n_norm = np.linalg.norm(n_hat)
            if n_norm > 0:
                n_hat = n_hat / n_norm
            v = v @ n_hat
        else:
            v = np.linalg.norm(v, axis=1)
    # `isfinite` alone is not enough: the ~1e31 fill convention is a perfectly finite
    # float, and one of them in a window returns a compression ratio of 1e30. Data from
    # load_data() is already blanked; data fetched directly is not, and this recipe gets
    # copied into standalone scripts.
    inside = (t >= t0) & (t <= t1) & np.isfinite(v) & (np.abs(v) < 1e30)
    n = int(inside.sum())
    if n < min_samples:
        print(f"  ! only {n} valid samples in [{t0}, {t1}] — returning NaN")
        return float("nan")
    return float(np.mean(v[inside]))


def upstream_downstream(
    t,
    values,
    shock_time,
    guard_min: float = GUARD_MIN,
    span_min: float = SPAN_MIN,
    normal=None,
):
    """Upstream and downstream means over windows derived from `shock_time`.

    Accepts a vector series (n, 3). If `normal` is provided (e.g. from theta_bn or
    mvab), projects the vector onto the shock normal (V·n̂). If `normal` is None,
    averages its Euclidean magnitude (|B|, |V|).

    Returns
    -------
    (mean_upstream, mean_downstream)
    """
    u0, u1, d0, d1 = shock_windows(shock_time, guard_min, span_min)
    return (
        window_mean(t, values, u0, u1, normal=normal),
        window_mean(t, values, d0, d1, normal=normal),
    )


def _rh_core(n_u, n_d, V_u, V_d, B_u, B_d, T_u=0.0, T_d=0.0) -> dict:
    """Pure computation — no printing, no exports. Everything below is derived here."""
    V_shock = 0.0 if abs(n_d - n_u) < 1e-12 else (n_d * V_d - n_u * V_u) / (n_d - n_u)
    r = n_d / n_u

    n_u_si, n_d_si = n_u * CM3_TO_M3, n_d * CM3_TO_M3
    V_u_si, V_d_si, V_sh_si = V_u * 1e3, V_d * 1e3, V_shock * 1e3
    B_u_si, B_d_si = B_u * 1e-9, B_d * 1e-9

    P_u = n_u_si * T_u * EV + B_u_si**2 / (2 * MU0)
    P_d = n_d_si * T_d * EV + B_d_si**2 / (2 * MU0)
    mom_u = n_u_si * MP * (V_u_si - V_sh_si) ** 2 + P_u
    mom_d = n_d_si * MP * (V_d_si - V_sh_si) ** 2 + P_d
    mom_residual = abs(mom_d - mom_u) / (abs(mom_u) + 1e-30)

    # Upstream frame — the only frame in which the Mach numbers mean anything.
    V_A = B_u_si / np.sqrt(MU0 * n_u_si * MP) / 1e3  # km/s
    c_s = np.sqrt(GAMMA * T_u * EV / MP) / 1e3 if T_u > 0 else float("nan")
    V_ms = np.sqrt(V_A**2 + (0.0 if np.isnan(c_s) else c_s**2))
    U_u = V_shock - V_u
    M_A = U_u / V_A if V_A > 0 else float("nan")
    M_ms = U_u / V_ms if V_ms > 0 else float("nan")

    # The compression a perpendicular shock at this Mach number should show. Gas-dynamic
    # form using the magnetosonic Mach number: an approximation, but close enough that a
    # real disagreement means the windows are wrong, not the physics.
    if np.isfinite(M_ms) and M_ms > 1:
        r_pred = ((GAMMA + 1) * M_ms**2) / ((GAMMA - 1) * M_ms**2 + 2)
    else:
        r_pred = float("nan")
    mismatch = abs(r_pred - r) / r if np.isfinite(r_pred) and r > 0 else float("nan")

    return {
        "V_shock": V_shock,
        "r": r,
        "B_ratio": B_d / B_u,
        "mom_residual": mom_residual,
        "V_A": V_A,
        "c_s": c_s,
        "U_upstream": U_u,
        "M_A": M_A,
        "M_ms": M_ms,
        "r_predicted": r_pred,
        "r_mismatch": mismatch,
        "r_strong_shock_limit": (GAMMA + 1) / (GAMMA - 1),
    }


def rh_jump(
    n_u: float,
    n_d: float,
    V_u: float,
    V_d: float,
    B_u: float,
    B_d: float,
    T_u: float = 0.0,
    T_d: float = 0.0,
) -> tuple[float, float]:
    """Compute shock velocity and compression ratio from RH jump conditions.

    Parameters
    ----------
    n_u, n_d : upstream/downstream number density (cm⁻³)
    V_u, V_d : upstream/downstream bulk speed along shock normal (km/s)
    B_u, B_d : upstream/downstream magnetic field magnitude (nT)
    T_u, T_d : upstream/downstream temperature (eV, optional)

    Returns
    -------
    V_shock : shock velocity in spacecraft frame (km/s)
    r       : density compression ratio n_d / n_u
    """
    c = _rh_core(n_u, n_d, V_u, V_d, B_u, B_d, T_u, T_d)

    for key in (
        "V_shock",
        "r",
        "B_ratio",
        "mom_residual",
        "V_A",
        "c_s",
        "U_upstream",
        "M_A",
        "M_ms",
        "r_predicted",
        "r_mismatch",
    ):
        export(key, np.array([c[key]]))  # noqa: F821 — sandbox preamble

    print(f"Shock velocity   : {c['V_shock']:.1f} km/s (spacecraft frame)")
    print(f"Upstream inflow  : {c['U_upstream']:.1f} km/s (shock frame)")
    print(f"Compression r    : {c['r']:.2f}   (strong-shock limit {c['r_strong_shock_limit']:.0f})")
    print(f"B ratio Bd/Bu    : {c['B_ratio']:.2f}   (expected ~r for a perpendicular shock)")
    print(f"V_A / c_s        : {c['V_A']:.1f} / {c['c_s']:.1f} km/s")
    print(f"M_A / M_ms       : {c['M_A']:.2f} / {c['M_ms']:.2f}")
    print(f"Momentum residual: {c['mom_residual'] * 100:.1f}%  (< 10% = good)")

    # The check the analysis is supposed to make on itself. It was asked for in the
    # prompt, computed, and then not acted on: a run reported r = 1.89 next to a
    # predicted 3.50 without comment. Saying it out loud costs nothing.
    if np.isfinite(c["r_mismatch"]):
        verdict = "consistent" if c["r_mismatch"] <= 0.25 else "INCONSISTENT"
        print(
            f"Compression check: measured {c['r']:.2f} vs {c['r_predicted']:.2f} "
            f"expected from M_ms = {c['M_ms']:.2f} → {verdict} "
            f"({c['r_mismatch'] * 100:.0f}% apart)"
        )
        if c["r_mismatch"] > 0.25:
            print(
                "  ! These disagree. The usual cause is an averaging window that left "
                "the shocked plasma — check the downstream window before trusting "
                "either number, and report the disagreement."
            )

    return c["V_shock"], c["r"]


# ── Self-check: the published 2015-03-17 shock, upstream/downstream from the archive ──
# Pure maths, no sandbox helpers, so it also runs outside run_python.
_ref = _rh_core(
    n_u=17.43, n_d=45.12, V_u=411.3, V_d=514.1, B_u=10.00, B_d=25.27, T_u=8.34, T_d=45.0
)
assert abs(_ref["r"] - 2.59) < 0.02, _ref["r"]
assert abs(_ref["V_shock"] - 579) < 5, _ref["V_shock"]
assert abs(_ref["U_upstream"] - 167) < 5, _ref["U_upstream"]
assert abs(_ref["V_A"] - 52.3) < 1.0, _ref["V_A"]
assert abs(_ref["M_A"] - 3.20) < 0.15, _ref["M_A"]
assert _ref["r_mismatch"] < 0.25, f"the reference event must pass its own check: {_ref}"

# And the run that got it wrong must fail that check rather than sail through.
_bad = _rh_core(n_u=16.99, n_d=32.11, V_u=410.3, V_d=511.9, B_u=8.86, B_d=20.81, T_u=3.42, T_d=15.6)
assert _bad["r_mismatch"] > 0.25, f"a 1.89 compression must be flagged: {_bad}"

_w = shock_windows(np.datetime64("2015-03-17T04:00:59"))
assert str(_w[0]) == "2015-03-17T03:35:59" and str(_w[1]) == "2015-03-17T03:55:59", _w
assert str(_w[2]) == "2015-03-17T04:05:59" and str(_w[3]) == "2015-03-17T04:25:59", _w

# Oblique shock velocity projection vs Euclidean norm check
_t_synth = np.array(
    [
        "2015-03-17T03:40:00",
        "2015-03-17T03:45:00",
        "2015-03-17T03:50:00",
        "2015-03-17T04:10:00",
        "2015-03-17T04:15:00",
        "2015-03-17T04:20:00",
    ],
    dtype="datetime64[s]",
)
# 45-degree flow: Vx = 282.84, Vy = 282.84 (|V| = 400.0 km/s)
_v_synth = np.array(
    [
        [282.8427, 282.8427, 0.0],
        [282.8427, 282.8427, 0.0],
        [282.8427, 282.8427, 0.0],
        [353.5534, 353.5534, 0.0],
        [353.5534, 353.5534, 0.0],
        [353.5534, 353.5534, 0.0],
    ]
)
_n_x = np.array([1.0, 0.0, 0.0])  # shock normal along X
_vu_norm, _vd_norm = upstream_downstream(_t_synth, _v_synth, np.datetime64("2015-03-17T04:00:59"))
_vu_proj, _vd_proj = upstream_downstream(
    _t_synth, _v_synth, np.datetime64("2015-03-17T04:00:59"), normal=_n_x
)
assert abs(_vu_norm - 400.0) < 0.1, _vu_norm
assert abs(_vu_proj - 282.84) < 0.1, _vu_proj
