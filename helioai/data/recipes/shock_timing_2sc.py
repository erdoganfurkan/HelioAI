# name: shock_timing_2sc
# description: Two-spacecraft shock timing. Given a shock normal obtained independently (coplanarity/MVA), turns two crossing times and two positions into the shock speed along the normal, and checks it against a Rankine-Hugoniot speed. Refuses to invent the normal.
# inputs: t1, t2 (numpy datetime64 crossing times), r1, r2 (position 3-vectors in km, same frame, at the crossing times), n_hat (shock normal unit 3-vector from theta_bn or mvab), V_shock_rh (km/s, optional, for the consistency check)
# outputs: shock_speed_km_s (along the normal, spacecraft frame), lag_s, along_normal_separation_km, transverse_separation_km, warnings, and a consistency verdict against V_shock_rh
# reference: Russell et al. (1983), "Multiple spacecraft observations of interplanetary shocks", JGR 88, 9941; Schwartz (1998), "Shock and Discontinuity Normals, Mach Numbers and Related Parameters", ISSI SR-001, ch. 10, §10.4 (timing methods).

"""Two-spacecraft shock timing — what it can give you, and what it cannot.

**Two spacecraft cannot determine a shock normal.** One crossing pair gives one scalar,
the delay, and a plane orientation has two free angles: the system is underdetermined.
Four spacecraft close the system (that is what MMS and Cluster are for); with two, the
normal has to come from somewhere else — coplanarity (`theta_bn`) or minimum variance
(`mvab`) on the field itself.

This recipe exists because that constraint is not obvious, and the way it fails is silent.
A run of the St Patrick 2015 notebook wrote its own timing analysis and set

    n = Δr / (V_shock · Δt)        # "shock normal"

which is the separation vector rescaled. Everything computed from it is then an identity:
the along-normal separation comes back as |Δr| and the transverse separation as exactly
0 km, for any input whatsoever. That zero was published as a geometrical result. The
real transverse separation was ~470 000 km, and it was the explanation for the very
discrepancy the analysis had set out to explain.

So `n_hat` is a required argument and it is never derived from the timing.

What the timing legitimately gives, once the normal is known, is the shock speed along
that normal in the spacecraft frame:

    V_n = (Δr · n̂) / Δt

Compare it with the Rankine-Hugoniot speed from the jump conditions: two independent
routes to the same quantity. Agreement is the strongest statement this analysis makes;
disagreement means the normal, the crossing times, or the planar assumption is wrong.

Usage inside run_python, after `rankine_hugoniot` has given you V_shock and `theta_bn`
the normal:

    t1 = shock_time(t_b1, b1)           # or your own, checked against the plot
    t2 = shock_time(t_b2, b2)
    out = timing_2sc(t1, t2, r1_km, r2_km, n_hat, V_shock_rh=585.4)

    export("lag_s", np.array([out["lag_s"]]), units="s")
    export("shock_speed_timing", np.array([out["shock_speed_km_s"]]), units="km/s")
    export("transverse_separation", np.array([out["transverse_separation_km"]]), units="km")
    for w in out["warnings"]:
        print("!", w)
    print(out.get("verdict"), out.get("mismatch"))

With the normal pointing upstream (+X at L1, the usual convention), a fast forward shock
travels along -n, so `V_along_normal` is negative and `shock_speed_km_s` is the comparable
magnitude. Comparing the signed value against a Rankine-Hugoniot speed reports a 168%
mismatch on a case that is fine.

Detect the crossing on a magnitude built with `magnitude()`, never with
`np.sqrt(np.nansum(v**2, axis=1))`: nansum reads a data gap as zero, and the recovery out
of the gap is then the largest jump in the interval, which is how the same run put the
Wind crossing 3.5 minutes early and got a lag of -464.5 s instead of -260.5 s.
"""

import numpy as np

# Below this, the front is running nearly parallel to the separation: the delay carries
# almost no information about the speed and V_n explodes on a few seconds of timing error.
MIN_ALONG_FRACTION = 0.15


def shock_time(t, values, search=None):
    """Crossing time = the largest positive jump of the magnitude, gaps excluded.

    Crude on purpose — a sharp fast forward shock is the one case where the biggest
    gradient is the right answer. Always look at it on the plot before using it; for a
    slow or reverse shock, or a crossing inside a turbulent sheath, pick the time by eye.

    `search` optionally restricts to a (start, stop) datetime64 pair.
    """
    t = np.asarray(t)
    v = magnitude(values) if np.ndim(values) > 1 else clean(values)  # noqa: F821 — sandbox
    if search is not None:
        keep = (t >= np.datetime64(search[0])) & (t <= np.datetime64(search[1]))
        t, v = t[keep], v[keep]
    if t.size < 3:
        return None
    d = np.diff(v)
    if not np.any(np.isfinite(d)):
        return None
    return t[int(np.nanargmax(d)) + 1]


def timing_2sc(t1, t2, r1, r2, n_hat, V_shock_rh=None):
    """Shock speed along a KNOWN normal, from two crossings.

    Parameters
    ----------
    t1, t2 : numpy datetime64 crossing times at spacecraft 1 and 2.
    r1, r2 : position 3-vectors in km, same frame, at those times.
    n_hat  : shock normal unit vector, from `theta_bn` (coplanarity) or `mvab`.
             REQUIRED — it cannot be obtained from two spacecraft.
    V_shock_rh : optional Rankine-Hugoniot shock speed (km/s) to check against.
    """
    n = np.asarray(n_hat, dtype=float).ravel()
    if n.size != 3 or not np.all(np.isfinite(n)) or np.linalg.norm(n) < 1e-12:
        return {
            "error": "n_hat must be a finite 3-vector. Two spacecraft cannot determine a "
                     "shock normal — run theta_bn (coplanarity) or mvab on the field first "
                     "and pass its normal here."
        }
    n = n / np.linalg.norm(n)

    dr = np.asarray(r2, dtype=float).ravel() - np.asarray(r1, dtype=float).ravel()
    dt = float((np.datetime64(t2) - np.datetime64(t1)) / np.timedelta64(1, "s"))
    sep = float(np.linalg.norm(dr))
    along = float(np.dot(dr, n))
    transverse = float(np.linalg.norm(dr - along * n))

    v_signed = float("nan") if abs(dt) < 1e-9 else along / dt
    out = {
        "lag_s": dt,
        "separation_km": sep,
        "along_normal_separation_km": along,
        "transverse_separation_km": transverse,
        "V_along_normal": v_signed,
        "shock_speed_km_s": abs(v_signed),
        "first": "spacecraft 1" if dt > 0 else "spacecraft 2",
        "warnings": [],
    }

    # Sign, not an error: with the normal pointing upstream — the usual convention, and
    # what theta_bn returns once you orient it sunward — a fast forward shock travels
    # along -n, so V_along_normal comes out negative. Only its magnitude is comparable
    # with a Rankine-Hugoniot speed, and comparing the signed value was a 168% "mismatch"
    # on a case that was fine.
    out["propagation"] = "along -n_hat" if v_signed < 0 else "along +n_hat"

    if sep > 0 and abs(along) / sep < MIN_ALONG_FRACTION:
        out["warnings"].append(
            f"the separation is only {abs(along) / sep * 100:.0f}% along the normal — the two "
            "spacecraft sit almost on the same shock plane, so the delay barely constrains "
            "the speed and V_along_normal is dominated by the timing error"
        )
    if abs(transverse) > abs(along):
        out["warnings"].append(
            f"transverse separation ({transverse:.0f} km) exceeds the along-normal one "
            f"({abs(along):.0f} km): the result assumes one flat front across all of that, "
            "and interplanetary shock fronts are rippled on far smaller scales"
        )

    if V_shock_rh is not None and np.isfinite(v_signed):
        rel = abs(abs(v_signed) - abs(V_shock_rh)) / abs(V_shock_rh)
        out["V_shock_rh"] = float(V_shock_rh)
        out["mismatch"] = float(rel)
        out["verdict"] = "consistent" if rel <= 0.25 else "INCONSISTENT"
    return out


# ── Self-check — the real St Patrick 2015 crossing (Wind then ACE) ─────────────
# Wind and ACE GSE positions in km at the crossings, normal from coplanarity on Wind/MFI.

_t1, _t2 = np.datetime64("2015-03-17T04:00:04"), np.datetime64("2015-03-17T04:04:25")
_wind = np.array([1610910.0, 346281.0, 80077.0])
_ace = np.array([1406537.0, -68250.0, -148602.0])
_n = np.array([0.8974, 0.3153, -0.3087])

_out = timing_2sc(_t1, _t2, _wind, _ace, _n, V_shock_rh=585.4)
assert abs(_out["lag_s"] - 261.0) < 1.0, _out["lag_s"]
assert _out["shock_speed_km_s"] > 0, "the comparable speed is a magnitude, never signed"
assert _out["propagation"] == "along -n_hat", _out["propagation"]

# The regression this recipe exists for: a normal that is not the separation direction
# must leave a real transverse component. The code it replaces returned 0 by construction.
assert _out["transverse_separation_km"] > 4e5, _out["transverse_separation_km"]
assert any("transverse" in w for w in _out["warnings"]), _out["warnings"]
assert _out["verdict"] == "INCONSISTENT", "this pair does not pass its own cross-check"

# A normal that IS the separation direction is the degenerate case, and it must show as
# such — zero transverse — rather than being silently accepted as a measurement.
_par = timing_2sc(_t1, _t2, _wind, _ace, _ace - _wind)
assert _par["transverse_separation_km"] < 1.0, _par["transverse_separation_km"]

# And no normal at all is an error, not a guess.
assert "error" in timing_2sc(_t1, _t2, _wind, _ace, [0.0, 0.0, 0.0])
assert "error" in timing_2sc(_t1, _t2, _wind, _ace, [1.0, 2.0])
