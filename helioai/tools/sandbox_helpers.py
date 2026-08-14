"""Standalone physics helpers importable inside the sandbox.

MUST NOT import anything from helioai.* — the sandbox masks .env and strips
the environment, so helioai.config would fail fast at import time.

Boundary models are clean-room implementations from the published papers:
- Shue et al. (1998), JGR 103, 17691, doi:10.1029/98JA01103
- Jelinek et al. (2012), JGR 117, A05208, doi:10.1029/2011JA017252
Coordinate transforms wrap geopack (MIT, Tsyganenko models port).
"""

from __future__ import annotations

import numpy as np


def _epoch_seconds(time) -> np.ndarray:
    a = np.asarray(time)
    if a.dtype.kind == "M":
        return a.astype("datetime64[s]").astype("int64").astype(float).reshape(-1)
    if a.dtype.kind in "if":
        return a.astype(float).reshape(-1)
    flat = a.reshape(-1)
    return np.array(
        [np.datetime64(t).astype("datetime64[s]").astype("int64") for t in flat], dtype=float
    )


def transform_coords(time, vectors, frm: str = "gse", to: str = "gsm") -> np.ndarray:
    """Rotate vectors between geocentric frames: gse, gsm, sm, geo, mag, gei.

    time: ISO string(s), datetime(s), numpy datetime64 or epoch seconds (UTC);
    vectors: shape (3,) or (N, 3). Returns the same shape.
    Per-point geopack.recalc — fine up to ~1e4 points.

    Example:
        >>> t = np.array(["2015-03-17T04:00:00"], dtype="datetime64[s]")
        >>> transform_coords(t, np.array([[10.0, 0.0, 0.0]]), "gse", "gsm")
        array([[10., 0., 0.]])   # the X axis is shared by GSE and GSM
    """
    from geopack import geopack as gp

    frm, to = frm.lower(), to.lower()
    to_gsm = {
        "gsm": lambda x, y, z: (x, y, z),
        "gse": lambda x, y, z: gp.gsmgse(x, y, z, -1),
        "sm": lambda x, y, z: gp.smgsm(x, y, z, 1),
        "geo": lambda x, y, z: gp.geogsm(x, y, z, 1),
        "mag": lambda x, y, z: gp.geogsm(*gp.geomag(x, y, z, -1), 1),
        "gei": lambda x, y, z: gp.geogsm(*gp.geigeo(x, y, z, 1), 1),
    }
    from_gsm = {
        "gsm": lambda x, y, z: (x, y, z),
        "gse": lambda x, y, z: gp.gsmgse(x, y, z, 1),
        "sm": lambda x, y, z: gp.smgsm(x, y, z, -1),
        "geo": lambda x, y, z: gp.geogsm(x, y, z, -1),
        "mag": lambda x, y, z: gp.geomag(*gp.geogsm(x, y, z, -1), 1),
        "gei": lambda x, y, z: gp.geigeo(*gp.geogsm(x, y, z, -1), -1),
    }
    if frm not in to_gsm or to not in from_gsm:
        raise ValueError(f"unknown frame: {frm!r} or {to!r} — use one of {sorted(to_gsm)}")

    vec = np.asarray(vectors, dtype=float)
    single = vec.ndim == 1
    vec = np.atleast_2d(vec)
    if vec.shape[1] != 3:
        raise ValueError(f"vectors must be (N, 3), got {vec.shape}")
    ts = _epoch_seconds(time)
    if ts.size == 1:
        ts = np.full(vec.shape[0], ts[0])
    if ts.size != vec.shape[0]:
        raise ValueError(f"{ts.size} times for {vec.shape[0]} vectors")

    out = np.empty_like(vec)
    for i in range(vec.shape[0]):
        gp.recalc(ts[i])
        out[i] = from_gsm[to](*to_gsm[frm](*vec[i]))
    return out[0] if single else out


def mp_shue1998(pdyn_nPa: float, bz_nT: float, theta_deg=None):
    """Shue et al. (1998) magnetopause: r = r0 * (2 / (1 + cos(theta)))**alpha.

    r0 = (10.22 + 1.29*tanh(0.184*(Bz + 8.14))) * Pdyn**(-1/6.6)
    alpha = (0.58 - 0.007*Bz) * (1 + 0.024*ln(Pdyn))
    Returns (theta_deg, r_RE); theta defaults to 0..170 deg. theta is the
    angle from the Earth-Sun line, r in Earth radii (aberrated GSE).
    Reference: Shue et al. (1998), JGR 103, 17691, doi:10.1029/98JA01103.

    Example:
        >>> theta, r = mp_shue1998(2.0, -5.0)   # Pdyn=2 nPa, Bz=-5 nT
        >>> round(float(r[0]), 2), round(float(r[90]), 2)
        (9.81, 15.13)                            # standoff and flank, in R_E
    """
    theta = (
        np.linspace(0.0, 170.0, 171)
        if theta_deg is None
        else np.atleast_1d(np.asarray(theta_deg, dtype=float))
    )
    r0 = (10.22 + 1.29 * np.tanh(0.184 * (bz_nT + 8.14))) * pdyn_nPa ** (-1.0 / 6.6)
    alpha = (0.58 - 0.007 * bz_nT) * (1.0 + 0.024 * np.log(pdyn_nPa))
    r = r0 * (2.0 / (1.0 + np.cos(np.radians(theta)))) ** alpha
    return theta, r


def bs_jelinek2012(pdyn_nPa: float, theta_deg=None):
    """Jelinek et al. (2012) bow shock: parabola rho^2 = 4*R*(R - x) / lam^2.

    R = 15.02 * Pdyn**(-1/6.55) is the subsolar standoff (RE), lam = 1.17.
    Solved in polar form r(theta) from the Earth-Sun line; returns
    (theta_deg, r_RE) with NaN where the parabola has no solution.
    theta defaults to 0..120 deg.
    Reference: Jelinek et al. (2012), JGR 117, A05208, doi:10.1029/2011JA017252.

    Example:
        >>> theta, r = bs_jelinek2012(2.0)   # Pdyn=2 nPa
        >>> round(float(r[0]), 2)
        13.51                                 # subsolar standoff, in R_E
    """
    theta = (
        np.linspace(0.0, 120.0, 121)
        if theta_deg is None
        else np.atleast_1d(np.asarray(theta_deg, dtype=float))
    )
    lam = 1.17
    big_r = 15.02 * pdyn_nPa ** (-1.0 / 6.55)
    s, c = np.sin(np.radians(theta)), np.cos(np.radians(theta))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 2.0 * big_r * (np.sqrt(c**2 + lam**2 * s**2) - c) / (lam**2 * s**2)
    on_axis = np.isclose(s, 0.0)
    r = np.where(on_axis, np.where(c > 0, big_r, np.nan), r)
    return theta, r
