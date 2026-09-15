# name: walen_test
# description: Walén test — compare plasma velocity in the de Hoffmann-Teller frame with the Alfvén velocity to identify rotational discontinuities.
# inputs: V (ion velocity array shape (N,3) in km/s), B (magnetic field array shape (N,3) in nT), n_cm3 (ion density array shape (N,) in cm⁻³)
# outputs: walen_slope (dimensionless), walen_R2 (dimensionless), V_HT (km/s)
# reference: Walén (1944), Ark. Mat. Astron. Fys. 30A; Sonnerup et al. (1987), JGR 92, 12137; Khrabrov & Sonnerup (1998), ISSI SR-001, ch. 9; Paschmann & Sonnerup (2008), ISSI SR-008, ch. 3.

"""Walén test for rotational discontinuities.

The Walén relation for an ideal rotational discontinuity is

    v_i - V_HT = ± V_A,i

where V_HT is the de Hoffmann-Teller velocity: the constant velocity that
minimises the motional electric field ``|(v_i - V_HT) × B_i|`` over the interval.
This recipe solves that least-squares problem directly following Sonnerup et al.
(1987) and Khrabrov & Sonnerup (1998), then regresses the HT-frame plasma
velocity against the Alfvén velocity component by component and as one stacked
global fit.

The older mean-subtracted comparison is still available with ``frame="mean"``
for reproducing legacy results, but it is not a de Hoffmann-Teller frame.
"""

import numpy as np

MU0 = 4 * np.pi * 1e-7       # H/m
MP = 1.6726e-27               # kg (proton mass)
CM3_TO_M3 = 1e6               # 1 cm⁻³ = 1e6 m⁻³

# Paschmann & Sonnerup (2008), ISSI SR-008 ch. 3: |slope| ≳ 0.7-0.8
# with high correlation is commonly taken as RD evidence. These are deliberately
# named constants so event specialists can tune the convention.
RD_SLOPE_THRESHOLD = 0.7
RD_R2_THRESHOLD = 0.8
PARTIAL_SLOPE_THRESHOLD = 0.4


def alfven_velocity(B_nT, n_cm3):
    """Alfvén velocity vector in km/s.

    Parameters
    ----------
    B_nT  : (N, 3) magnetic field vectors in nT.
    n_cm3 : (N,) proton number density in cm^-3.

    Returns
    -------
    (N, 3) array of V_A = B / sqrt(mu0 * n * m_p), in km/s, component by
    component — the Walén test compares it with the velocity vector, so the
    magnitude alone would not do.
    """
    B_T = B_nT * 1e-9                          # nT → T
    n_m3 = n_cm3 * CM3_TO_M3
    rho = n_m3 * MP                            # kg/m³
    VA = B_T / np.sqrt(MU0 * rho[:, None])    # m/s vector
    return VA * 1e-3                           # km/s


def dehoffmann_teller_velocity(V, B):
    """Least-squares de Hoffmann-Teller velocity in km/s.

    Sonnerup et al. write the normal equations as
    ``K0 · V_HT = <K_i v_i>``, with ``K_i = B_i² I - B_i B_iᵀ``. A singular
    ``K0`` means the sampled magnetic-field directions do not constrain all
    three velocity components, so the recipe refuses the HT-frame Walén slope
    instead of reporting a NaN or an arbitrary pseudo-inverse solution.
    """
    identity = np.eye(3)
    b2 = np.einsum("ij,ij->i", B, B)
    K = b2[:, None, None] * identity - B[:, :, None] * B[:, None, :]
    K0 = K.mean(axis=0)
    rhs = np.einsum("nij,nj->ni", K, V).mean(axis=0)

    if np.linalg.matrix_rank(K0) < 3:
        return None, "HT frame is underdetermined: magnetic-field directions make K0 singular"
    if not np.isfinite(np.linalg.cond(K0)) or np.linalg.cond(K0) > 1e12:
        return None, "HT frame is underdetermined: magnetic-field directions make K0 ill-conditioned"

    try:
        return np.linalg.solve(K0, rhs), None
    except np.linalg.LinAlgError:
        return None, "HT frame is underdetermined: magnetic-field directions make K0 singular"


def _regression(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    if x.size < 2:
        return None, None, None

    x0 = x - x.mean()
    y0 = y - y.mean()
    ss_x = float(np.dot(x0, x0))
    if ss_x <= 0:
        return None, None, None

    slope = float(np.dot(x0, y0) / ss_x)
    intercept = float(y.mean() - slope * x.mean())
    residual = y - (slope * x + intercept)
    ss_res = float(np.dot(residual, residual))
    ss_tot = float(np.dot(y0, y0))
    R2 = 1.0 if ss_tot <= 0 and ss_res <= 0 else 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return slope, intercept, float(R2)


def _rounded(value):
    return None if value is None else round(float(value), 4)


def walen_test(V, B, n_cm3, frame="ht"):
    """Run the Walén test in the requested velocity frame.

    Parameters
    ----------
    V : (N, 3) array
        Ion bulk velocity in km/s.
    B : (N, 3) array
        Magnetic field in nT.
    n_cm3 : (N,) array
        Ion number density in cm⁻³.
    frame : {"ht", "mean"}
        ``"ht"`` solves the de Hoffmann-Teller frame and compares
        ``V - V_HT`` with ``V_A``. ``"mean"`` preserves the previous
        mean-subtracted regression, ``V - <V>`` against ``V_A - <V_A>``.

    Returns
    -------
    dict
        Global stacked slope and R², per-component slopes and R², the frame
        used, V_HT when solved, and the physical interpretation. Degenerate HT
        inputs return an ``error`` key and no slope.
    """
    V = np.asarray(V, dtype=float)
    B = np.asarray(B, dtype=float)
    n = np.asarray(n_cm3, dtype=float)
    frame = str(frame).lower()

    if frame not in {"ht", "mean"}:
        return {"error": 'frame must be "ht" or "mean"', "frame": frame}
    if V.ndim != 2 or B.ndim != 2 or V.shape[1] != 3 or B.shape[1] != 3 or V.shape != B.shape:
        return {"error": "V and B must be finite arrays with shape (N, 3)", "frame": frame}
    if n.shape != (V.shape[0],):
        return {"error": "n_cm3 must be a finite array with shape (N,)", "frame": frame}
    if not (np.isfinite(V).all() and np.isfinite(B).all() and np.isfinite(n).all()):
        return {"error": "V, B and n_cm3 must contain only finite values", "frame": frame}
    if np.any(n <= 0):
        return {"error": "n_cm3 must be positive to compute the Alfvén velocity", "frame": frame}

    VA = alfven_velocity(B, n)

    if frame == "ht":
        V_HT, error = dehoffmann_teller_velocity(V, B)
        if error:
            return {"error": error, "frame": frame}
        dV = V - V_HT
        dVA = VA
    else:
        V_HT = None
        dV = V - V.mean(axis=0)
        dVA = VA - VA.mean(axis=0)

    slope, intercept, R2 = _regression(dVA.ravel(), dV.ravel())
    if slope is None:
        return {"error": "Walén regression is underdetermined: Alfvén velocity has no variance", "frame": frame}

    component_slopes = {}
    component_R2 = {}
    component_intercepts = {}
    for component, name in enumerate(("x", "y", "z")):
        c_slope, c_intercept, c_R2 = _regression(dVA[:, component], dV[:, component])
        component_slopes[name] = _rounded(c_slope)
        component_R2[name] = _rounded(c_R2)
        component_intercepts[name] = _rounded(c_intercept)

    if abs(slope) >= RD_SLOPE_THRESHOLD and R2 >= RD_R2_THRESHOLD:
        interpretation = "consistent with a rotational discontinuity (Walén relation satisfied)"
    elif PARTIAL_SLOPE_THRESHOLD <= abs(slope) < RD_SLOPE_THRESHOLD:
        interpretation = "partial Alfvénic correlation"
    else:
        interpretation = "not Alfvénic"

    result = {
        "slope": round(float(slope), 4),
        "R2": round(float(R2), 4),
        "intercept_km_s": round(float(intercept), 4),
        "component_slopes": component_slopes,
        "component_R2": component_R2,
        "component_intercepts_km_s": component_intercepts,
        "interpretation": interpretation,
        "frame": frame,
        "VA_mean_km_s": VA.mean(axis=0).tolist(),
    }
    if V_HT is not None:
        result["V_HT"] = V_HT.tolist()
    return result


# ── Run ────────────────────────────────────────────────────────────────────────
# Define V, B and n_cm3 before this script — from aligned, common-cadence data:
#   var_B = load_data("b_gse")       # magnetic field, nT, shape (N, 3)
#   var_V = load_data("ion_v_gse")   # ion bulk velocity, km/s, shape (N, 3)
#   var_n = load_data("ion_n")       # ion density, cm^-3, shape (N,)
# Then bind V, B and n_cm3 to those arrays and run this recipe. The optional
# global `frame = "mean"` reproduces the legacy mean-subtracted comparison.

V, B, n_cm3 = globals().get("V"), globals().get("B"), globals().get("n_cm3")
frame = globals().get("frame", "ht")

if V is not None and B is not None and n_cm3 is not None:
    result = walen_test(V, B, n_cm3, frame=frame)
    if "error" in result:
        print("walen_test:", result["error"])
    else:
        export("walen_slope", np.array([result["slope"]]), "")
        export("walen_R2", np.array([result["R2"]]), "")
        if "V_HT" in result:
            export("V_HT", np.asarray(result["V_HT"]), "km/s")
        print(result)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N = 200
    t = np.linspace(0, np.pi, N)
    B_demo = np.column_stack([5 + 3 * np.cos(t), -2 + np.sin(t), 1 + 0.5 * np.sin(2 * t)])
    n_demo = 5 + rng.normal(0, 0.3, N)
    V_HT_demo = np.array([-400.0, 30.0, -10.0])
    V_demo = V_HT_demo + alfven_velocity(B_demo, n_demo) + rng.normal(0, 5, (N, 3))
    print(walen_test(V_demo, B_demo, n_demo))
