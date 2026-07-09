"""Tests for the standalone sandbox physics helpers."""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from helioai.tools.sandbox_helpers import bs_jelinek2012, mp_shue1998, transform_coords


def test_shue_standoff_reference():
    theta, r = mp_shue1998(2.0, 0.0, theta_deg=0.0)
    expected = (10.22 + 1.29 * np.tanh(0.184 * 8.14)) * 2.0 ** (-1.0 / 6.6)
    assert abs(float(r[0]) - expected) < 1e-9
    assert 10.0 < float(r[0]) < 10.5


def test_shue_compression_with_pressure():
    _, r_low = mp_shue1998(1.0, 0.0, theta_deg=0.0)
    _, r_high = mp_shue1998(8.0, 0.0, theta_deg=0.0)
    assert float(r_high[0]) < float(r_low[0])


def test_shue_erosion_with_southward_bz():
    _, r_north = mp_shue1998(2.0, 5.0, theta_deg=0.0)
    _, r_south = mp_shue1998(2.0, -10.0, theta_deg=0.0)
    assert float(r_south[0]) < float(r_north[0])


def test_shue_default_profile_shape():
    theta, r = mp_shue1998(2.0, 0.0)
    assert theta.shape == r.shape
    assert np.all(np.diff(r) > 0)


def test_jelinek_standoff_and_terminator():
    _, r0 = bs_jelinek2012(2.0, theta_deg=0.0)
    expected = 15.02 * 2.0 ** (-1.0 / 6.55)
    assert abs(float(r0[0]) - expected) < 1e-9
    _, r90 = bs_jelinek2012(2.0, theta_deg=90.0)
    assert abs(float(r90[0]) - 2.0 * expected / 1.17) < 1e-9


def test_jelinek_outside_shue():
    theta, r_bs = bs_jelinek2012(2.0, theta_deg=[0.0, 45.0, 90.0])
    _, r_mp = mp_shue1998(2.0, 0.0, theta_deg=[0.0, 45.0, 90.0])
    assert np.all(r_bs > r_mp)


def test_jelinek_no_solution_antisunward():
    _, r = bs_jelinek2012(2.0, theta_deg=180.0)
    assert np.isnan(r[0])


def test_transform_roundtrip():
    v = np.array([10.0, 2.0, 3.0])
    gsm = transform_coords("2019-01-01T00:00:00", v, "gse", "gsm")
    back = transform_coords("2019-01-01T00:00:00", gsm, "gsm", "gse")
    assert np.allclose(back, v, atol=1e-6)


def test_transform_matches_raw_geopack():
    from geopack import geopack as gp

    t = np.datetime64("2019-01-01T00:00:00").astype("datetime64[s]").astype("int64")
    gp.recalc(float(t))
    expected = gp.gsmgse(10.0, 2.0, 3.0, -1)
    out = transform_coords("2019-01-01T00:00:00", [10.0, 2.0, 3.0], "gse", "gsm")
    assert np.allclose(out, expected, atol=1e-9)


def test_transform_vector_batch_shapes():
    vecs = np.arange(12.0).reshape(4, 3)
    times = np.array(["2019-01-01T00:00:00"] * 4, dtype="datetime64[s]")
    out = transform_coords(times, vecs, "gei", "mag")
    assert out.shape == (4, 3)


def test_transform_unknown_frame_raises():
    try:
        transform_coords("2019-01-01", [1.0, 2.0, 3.0], "gse", "banana")
    except ValueError as e:
        assert "banana" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_import_without_helioai_config():
    code = (
        "import sys\n"
        "from helioai.tools.sandbox_helpers import mp_shue1998\n"
        "assert 'helioai.config' not in sys.modules, 'sandbox_helpers pulled helioai.config'\n"
        "theta, r = mp_shue1998(2.0, 0.0, theta_deg=0.0)\n"
        "print(round(float(r[0]), 2))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "10.25" in proc.stdout
