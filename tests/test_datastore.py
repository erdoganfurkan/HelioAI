"""Tests for helioai.datastore — session data persistence."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import helioai.workspace as ws_module
from helioai.datastore import (
    DATA_SUBDIR,
    read_manifest,
    save_event_collection,
    save_timeseries,
)


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def session_dir(tmp_path, monkeypatch):
    """Patch data root + set user/label so get_session_dir() resolves under tmp_path."""
    from helioai.config import settings
    from helioai.workspace import DEFAULT_USER

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    utok = ws_module.set_user(DEFAULT_USER)
    token = ws_module.set_label("test_session")
    yield tmp_path / "users" / DEFAULT_USER / "workspace" / "test_session"
    ws_module.reset_label(token)
    ws_module.reset_user(utok)


# ── save_timeseries ────────────────────────────────────────────────────────────


def test_save_timeseries_creates_npz_and_manifest(session_dir):
    t = np.array(["2005-01-17T00:00:00", "2005-01-17T01:00:00"], dtype="datetime64[s]")
    v = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    result = save_timeseries(
        "amda/imf_gsm",
        time=t,
        values=v,
        param_id="amda/imf_gsm",
        units="nT",
        start="2005-01-17T00:00:00",
        stop="2005-01-18T00:00:00",
        columns=["bx", "by", "bz"],
        source="get_timeseries",
    )

    assert result is not None
    ds_name = result["dataset"]
    data_dir = session_dir / DATA_SUBDIR
    assert (data_dir / f"{ds_name}.npz").exists()
    manifest = read_manifest(session_dir)
    entry = manifest["datasets"][ds_name]
    assert entry["kind"] == "timeseries"
    assert entry["param_id"] == "amda/imf_gsm"
    assert entry["units"] == "nT"
    assert entry["columns"] == ["bx", "by", "bz"]
    z = np.load(data_dir / f"{ds_name}.npz", allow_pickle=False)
    np.testing.assert_array_equal(z["time"], t)
    np.testing.assert_array_equal(z["values"], v)


def test_save_timeseries_idempotent_same_window(session_dir):
    t = np.array(["2005-01-17T00:00:00"], dtype="datetime64[s]")
    v = np.array([[1.0]])

    r1 = save_timeseries(
        "amda/imf_gsm",
        time=t,
        values=v,
        param_id="amda/imf_gsm",
        units="nT",
        start="2005-01-17T00:00:00",
        stop="2005-01-18T00:00:00",
        columns=[],
        source="get_timeseries",
    )
    r2 = save_timeseries(
        "amda/imf_gsm",
        time=t,
        values=v,
        param_id="amda/imf_gsm",
        units="nT",
        start="2005-01-17T00:00:00",
        stop="2005-01-18T00:00:00",
        columns=[],
        source="get_timeseries",
    )

    assert r1["dataset"] == r2["dataset"]
    manifest = read_manifest(session_dir)
    assert len(manifest["datasets"]) == 1


def test_save_timeseries_collision_different_window(session_dir):
    t = np.array(["2005-01-17T00:00:00"], dtype="datetime64[s]")
    v = np.array([[1.0]])

    r1 = save_timeseries(
        "amda/imf_gsm",
        time=t,
        values=v,
        param_id="amda/imf_gsm",
        units="nT",
        start="2005-01-17T00:00:00",
        stop="2005-01-18T00:00:00",
        columns=[],
        source="get_timeseries",
    )
    r2 = save_timeseries(
        "amda/imf_gsm",
        time=t,
        values=v,
        param_id="amda/imf_gsm",
        units="nT",
        start="2006-01-01T00:00:00",
        stop="2006-01-02T00:00:00",
        columns=[],
        source="get_timeseries",
    )

    assert r1["dataset"] != r2["dataset"]
    manifest = read_manifest(session_dir)
    assert len(manifest["datasets"]) == 2


def test_save_timeseries_exotic_param_id(session_dir):
    t = np.array(["2005-01-17T00:00:00"], dtype="datetime64[s]")
    v = np.array([1.0])

    result = save_timeseries(
        "cda/AC_H0_MFI/BGSEc",
        time=t,
        values=v,
        param_id="cda/AC_H0_MFI/BGSEc",
        units="nT",
        start="2005-01-17T00:00:00",
        stop="2005-01-18T00:00:00",
        columns=[],
        source="get_timeseries",
    )
    assert result is not None
    assert result["dataset"].startswith("bgsec")


def test_save_timeseries_mock_columns_ignored(session_dir):
    from unittest.mock import MagicMock

    t = np.array(["2005-01-17T00:00:00"], dtype="datetime64[s]")
    v = np.array([1.0])
    mock_cols = MagicMock()

    result = save_timeseries(
        "amda/vsw",
        time=t,
        values=v,
        param_id="amda/vsw",
        units="km/s",
        start="2005-01-17T00:00:00",
        stop="2005-01-18T00:00:00",
        columns=mock_cols,
        source="get_timeseries",
    )
    assert result is not None
    manifest = read_manifest(session_dir)
    entry = list(manifest["datasets"].values())[0]
    assert entry["columns"] == []


# ── save_event_collection ──────────────────────────────────────────────────────


def _make_ts(t_strs, values):
    ts = SimpleNamespace()
    ts.time = np.array(t_strs, dtype="datetime64[s]")
    ts.values = np.array(values)
    return ts


def test_save_event_collection_creates_npz(session_dir):
    ts1 = _make_ts(["2005-01-17T00:00:00"], [[1.0, 2.0, 3.0]])
    ts2 = _make_ts(["2005-05-15T00:00:00"], [[4.0, 5.0, 6.0]])
    series = [
        ("2005-01-17T00:00:00", "2005-01-18T00:00:00", ts1),
        ("2005-05-15T00:00:00", "2005-05-16T00:00:00", ts2),
    ]
    result = save_event_collection(
        "amda/imf_gsm",
        series=series,
        param_id="amda/imf_gsm",
        units="nT",
        source="get_events_timeseries",
    )
    assert result is not None
    ds_name = result["dataset"]
    data_dir = session_dir / DATA_SUBDIR
    assert (data_dir / f"{ds_name}.npz").exists()
    manifest = read_manifest(session_dir)
    entry = manifest["datasets"][ds_name]
    assert entry["kind"] == "event_collection"
    assert entry["n_events"] == 2
    z = np.load(data_dir / f"{ds_name}.npz", allow_pickle=False)
    assert "t0" in z and "v0" in z
    assert "t1" in z and "v1" in z


def test_save_event_collection_no_data_event_absent(session_dir):
    ts1 = _make_ts(["2005-01-17T00:00:00"], [[1.0, 2.0, 3.0]])
    series = [
        ("2005-01-17T00:00:00", "2005-01-18T00:00:00", ts1),
        ("2005-05-15T00:00:00", "2005-05-16T00:00:00", None),
    ]
    result = save_event_collection(
        "amda/imf_gsm",
        series=series,
        param_id="amda/imf_gsm",
        units="nT",
        source="get_events_timeseries",
    )
    assert result is not None
    data_dir = session_dir / DATA_SUBDIR
    ds_name = result["dataset"]
    z = np.load(data_dir / f"{ds_name}.npz", allow_pickle=False)
    assert "t0" in z
    assert "t1" not in z


# ── read_manifest ──────────────────────────────────────────────────────────────


def test_read_manifest_empty_dir(tmp_path):
    manifest = read_manifest(tmp_path)
    assert manifest == {"datasets": {}}


def test_read_manifest_roundtrip(session_dir):
    t = np.array(["2005-01-17T00:00:00"], dtype="datetime64[s]")
    v = np.array([1.0])
    save_timeseries(
        "amda/vsw",
        time=t,
        values=v,
        param_id="amda/vsw",
        units="km/s",
        start="2005-01-17T00:00:00",
        stop="2005-01-18T00:00:00",
        columns=[],
        source="get_timeseries",
    )
    manifest = read_manifest(session_dir)
    assert "vsw" in manifest["datasets"] or any(
        e["param_id"] == "amda/vsw" for e in manifest["datasets"].values()
    )


# ── sandbox load_data integration ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sandbox_load_data_roundtrip(session_dir):
    from helioai.tools.sandbox import run_python

    t = np.array(["2005-01-17T00:00:00", "2005-01-17T01:00:00"], dtype="datetime64[s]")
    v = np.array([10.0, 20.0])
    save_timeseries(
        "amda/vsw",
        time=t,
        values=v,
        param_id="amda/vsw",
        units="km/s",
        start="2005-01-17T00:00:00",
        stop="2005-01-18T00:00:00",
        columns=[],
        source="get_timeseries",
    )

    code = "data = load_data('vsw'); export('mean_v', data.values)"
    result = await run_python(code, _plot_dir=str(session_dir))
    assert "error" not in result or "Code exited" not in result.get("error", "")
    assert "mean_v" in result.get("exports", {})
    assert abs(result["exports"]["mean_v"]["mean"] - 15.0) < 0.01


@pytest.mark.asyncio
async def test_sandbox_load_data_unknown_name(session_dir):
    from helioai.tools.sandbox import run_python

    t = np.array(["2005-01-17T00:00:00"], dtype="datetime64[s]")
    v = np.array([1.0])
    save_timeseries(
        "amda/vsw",
        time=t,
        values=v,
        param_id="amda/vsw",
        units="km/s",
        start="2005-01-17T00:00:00",
        stop="2005-01-18T00:00:00",
        columns=[],
        source="get_timeseries",
    )

    code = "data = load_data('nonexistent')"
    result = await run_python(code, _plot_dir=str(session_dir))
    assert result.get("returncode", result.get("error", "")) is not None
    assert (
        "nonexistent" in result.get("stderr", "")
        or "nonexistent" in result.get("stdout", "")
        or result.get("error")
    )


@pytest.mark.asyncio
async def test_sandbox_load_data_rejects_traversal(session_dir):
    from helioai.tools.sandbox import run_python

    session_dir.mkdir(parents=True, exist_ok=True)
    code = "load_data('../etc/passwd')"
    result = await run_python(code, _plot_dir=str(session_dir))
    assert (
        result.get("error")
        or result.get("returncode") != 0
        or "invalid" in result.get("stderr", "").lower()
    )
