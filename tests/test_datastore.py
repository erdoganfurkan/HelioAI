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


def test_save_event_collection_cap_warning_logged_once(session_dir, monkeypatch, caplog):
    """Real bug: the cap warning fired once per event PAST the cap, spamming the logs."""
    from helioai import datastore

    monkeypatch.setattr(datastore, "_MAX_BYTES", 1)  # first event already exceeds it
    series = [
        (
            f"2005-0{i}-01T00:00:00",
            f"2005-0{i}-02T00:00:00",
            _make_ts(["2005-01-01T00:00:00"], [[1.0]]),
        )
        for i in range(1, 6)
    ]
    with caplog.at_level("WARNING"):
        result = datastore.save_event_collection(
            "amda/imf_gsm", series=series, param_id="amda/imf_gsm", units="nT", source="test"
        )
    assert result is None  # every event truncated, nothing to persist
    assert sum("exceeds 100 MB cap" in r.message for r in caplog.records) == 1


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
async def test_sandbox_load_data_accepts_the_product_id(session_dir):
    """The id every tool result and prompt shows must work, not just the slug.

    HelioBench n3 spent 41 of its 51 tool errors here: the prompt said "the dataset
    saved from cda/WI_H0_MFI/BGSM", the model typed exactly that, and load_data
    rejected it for not being lowercase. The `_data` suffix case matters too — ACE
    density slugs to `np`, which the preamble already binds.
    """
    from helioai.tools.sandbox import run_python

    t = np.array(["2005-01-17T00:00:00", "2005-01-17T01:00:00"], dtype="datetime64[s]")
    for pid, vals in (("cda/WI_H0_MFI/BGSM", [10.0, 20.0]), ("cda/AC_H0_SWE/Np", [3.0, 5.0])):
        save_timeseries(
            pid,
            time=t,
            values=np.array(vals),
            param_id=pid,
            units="nT",
            start="2005-01-17T00:00:00",
            stop="2005-01-18T00:00:00",
            columns=[],
            source="get_timeseries",
        )

    code = (
        "export('b', load_data('cda/WI_H0_MFI/BGSM').values)\n"
        "export('n', load_data('cda/AC_H0_SWE/Np').values)"
    )
    result = await run_python(code, _plot_dir=str(session_dir))

    assert abs(result["exports"]["b"]["mean"] - 15.0) < 0.01
    assert abs(result["exports"]["n"]["mean"] - 4.0) < 0.01, "the _data suffix case"


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


def test_slug_avoids_sandbox_names():
    """ACE density is `.../Np`, which would slug to `np` and rebind numpy.

    load_data itself calls np.load, so a rebound `np` kills every later
    load_data in the same cell — one plasma variable poisons the rest.
    """
    from helioai.datastore import _slug

    assert _slug("cda/AC_H0_SWE/Np") == "np_data"
    assert _slug("cda/AC_H0_MFI/BGSM") == "bgsm"
    assert _slug("amda/imf_gsm") == "imf_gsm"
    for reserved in ("np", "plt", "os", "json", "u"):
        assert _slug(f"cda/X/{reserved}") != reserved


# ── fill values ────────────────────────────────────────────────────────────────


def test_fill_mask_catches_all_three_conventions():
    import numpy as np

    from helioai.datastore import fill_mask

    values = np.array([400.0, np.nan, np.inf, -1e31, 99999.8984375, 450.0])

    mask = fill_mask(values, np.float32(99999.9))

    assert mask.tolist() == [False, True, True, True, True, False]


def test_fill_mask_keeps_real_values_near_the_sentinel():
    """OMNI carries a real proton temperature of 99093 K — not a fill value."""
    import numpy as np

    from helioai.datastore import fill_mask

    values = np.array([77695.0, 99093.0, 91000.0])

    assert not fill_mask(values, np.float32(99999.9)).any()


def test_blank_fill_does_not_mutate_the_caller():
    import numpy as np

    from helioai.datastore import blank_fill

    original = np.array([1.0, -1e31, 3.0])
    cleaned, mask = blank_fill(original)

    assert np.isnan(cleaned[1])
    assert original[1] == -1e31, "the caller's array must be left alone"
    assert mask.tolist() == [False, True, False]


def test_blank_fill_passes_non_numeric_through():
    from helioai.datastore import blank_fill

    values = np.array(["a", "b"], dtype=object)
    out, mask = blank_fill(values)

    assert out is values
    assert mask is None


def test_saved_timeseries_records_missing_pct(tmp_path, monkeypatch):
    """The manifest figure is derived from the file, so it cannot drift from it."""
    import json

    import numpy as np

    import helioai.datastore as ds

    monkeypatch.setattr(ds, "_session_data_dir", lambda: tmp_path)
    values = np.array([1.0, np.nan, 3.0, np.nan])

    saved = ds.save_timeseries(
        "cda/X/Np",
        time=np.arange("2015-03-17T00:00", 4, dtype="datetime64[m]"),
        values=values,
        param_id="cda/X/Np",
        units="cm^-3",
        start="2015-03-17T00:00:00",
        stop="2015-03-17T00:04:00",
        columns=["Np"],
        source="test",
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["datasets"][saved["dataset"]]["missing_pct"] == 50.0


def test_find_existing_matches_param_and_window(tmp_path, monkeypatch):
    """The lead re-fetched the same Wind field three times with the name in its history.

    The prompt rule was there and ignored, so the tool answers from the manifest instead
    of asking the model to remember.
    """
    import helioai.workspace as ws
    from helioai.datastore import find_existing, save_timeseries

    monkeypatch.setattr(ws, "_root", lambda: tmp_path)
    tok = ws.set_label("sess")
    try:
        save_timeseries(
            "b3gsm",
            time=np.array(["2015-03-16T18:00:00"], dtype="datetime64[s]"),
            values=np.array([[1.0, 2.0, 3.0]]),
            param_id="cda/WI_H0_MFI/B3GSM",
            units="nT",
            start="2015-03-16T18:00:00",
            stop="2015-03-18T12:00:00",
            columns=["Bx", "By", "Bz"],
            source="get_timeseries",
        )
        same = find_existing("cda/WI_H0_MFI/B3GSM", "2015-03-16T18:00:00", "2015-03-18T12:00:00")
        assert same == "b3gsm"

        # A different window is a different dataset — the shock-zoom download must proceed.
        assert (
            find_existing("cda/WI_H0_MFI/B3GSM", "2015-03-17T03:30:00", "2015-03-17T05:00:00")
            is None
        )
        assert (
            find_existing("cda/AC_H0_MFI/BGSM", "2015-03-16T18:00:00", "2015-03-18T12:00:00")
            is None
        )
    finally:
        ws.reset_label(tok)


def test_find_existing_is_silent_without_a_session(monkeypatch):
    import helioai.workspace as ws
    from helioai.datastore import find_existing

    monkeypatch.setattr(ws, "get_session_dir", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert find_existing("cda/X/Y", "a", "b") is None
