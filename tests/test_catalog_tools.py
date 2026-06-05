"""Tests for helioai.tools.catalog_tools — mocked speasy."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from helioai.tools.catalog_tools import list_catalogs, get_catalog, get_events_timeseries


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_catalog_index(uid, name, desc, nb, s_start, s_stop, spz_type="CatalogIndex"):
    idx = MagicMock()
    idx.__spz_uid__ = uid
    idx.__spz_name__ = name
    idx.__spz_type__ = spz_type
    idx.desc = desc
    idx.nbIntervals = nb
    idx.surveyStart = s_start
    idx.surveyStop = s_stop
    return idx


def _make_event(start, stop):
    ev = MagicMock()
    ev.start_time = start
    ev.stop_time = stop
    ev.meta = {"shock_type": "FF"}
    return ev


def _make_flat(catalogs: dict, timetables: dict):
    flat = MagicMock()
    flat.catalogs = catalogs
    flat.timetables = timetables
    return flat


def _make_spz(catalogs=None, timetables=None):
    mock_spz = MagicMock()
    mock_spz.inventories.flat_inventories.amda = _make_flat(
        catalogs or {}, timetables or {}
    )
    return mock_spz


# ── list_catalogs ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_catalogs_returns_all(monkeypatch) -> None:
    cat_idx = _make_catalog_index("sharedcatalog_41", "ICME list", "Richardson & Cane ICME", 341, "1996-01-01", "2022-12-31")
    tt_idx  = _make_catalog_index("sharedtimetable_1", "Substorm onset", "Frey substorms", 2437, "2000-01-01", "2010-12-31", "TimetableIndex")
    mock_spz = _make_spz({"sharedcatalog_41": cat_idx}, {"sharedtimetable_1": tt_idx})
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await list_catalogs()
    assert result["total"] == 2
    ids = [e["id"] for e in result["catalogs"]]
    assert "amda/sharedcatalog_41" in ids
    assert "amda/sharedtimetable_1" in ids


@pytest.mark.asyncio
async def test_list_catalogs_type_filter(monkeypatch) -> None:
    cat_idx = _make_catalog_index("c1", "Cat", "", 10, "", "")
    tt_idx  = _make_catalog_index("t1", "TT", "", 5, "", "", "TimetableIndex")
    mock_spz = _make_spz({"c1": cat_idx}, {"t1": tt_idx})
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await list_catalogs(type="catalog")
    assert all(e["type"] == "catalog" for e in result["catalogs"])
    assert result["total"] == 1


@pytest.mark.asyncio
async def test_list_catalogs_region_filter(monkeypatch) -> None:
    icme_idx  = _make_catalog_index("c1", "ICME list", "Richardson ICMEs", 100, "", "")
    shock_idx = _make_catalog_index("c2", "Bow shock crossings", "MMS bow shock", 2797, "", "")
    mock_spz = _make_spz({"c1": icme_idx, "c2": shock_idx}, {})
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await list_catalogs(region="ICME")
    assert result["total"] == 1
    assert result["catalogs"][0]["id"] == "amda/c1"


# ── get_catalog ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_catalog_returns_events(monkeypatch) -> None:
    cat_idx = _make_catalog_index("sharedcatalog_41", "ICME list", "", 3, "1996-01-01", "2022-12-31")
    events = [
        _make_event("2005-01-17T00:00:00", "2005-01-18T00:00:00"),
        _make_event("2005-05-15T00:00:00", "2005-05-16T00:00:00"),
        _make_event("2006-12-14T00:00:00", "2006-12-15T00:00:00"),
    ]
    mock_cat = MagicMock()
    mock_cat.__iter__ = MagicMock(return_value=iter(events))

    mock_spz = _make_spz({"sharedcatalog_41": cat_idx}, {})
    mock_spz.get_data.return_value = mock_cat
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await get_catalog("amda/sharedcatalog_41")
    assert "error" not in result
    assert result["nb_events_total"] == 3
    assert len(result["sample"]) == 3
    assert result["sample"][0]["start"] == "2005-01-17T00:00:00"


@pytest.mark.asyncio
async def test_get_catalog_time_filter(monkeypatch) -> None:
    cat_idx = _make_catalog_index("sharedcatalog_41", "ICME list", "", 3, "1996-01-01", "2022-12-31")
    events = [
        _make_event("2005-01-17T00:00:00", "2005-01-18T00:00:00"),
        _make_event("2006-05-15T00:00:00", "2006-05-16T00:00:00"),
    ]
    mock_cat = MagicMock()
    mock_cat.__iter__ = MagicMock(return_value=iter(events))

    mock_spz = _make_spz({"sharedcatalog_41": cat_idx}, {})
    mock_spz.get_data.return_value = mock_cat
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await get_catalog("amda/sharedcatalog_41", start="2005-01-01", stop="2005-12-31")
    assert result["nb_events_filtered"] == 1
    assert result["sample"][0]["start"] == "2005-01-17T00:00:00"


@pytest.mark.asyncio
async def test_get_catalog_not_found(monkeypatch) -> None:
    mock_spz = _make_spz({}, {})
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await get_catalog("amda/nonexistent")
    assert "error" in result


@pytest.mark.asyncio
async def test_get_catalog_kind_marker(monkeypatch) -> None:
    cat_idx = _make_catalog_index("c1", "ICME list", "", 2, "2005-01-01", "2022-12-31")
    events = [_make_event("2005-01-17T00:00:00", "2005-01-18T00:00:00")]
    mock_cat = MagicMock()
    mock_cat.__iter__ = MagicMock(return_value=iter(events))
    mock_spz = _make_spz({"c1": cat_idx}, {})
    mock_spz.get_data.return_value = mock_cat
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await get_catalog("amda/c1")
    assert result.get("_kind") == "catalog_preview"
    assert "survey_start" in result
    assert "survey_stop" in result


@pytest.mark.asyncio
async def test_get_catalog_max_events_default(monkeypatch) -> None:
    cat_idx = _make_catalog_index("c1", "Big catalog", "", 50, "2005-01-01", "2022-12-31")
    events = [_make_event(f"2005-{i:02d}-01T00:00:00", f"2005-{i:02d}-02T00:00:00") for i in range(1, 21)]
    mock_cat = MagicMock()
    mock_cat.__iter__ = MagicMock(return_value=iter(events))
    mock_spz = _make_spz({"c1": cat_idx}, {})
    mock_spz.get_data.return_value = mock_cat
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await get_catalog("amda/c1")
    assert len(result["sample"]) <= 10


# ── get_events_timeseries ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_events_timeseries_returns_stats(monkeypatch) -> None:
    import numpy as np

    cat_idx = _make_catalog_index("sharedcatalog_41", "ICME list", "", 2, "2005-01-01", "2005-12-31")
    events = [
        _make_event("2005-01-17T00:00:00", "2005-01-18T00:00:00"),
        _make_event("2005-05-15T00:00:00", "2005-05-16T00:00:00"),
    ]
    mock_cat = MagicMock()
    mock_cat.__iter__ = MagicMock(return_value=iter(events))

    fake_ts = MagicMock()
    fake_ts.time = np.array(["2005-01-17T01:00:00"], dtype="datetime64[s]")
    fake_ts.values = np.array([[5.0, -3.0, 2.0]])
    fake_ts.unit = "nT"

    def get_data_side_effect(arg, intervals=None):
        if intervals is not None:
            return [fake_ts, fake_ts]
        return mock_cat

    mock_spz = _make_spz({"sharedcatalog_41": cat_idx}, {})
    mock_spz.get_data.side_effect = get_data_side_effect
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await get_events_timeseries(
        "amda/sharedcatalog_41", "amda/imf_gsm",
        "2005-01-01T00:00:00", "2005-12-31T23:59:59",
    )
    assert "error" not in result
    assert result["n_events_downloaded"] == 2
    assert len(result["per_event_stats"]) == 2
    assert "mean" in result["per_event_stats"][0]


@pytest.mark.asyncio
async def test_get_events_timeseries_no_events_in_window(monkeypatch) -> None:
    cat_idx = _make_catalog_index("sharedcatalog_41", "ICME list", "", 1, "2005-01-01", "2005-12-31")
    events = [_make_event("2005-01-17T00:00:00", "2005-01-18T00:00:00")]
    mock_cat = MagicMock()
    mock_cat.__iter__ = MagicMock(return_value=iter(events))

    mock_spz = _make_spz({"sharedcatalog_41": cat_idx}, {})
    mock_spz.get_data.return_value = mock_cat
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await get_events_timeseries(
        "amda/sharedcatalog_41", "amda/imf_gsm",
        "2010-01-01T00:00:00", "2010-12-31T23:59:59",
    )
    assert "warning" in result
    assert "suggestion" in result
