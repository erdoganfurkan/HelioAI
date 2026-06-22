"""Tests for helioai.tools.catalog_tools — mocked speasy."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

import helioai.tools.catalog_tools as ct_module
from helioai.tools.catalog_tools import (
    _walk_catalogs,
    list_catalogs,
    get_catalog,
    get_events_timeseries,
)


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_catalog_cache():
    ct_module._catalog_cache["ts"] = 0.0
    ct_module._catalog_cache["entries"] = []
    yield


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


def _make_event(start, stop, meta=None):
    ev = MagicMock()
    ev.start_time = start
    ev.stop_time = stop
    ev.meta = meta if meta is not None else {"shock_type": "FF"}
    return ev


def _make_flat(catalogs: dict, timetables: dict):
    flat = MagicMock()
    flat.catalogs = catalogs
    flat.timetables = timetables
    return flat


def _make_spz(catalogs=None, timetables=None):
    mock_spz = MagicMock()
    mock_spz.inventories.flat_inventories.amda = _make_flat(catalogs or {}, timetables or {})
    return mock_spz


# ── _walk_catalogs TTL cache ──────────────────────────────────────────────────


def test_walk_catalogs_ttl_cache() -> None:
    # Reset the module-level cache so this test starts fresh.
    ct_module._catalog_cache["ts"] = 0.0
    ct_module._catalog_cache["entries"] = []

    access_log: list[int] = []

    class FakeAmda:
        catalogs = {"c1": _make_catalog_index("c1", "ICME", "", 10, "2005-01-01", "2022-01-01")}
        timetables: dict = {}

    class FakeFlat:
        @property
        def amda(self):
            access_log.append(1)
            return FakeAmda()

    class FakeInv:
        flat_inventories = FakeFlat()

    class FakeSpz:
        inventories = FakeInv()

    fake_spz = FakeSpz()

    _walk_catalogs(fake_spz)  # first call — walks the inventory
    _walk_catalogs(fake_spz)  # second call — TTL hit, returns cache

    assert len(access_log) == 1, "inventory accessed more than once within TTL"

    # Backdate the timestamp past the TTL to force a re-walk
    import time

    ct_module._catalog_cache["ts"] = time.monotonic() - 3601
    _walk_catalogs(fake_spz)
    assert len(access_log) == 2, "expired cache should trigger a re-walk"


# ── list_catalogs ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_catalogs_returns_all(monkeypatch, tmp_path) -> None:
    from helioai.config import settings

    monkeypatch.setattr(settings.catalogs, "catalogs_dir", str(tmp_path))
    cat_idx = _make_catalog_index(
        "sharedcatalog_41", "ICME list", "Richardson & Cane ICME", 341, "1996-01-01", "2022-12-31"
    )
    tt_idx = _make_catalog_index(
        "sharedtimetable_1",
        "Substorm onset",
        "Frey substorms",
        2437,
        "2000-01-01",
        "2010-12-31",
        "TimetableIndex",
    )
    mock_spz = _make_spz({"sharedcatalog_41": cat_idx}, {"sharedtimetable_1": tt_idx})
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await list_catalogs()
    assert result["total"] == 2
    ids = [e["id"] for e in result["catalogs"]]
    assert "amda/sharedcatalog_41" in ids
    assert "amda/sharedtimetable_1" in ids


@pytest.mark.asyncio
async def test_list_catalogs_type_filter(monkeypatch, tmp_path) -> None:
    from helioai.config import settings

    monkeypatch.setattr(settings.catalogs, "catalogs_dir", str(tmp_path))
    cat_idx = _make_catalog_index("c1", "Cat", "", 10, "", "")
    tt_idx = _make_catalog_index("t1", "TT", "", 5, "", "", "TimetableIndex")
    mock_spz = _make_spz({"c1": cat_idx}, {"t1": tt_idx})
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await list_catalogs(type="catalog")
    assert all(e["type"] == "catalog" for e in result["catalogs"])
    assert result["total"] == 1


@pytest.mark.asyncio
async def test_list_catalogs_region_filter(monkeypatch) -> None:
    icme_idx = _make_catalog_index("c1", "ICME list", "Richardson ICMEs", 100, "", "")
    shock_idx = _make_catalog_index("c2", "Bow shock crossings", "MMS bow shock", 2797, "", "")
    mock_spz = _make_spz({"c1": icme_idx, "c2": shock_idx}, {})
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await list_catalogs(region="ICME")
    assert result["total"] == 1
    assert result["catalogs"][0]["id"] == "amda/c1"


# ── get_catalog ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_catalog_returns_events(monkeypatch) -> None:
    cat_idx = _make_catalog_index(
        "sharedcatalog_41", "ICME list", "", 3, "1996-01-01", "2022-12-31"
    )
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
    cat_idx = _make_catalog_index(
        "sharedcatalog_41", "ICME list", "", 3, "1996-01-01", "2022-12-31"
    )
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
    events = [
        _make_event(f"2005-{i:02d}-01T00:00:00", f"2005-{i:02d}-02T00:00:00") for i in range(1, 21)
    ]
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

    cat_idx = _make_catalog_index(
        "sharedcatalog_41", "ICME list", "", 2, "2005-01-01", "2005-12-31"
    )
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
        "amda/sharedcatalog_41",
        "amda/imf_gsm",
        "2005-01-01T00:00:00",
        "2005-12-31T23:59:59",
    )
    assert "error" not in result
    assert result["n_events_downloaded"] == 2
    assert len(result["per_event_stats"]) == 2
    stat = result["per_event_stats"][0]
    assert "components" in stat or "mean" in stat


@pytest.mark.asyncio
async def test_get_events_timeseries_no_events_in_window(monkeypatch) -> None:
    cat_idx = _make_catalog_index(
        "sharedcatalog_41", "ICME list", "", 1, "2005-01-01", "2005-12-31"
    )
    events = [_make_event("2005-01-17T00:00:00", "2005-01-18T00:00:00")]
    mock_cat = MagicMock()
    mock_cat.__iter__ = MagicMock(return_value=iter(events))

    mock_spz = _make_spz({"sharedcatalog_41": cat_idx}, {})
    mock_spz.get_data.return_value = mock_cat
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await get_events_timeseries(
        "amda/sharedcatalog_41",
        "amda/imf_gsm",
        "2010-01-01T00:00:00",
        "2010-12-31T23:59:59",
    )
    assert "warning" in result
    assert "suggestion" in result


# ── QW2: agent-side filters ───────────────────────────────────────────────────


def _setup_catalog_mock(monkeypatch, events, catalog_uid="c1"):
    cat_idx = _make_catalog_index(
        catalog_uid, "Test catalog", "", len(events), "2000-01-01", "2030-01-01"
    )
    mock_cat = MagicMock()
    mock_cat.__iter__ = MagicMock(return_value=iter(events))
    mock_spz = _make_spz({catalog_uid: cat_idx}, {})
    mock_spz.get_data.return_value = mock_cat
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)
    return f"amda/{catalog_uid}"


@pytest.mark.asyncio
async def test_get_catalog_where_filter_gt(monkeypatch) -> None:
    events = [
        _make_event("2005-01-01T00:00:00", "2005-01-02T00:00:00", meta={"speed": 400}),
        _make_event("2005-02-01T00:00:00", "2005-02-02T00:00:00", meta={"speed": 700}),
        _make_event("2005-03-01T00:00:00", "2005-03-02T00:00:00", meta={"speed": 550}),
    ]
    cid = _setup_catalog_mock(monkeypatch, events)
    result = await get_catalog(cid, where={"column": "speed", "op": "gt", "value": 500})
    assert "error" not in result
    assert result["nb_events_filtered"] == 2
    speeds = [row["speed"] for row in result["sample"]]
    assert all(s > 500 for s in speeds)


@pytest.mark.asyncio
async def test_get_catalog_where_filter_contains(monkeypatch) -> None:
    events = [
        _make_event("2005-01-01T00:00:00", "2005-01-02T00:00:00", meta={"type": "FF shock"}),
        _make_event("2005-02-01T00:00:00", "2005-02-02T00:00:00", meta={"type": "RR shock"}),
        _make_event("2005-03-01T00:00:00", "2005-03-02T00:00:00", meta={"type": "FF shock"}),
    ]
    cid = _setup_catalog_mock(monkeypatch, events)
    result = await get_catalog(cid, where={"column": "type", "op": "contains", "value": "FF"})
    assert result["nb_events_filtered"] == 2
    assert all("FF" in row["type"] for row in result["sample"])


@pytest.mark.asyncio
async def test_get_catalog_sort_and_pagination(monkeypatch) -> None:
    events = [
        _make_event("2005-01-01T00:00:00", "2005-01-02T00:00:00", meta={"speed": 400}),
        _make_event("2005-02-01T00:00:00", "2005-02-02T00:00:00", meta={"speed": 700}),
        _make_event("2005-03-01T00:00:00", "2005-03-02T00:00:00", meta={"speed": 550}),
        _make_event("2005-04-01T00:00:00", "2005-04-02T00:00:00", meta={"speed": 300}),
    ]
    cat_idx = _make_catalog_index("c1", "Test", "", 4, "2000-01-01", "2030-01-01")
    mock_cat = MagicMock()
    mock_cat.__iter__ = MagicMock(side_effect=lambda: iter(events))
    mock_spz = _make_spz({"c1": cat_idx}, {})
    mock_spz.get_data.return_value = mock_cat
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)
    cid = "amda/c1"
    page1 = await get_catalog(cid, sort_by="speed", descending=True, max_events=2, offset=0)
    page2 = await get_catalog(cid, sort_by="speed", descending=True, max_events=2, offset=2)
    speeds1 = [row["speed"] for row in page1["sample"]]
    speeds2 = [row["speed"] for row in page2["sample"]]
    assert speeds1 == [700, 550]
    assert speeds2 == [400, 300]
    assert set(speeds1).isdisjoint(set(speeds2))


@pytest.mark.asyncio
async def test_get_catalog_columns_projection(monkeypatch) -> None:
    events = [
        _make_event(
            "2005-01-01T00:00:00",
            "2005-01-02T00:00:00",
            meta={"speed": 500, "type": "FF", "mach": 3.5},
        ),
    ]
    cid = _setup_catalog_mock(monkeypatch, events)
    result = await get_catalog(cid, columns=["speed"])
    row = result["sample"][0]
    assert "speed" in row
    assert "type" not in row
    assert "mach" not in row


@pytest.mark.asyncio
async def test_get_catalog_nb_events_filtered_differs_from_total(monkeypatch) -> None:
    events = [
        _make_event("2005-01-01T00:00:00", "2005-01-02T00:00:00", meta={"speed": 400}),
        _make_event("2005-02-01T00:00:00", "2005-02-02T00:00:00", meta={"speed": 700}),
    ]
    cid = _setup_catalog_mock(monkeypatch, events)
    result = await get_catalog(cid, where={"column": "speed", "op": "gt", "value": 600})
    assert result["nb_events_total"] == 2
    assert result["nb_events_filtered"] == 1


@pytest.mark.asyncio
async def test_get_catalog_offset_beyond_end(monkeypatch) -> None:
    events = [_make_event("2005-01-01T00:00:00", "2005-01-02T00:00:00")]
    cid = _setup_catalog_mock(monkeypatch, events)
    result = await get_catalog(cid, offset=100)
    assert "error" not in result
    assert result["sample"] == []
    assert result["returned"] == 0


# ── QW3: per-component stats ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_events_timeseries_vector_stats(monkeypatch) -> None:
    import numpy as np

    cat_idx = _make_catalog_index("c1", "ICME list", "", 1, "2005-01-01", "2005-12-31")
    events = [_make_event("2005-01-17T00:00:00", "2005-01-18T00:00:00")]
    mock_cat = MagicMock()
    mock_cat.__iter__ = MagicMock(return_value=iter(events))

    fake_ts = MagicMock()
    fake_ts.time = np.array(["2005-01-17T01:00:00", "2005-01-17T02:00:00"], dtype="datetime64[s]")
    fake_ts.values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    fake_ts.columns = ["bx", "by", "bz"]
    fake_ts.unit = "nT"

    def get_data_side_effect(arg, intervals=None):
        if intervals is not None:
            return [fake_ts]
        return mock_cat

    mock_spz = _make_spz({"c1": cat_idx}, {})
    mock_spz.get_data.side_effect = get_data_side_effect
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await get_events_timeseries(
        "amda/c1", "amda/imf_gsm", "2005-01-01T00:00:00", "2005-12-31T23:59:59"
    )
    assert "error" not in result
    stat = result["per_event_stats"][0]
    assert "components" in stat
    assert "bx" in stat["components"]
    assert "bz" in stat["components"]
    assert stat["components"]["bx"]["mean"] == pytest.approx(2.5)
    assert stat["components"]["bz"]["mean"] == pytest.approx(4.5)
    assert "magnitude" in stat
    assert "mean" not in stat


@pytest.mark.asyncio
async def test_get_events_timeseries_cap_warning_present(monkeypatch) -> None:
    """cap_warning is set when n_events_found > max_events."""
    import numpy as np

    cat_idx = _make_catalog_index("c1", "ICME list", "", 5, "2005-01-01", "2005-12-31")
    events = [
        _make_event(f"2005-0{i}-01T00:00:00", f"2005-0{i}-02T00:00:00") for i in range(1, 6)
    ]
    mock_cat = MagicMock()
    mock_cat.__iter__ = MagicMock(return_value=iter(events))

    fake_ts = MagicMock()
    fake_ts.time = np.array(["2005-01-01T01:00:00"], dtype="datetime64[s]")
    fake_ts.values = np.array([5.0])
    fake_ts.unit = "nT"

    def get_data_side_effect(arg, intervals=None):
        if intervals is not None:
            return [fake_ts] * len(intervals)
        return mock_cat

    mock_spz = _make_spz({"c1": cat_idx}, {})
    mock_spz.get_data.side_effect = get_data_side_effect
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await get_events_timeseries(
        "amda/c1", "amda/imf_gsm", "2005-01-01T00:00:00", "2005-12-31T23:59:59",
        max_events=3,
    )
    assert "error" not in result
    assert result["n_events_found"] == 5
    assert result["n_events_downloaded"] == 3
    assert "cap_warning" in result
    assert "3/5" in result["cap_warning"]


@pytest.mark.asyncio
async def test_get_events_timeseries_no_cap_warning_when_not_truncated(monkeypatch) -> None:
    """cap_warning is absent when all events are downloaded."""
    import numpy as np

    cat_idx = _make_catalog_index("c1", "ICME list", "", 2, "2005-01-01", "2005-12-31")
    events = [
        _make_event("2005-01-01T00:00:00", "2005-01-02T00:00:00"),
        _make_event("2005-02-01T00:00:00", "2005-02-02T00:00:00"),
    ]
    mock_cat = MagicMock()
    mock_cat.__iter__ = MagicMock(return_value=iter(events))

    fake_ts = MagicMock()
    fake_ts.time = np.array(["2005-01-01T01:00:00"], dtype="datetime64[s]")
    fake_ts.values = np.array([5.0])
    fake_ts.unit = "nT"

    def get_data_side_effect(arg, intervals=None):
        if intervals is not None:
            return [fake_ts] * len(intervals)
        return mock_cat

    mock_spz = _make_spz({"c1": cat_idx}, {})
    mock_spz.get_data.side_effect = get_data_side_effect
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await get_events_timeseries(
        "amda/c1", "amda/imf_gsm", "2005-01-01T00:00:00", "2005-12-31T23:59:59",
    )
    assert "error" not in result
    assert "cap_warning" not in result


@pytest.mark.asyncio
async def test_get_events_timeseries_per_event_stats_slimmed_above_10(monkeypatch) -> None:
    """per_event_stats is trimmed to 10 representative events when > 10 are downloaded."""
    import numpy as np

    n = 12
    cat_idx = _make_catalog_index("c1", "ICME list", "", n, "2005-01-01", "2005-12-31")
    events = [
        _make_event(f"2005-{i:02d}-01T00:00:00", f"2005-{i:02d}-02T00:00:00") for i in range(1, n + 1)
    ]
    mock_cat = MagicMock()
    mock_cat.__iter__ = MagicMock(return_value=iter(events))

    fake_ts = MagicMock()
    fake_ts.time = np.array(["2005-01-01T01:00:00"], dtype="datetime64[s]")
    fake_ts.values = np.array([5.0])
    fake_ts.unit = "nT"

    def get_data_side_effect(arg, intervals=None):
        if intervals is not None:
            return [fake_ts] * len(intervals)
        return mock_cat

    mock_spz = _make_spz({"c1": cat_idx}, {})
    mock_spz.get_data.side_effect = get_data_side_effect
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await get_events_timeseries(
        "amda/c1", "amda/imf_gsm", "2005-01-01T00:00:00", "2005-12-31T23:59:59",
        max_events=n,
    )
    assert "error" not in result
    assert result["n_events_downloaded"] == n
    assert len(result["per_event_stats"]) == 10
    assert "stats_note" in result
    assert str(n) in result["stats_note"]


@pytest.mark.asyncio
async def test_get_events_timeseries_scalar_stats_unchanged(monkeypatch) -> None:
    import numpy as np

    cat_idx = _make_catalog_index("c1", "Test", "", 1, "2005-01-01", "2005-12-31")
    events = [_make_event("2005-01-17T00:00:00", "2005-01-18T00:00:00")]
    mock_cat = MagicMock()
    mock_cat.__iter__ = MagicMock(return_value=iter(events))

    fake_ts = MagicMock()
    fake_ts.time = np.array(["2005-01-17T01:00:00"], dtype="datetime64[s]")
    fake_ts.values = np.array([5.0])
    fake_ts.columns = []
    fake_ts.unit = "km/s"

    def get_data_side_effect(arg, intervals=None):
        if intervals is not None:
            return [fake_ts]
        return mock_cat

    mock_spz = _make_spz({"c1": cat_idx}, {})
    mock_spz.get_data.side_effect = get_data_side_effect
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await get_events_timeseries(
        "amda/c1", "amda/vsw", "2005-01-01T00:00:00", "2005-12-31T23:59:59"
    )
    stat = result["per_event_stats"][0]
    assert "mean" in stat
    assert "components" not in stat
