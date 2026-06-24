"""Tests for save_catalog + local/ prefix integration."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from helioai.config import settings


@pytest.fixture()
def catalogs_dir(tmp_path, monkeypatch):
    from helioai.workspace import DEFAULT_USER, reset_user, set_user

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    tok = set_user(DEFAULT_USER)
    d = tmp_path / "users" / DEFAULT_USER / "catalogs"
    yield d
    reset_user(tok)


def _make_events(n=3):
    return [
        {
            "start": f"2005-01-{i:02d}T00:00:00",
            "stop": f"2005-01-{i:02d}T06:00:00",
            "speed": i * 100,
        }
        for i in range(1, n + 1)
    ]


# ── save_catalog ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_catalog_creates_json(catalogs_dir):
    from helioai.tools.catalog_tools import save_catalog

    result = await save_catalog("test_shocks", _make_events(3), description="IP shocks 2005")
    assert result.get("catalog_id") == "local/test_shocks"
    assert result["nb_events"] == 3
    assert result.get("overwritten") is False

    path = catalogs_dir / "test_shocks.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["name"] == "test_shocks"
    assert data["description"] == "IP shocks 2005"
    assert len(data["events"]) == 3
    assert data["events"][0]["start"] == "2005-01-01T00:00:00"
    assert data["events"][0]["meta"]["speed"] == 100


@pytest.mark.asyncio
async def test_save_catalog_overwrite(catalogs_dir):
    from helioai.tools.catalog_tools import save_catalog

    await save_catalog("shocks", _make_events(2))
    result = await save_catalog("shocks", _make_events(5))
    assert result.get("overwritten") is True
    assert result["nb_events"] == 5


@pytest.mark.asyncio
async def test_save_catalog_invalid_name(catalogs_dir):
    from helioai.tools.catalog_tools import save_catalog

    r = await save_catalog("../traversal", _make_events(1))
    assert "error" in r

    r = await save_catalog("", _make_events(1))
    assert "error" in r

    r = await save_catalog("UPPERCASE", _make_events(1))
    assert "error" in r


@pytest.mark.asyncio
async def test_save_catalog_empty_events(catalogs_dir):
    from helioai.tools.catalog_tools import save_catalog

    r = await save_catalog("empty", [])
    assert "error" in r


@pytest.mark.asyncio
async def test_save_catalog_start_ge_stop(catalogs_dir):
    from helioai.tools.catalog_tools import save_catalog

    r = await save_catalog("bad", [{"start": "2005-01-02T00:00:00", "stop": "2005-01-01T00:00:00"}])
    assert "error" in r


# ── list_catalogs with local/ ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_catalogs_includes_local(catalogs_dir):
    from helioai.tools.catalog_tools import save_catalog

    await save_catalog("my_shocks", _make_events(4), description="my catalog")

    spz_mock = MagicMock()
    spz_mock.inventories.flat_inventories.amda.catalogs = {}
    spz_mock.inventories.flat_inventories.amda.timetables = {}

    with patch("helioai.tools.catalog_tools._get_spz", return_value=spz_mock):
        with patch("helioai.tools.catalog_tools._walk_catalogs", return_value=[]):
            from helioai.tools.catalog_tools import list_catalogs

            result = await list_catalogs()

    assert "catalogs" in result
    ids = [e["id"] for e in result["catalogs"]]
    assert "local/my_shocks" in ids
    entry = next(e for e in result["catalogs"] if e["id"] == "local/my_shocks")
    assert entry["nb_events"] == 4


# ── get_catalog with local/ ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_catalog_local(catalogs_dir):
    from helioai.tools.catalog_tools import get_catalog, save_catalog

    await save_catalog("icme_list", _make_events(3))

    spz_mock = MagicMock()

    with patch("helioai.tools.catalog_tools._get_spz", return_value=spz_mock):
        result = await get_catalog("local/icme_list")

    assert result.get("catalog_id") == "local/icme_list"
    assert result["nb_events_total"] == 3
    assert len(result["sample"]) == 3


@pytest.mark.asyncio
async def test_get_catalog_local_not_found(catalogs_dir):
    from helioai.tools.catalog_tools import get_catalog

    spz_mock = MagicMock()
    with patch("helioai.tools.catalog_tools._get_spz", return_value=spz_mock):
        result = await get_catalog("local/nonexistent")

    assert "error" in result


# ── round-trip: save → get_catalog → get_events_timeseries ────────────────────


@pytest.mark.asyncio
async def test_round_trip_save_then_get_events(catalogs_dir, tmp_path, monkeypatch):
    """save_catalog → get_events_timeseries uses the local catalog correctly."""
    import numpy as np

    import helioai.workspace as ws_module
    from helioai.tools.catalog_tools import get_events_timeseries, save_catalog

    monkeypatch.setattr(settings.workspace, "workspace_dir", tmp_path)
    token = ws_module.set_label("test_f2")
    try:
        events = _make_events(2)
        await save_catalog("my_events", events)

        # Mock speasy get_data to return a fake timeseries per event
        fake_ts = MagicMock()
        fake_ts.time = np.array(["2005-01-01T00:00:00"], dtype="datetime64[s]")
        fake_ts.values = np.array([[1.0, 2.0, 3.0]])
        fake_ts.columns = ["Bx", "By", "Bz"]
        fake_ts.unit = "nT"

        spz_mock = MagicMock()
        spz_mock.get_data.return_value = [fake_ts, fake_ts]

        with patch("helioai.tools.catalog_tools._get_spz", return_value=spz_mock):
            result = await get_events_timeseries(
                "local/my_events",
                "amda/imf_gsm",
                "2005-01-01T00:00:00",
                "2005-01-03T00:00:00",
            )

        assert result.get("n_events_downloaded") == 2
        assert "per_event_stats" in result
        # Vector field → components dict
        stat = result["per_event_stats"][0]
        assert "components" in stat or "mean" in stat
    finally:
        ws_module.reset_label(token)
