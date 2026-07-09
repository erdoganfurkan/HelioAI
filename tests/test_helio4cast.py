"""Tests for the helio4cast/ remote catalog prefix (ICMECAT)."""

from __future__ import annotations

import os
import time

import pytest

from helioai.config import settings
from helioai.tools import catalog_tools as ct

_CSV = """\
,icmecat_id,sc_insitu,icme_start_time,mo_start_time,mo_end_time,mo_bmax
0,ICME_Wind_20230101_01,Wind,2023-01-01T04:30Z,2023-01-01T10:00Z,2023-01-02T06:00Z,18.4
1,ICME_Wind_20230215_01,Wind,2023-02-15T12:00Z,2023-02-15T18:30Z,2023-02-16T09:15Z,22.1
2,ICME_PSP_20230301_01,PSP,2023-03-01T00:10Z,2023-03-01T05:00Z,2023-03-01T20:45Z,45.0
3,ICME_STEREO_A_20230410_01,STEREO-A,2023-04-10T08:00Z,,2023-04-11T02:00Z,12.7
4,ICME_BAD_20230501_01,Wind,,2023-05-01T02:00Z,,9.9
"""


@pytest.fixture()
def seeded_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    path = tmp_path / "helio4cast" / "icmecat.csv"
    path.parent.mkdir(parents=True)
    path.write_text(_CSV, encoding="utf-8")
    return path


async def test_resolve_without_network(seeded_cache):
    cat = ct._load_helio4cast_catalog("icmecat")
    assert cat is not None
    events = list(cat)
    assert len(events) == 4
    assert events[0].start_time.isoformat().startswith("2023-01-01T04:30")


async def test_get_catalog_preview_and_where(seeded_cache):
    out = await ct.get_catalog("helio4cast/icmecat", columns=["sc_insitu", "mo_bmax"])
    assert out["nb_events_total"] == 4
    out_wind = await ct.get_catalog(
        "helio4cast/icmecat",
        where={"column": "sc_insitu", "op": "eq", "value": "Wind"},
    )
    assert out_wind["nb_events_filtered"] == 2


async def test_row_without_times_skipped(seeded_cache):
    cat = ct._load_helio4cast_catalog("icmecat")
    ids = [ev.meta.get("icmecat_id") for ev in cat]
    assert "ICME_BAD_20230501_01" not in ids
    assert "ICME_STEREO_A_20230410_01" in ids


async def test_meta_columns_exposed(seeded_cache):
    cat = ct._load_helio4cast_catalog("icmecat")
    ev = list(cat)[0]
    assert ev.meta["sc_insitu"] == "Wind"
    assert ev.meta["mo_bmax"] == "18.4"
    assert "" not in ev.meta


async def test_stale_cache_dead_url_fallback(seeded_cache, monkeypatch):
    old = time.time() - ct._HELIO4CAST_TTL_S - 10
    os.utime(seeded_cache, (old, old))
    monkeypatch.setitem(
        ct._HELIO4CAST["icmecat"], "url", "http://127.0.0.1:1/HELIO4CAST_ICMECAT_v23.csv"
    )
    cat = ct._load_helio4cast_catalog("icmecat")
    assert cat is not None
    assert len(list(cat)) == 4


async def test_no_cache_dead_url_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setitem(
        ct._HELIO4CAST["icmecat"], "url", "http://127.0.0.1:1/HELIO4CAST_ICMECAT_v23.csv"
    )
    out = await ct.get_catalog("helio4cast/icmecat")
    assert "error" in out


async def test_list_catalogs_includes_helio4cast(seeded_cache):
    out = await ct.list_catalogs(region="ICME")
    ids = [e["id"] for e in out["catalogs"]]
    assert "helio4cast/icmecat" in ids
    entry = next(e for e in out["catalogs"] if e["id"] == "helio4cast/icmecat")
    assert entry["nb_events"] == 5


def test_h4c_iso_normalization():
    assert ct._h4c_iso("2023-01-01T04:30Z") == "2023-01-01T04:30:00"
    assert ct._h4c_iso("2023-01-01 04:30:15Z") == "2023-01-01T04:30:15"
    assert ct._h4c_iso("") == ""


def test_export_citation_collected():
    from helioai.core.llm.base import Message, ToolCall
    from helioai.export import _collect_catalog_refs

    history = [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="1", name="get_catalog", arguments={"catalog_id": "helio4cast/icmecat"})
            ],
        ),
    ]
    refs = _collect_catalog_refs(history)
    assert len(refs) == 1
    assert "Moestl" in refs[0]
