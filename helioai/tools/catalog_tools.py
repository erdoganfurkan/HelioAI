"""Catalog and timetable tools for HelioAI.

Exposes the 29 CatalogIndex + 188 TimetableIndex from the AMDA speasy
inventory as first-class agent tools.  The key capability is
get_events_timeseries: download a parameter for every event in a catalog
in one speasy call, opening the door to superposed epoch analysis.
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)

_catalog_cache: dict = {"ts": 0.0, "entries": []}

# ── helpers ───────────────────────────────────────────────────────────────────


def _get_spz():
    try:
        import speasy as spz

        return spz
    except ImportError:
        return None


def _uid(index) -> str:
    return str(getattr(index, "__spz_uid__", "") or getattr(index, "uid", "") or "")


def _name(index) -> str:
    return str(getattr(index, "__spz_name__", "") or getattr(index, "name", "") or "")


def _desc(index) -> str:
    return str(getattr(index, "desc", "") or getattr(index, "description", "") or "")


def _nb(index) -> int:
    try:
        return int(getattr(index, "nbIntervals", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _survey(index) -> tuple[str, str]:
    start = str(getattr(index, "surveyStart", "") or "")[:10]
    stop = str(getattr(index, "surveyStop", "") or "")[:10]
    return start, stop


def _spz_type(index) -> str:
    t = str(getattr(index, "__spz_type__", "") or "")
    if t in ("CatalogIndex",):
        return "catalog"
    if t in ("TimetableIndex",):
        return "timetable"
    return "unknown"


def _walk_catalogs(spz) -> list[dict]:
    """Return a flat list of all AMDA catalog + timetable entries (TTL-cached 1h)."""
    if _catalog_cache["entries"] and time.monotonic() - _catalog_cache["ts"] < 3600:
        return _catalog_cache["entries"]

    entries: list[dict] = []
    try:
        flat = spz.inventories.flat_inventories.amda
        for uid, idx in (getattr(flat, "catalogs", None) or {}).items():
            start, stop = _survey(idx)
            entries.append(
                {
                    "id": f"amda/{uid}",
                    "name": _name(idx) or uid,
                    "type": "catalog",
                    "nb_events": _nb(idx),
                    "survey_start": start,
                    "survey_stop": stop,
                    "description": _desc(idx)[:200],
                }
            )
        for uid, idx in (getattr(flat, "timetables", None) or {}).items():
            start, stop = _survey(idx)
            entries.append(
                {
                    "id": f"amda/{uid}",
                    "name": _name(idx) or uid,
                    "type": "timetable",
                    "nb_events": _nb(idx),
                    "survey_start": start,
                    "survey_stop": stop,
                    "description": _desc(idx)[:200],
                }
            )
    except Exception as e:
        log.warning("catalog walk failed: %s", e)

    _catalog_cache["entries"] = entries
    _catalog_cache["ts"] = time.monotonic()
    return entries


# ── tools ─────────────────────────────────────────────────────────────────────


async def list_catalogs(
    type: str = "all",
    region: str | None = None,
) -> dict:
    """List available AMDA event catalogs and timetables.

    Args:
        type: 'catalog', 'timetable', or 'all' (default).
        region: optional keyword filter on name/description (e.g. 'ICME', 'bow shock', 'MMS').

    Returns a list of entries with id, name, type, nb_events, survey range and description.
    Use the `id` field with get_catalog() and get_events_timeseries().
    """
    spz = _get_spz()
    if spz is None:
        return {"error": "speasy is not installed"}

    entries = _walk_catalogs(spz)

    if type in ("catalog", "timetable"):
        entries = [e for e in entries if e["type"] == type]

    if region:
        kw = region.lower()
        entries = [e for e in entries if kw in e["name"].lower() or kw in e["description"].lower()]

    entries.sort(key=lambda e: e["nb_events"], reverse=True)

    return {
        "total": len(entries),
        "type_filter": type,
        "region_filter": region,
        "catalogs": entries,
    }


async def get_catalog(
    catalog_id: str,
    start: str | None = None,
    stop: str | None = None,
    max_events: int = 10,
) -> dict:
    """Download and summarize an AMDA event catalog or timetable.

    Args:
        catalog_id: speasy uid from list_catalogs (e.g. 'amda/sharedcatalog_41').
        start: optional ISO 8601 start — filter events that begin after this time.
        stop:  optional ISO 8601 stop  — filter events that begin before this time.
        max_events: maximum events to include in the sample (default 50).

    Returns catalog metadata + a sample of events (start, stop, key columns).
    Use with get_events_timeseries to download a parameter over all events.
    """
    spz = _get_spz()
    if spz is None:
        return {"error": "speasy is not installed"}

    uid = catalog_id.removeprefix("amda/")

    try:
        flat = spz.inventories.flat_inventories.amda
        cats = getattr(flat, "catalogs", {}) or {}
        tts = getattr(flat, "timetables", {}) or {}
        index = cats.get(uid) or tts.get(uid)
        if index is None:
            return {"error": f"Catalog {catalog_id!r} not found in AMDA inventory"}

        cat = spz.get_data(index)
    except Exception as e:
        return {"error": f"Failed to download catalog {catalog_id!r}: {e}"}

    if cat is None:
        return {"error": f"No data returned for {catalog_id!r}"}

    try:
        events = list(cat)
    except Exception as e:
        return {"error": f"Failed to iterate catalog events: {e}"}

    # Filter by time window (client-side)
    if start or stop:
        filtered = []
        for ev in events:
            ev_start = str(getattr(ev, "start_time", "") or getattr(ev, "start", "") or "")[:19]
            if start and ev_start < start:
                continue
            if stop and ev_start > stop:
                continue
            filtered.append(ev)
        events = filtered

    nb_total = len(events)
    sample = events[:max_events]

    # Build event rows
    rows: list[dict] = []
    for ev in sample:
        ev_start = str(getattr(ev, "start_time", "") or getattr(ev, "start", "") or "")[:19]
        ev_stop = str(getattr(ev, "stop_time", "") or getattr(ev, "stop", "") or "")[:19]
        row: dict[str, Any] = {"start": ev_start, "stop": ev_stop}
        meta = getattr(ev, "meta", None)
        if meta and isinstance(meta, dict):
            for k, v in list(meta.items())[:8]:
                row[k] = v
        rows.append(row)

    columns = list(rows[0].keys()) if rows else ["start", "stop"]

    survey_start, survey_stop = _survey(index)
    return {
        "_kind": "catalog_preview",
        "catalog_id": catalog_id,
        "name": _name(index),
        "type": _spz_type(index),
        "nb_events_total": nb_total,
        "nb_events_filtered": nb_total,
        "columns": columns,
        "sample": rows,
        "survey_start": survey_start,
        "survey_stop": survey_stop,
        "note": (
            f"Showing {len(sample)} of {nb_total} events. "
            "Use get_events_timeseries() to download a parameter over all events."
        ),
    }


async def get_events_timeseries(
    catalog_id: str,
    param_id: str,
    start: str,
    stop: str,
    max_events: int = 20,
) -> dict:
    """Download a parameter for every event in a catalog window (superposed epoch).

    This is the core catalog tool: it fetches N time series in a SINGLE speasy call
    using the native multi-interval API.  Use it for:
    - Superposed epoch analysis (stack-plot across events)
    - Statistical summaries per event (min/max/mean)
    - Comparing a parameter across e.g. all ICME crossings in a year

    Args:
        catalog_id: speasy uid from list_catalogs (e.g. 'amda/sharedcatalog_41').
        param_id:   speasy parameter id (e.g. 'amda/imf_gsm') — resolve via search_parameters first.
        start:      ISO 8601 start — restrict to events beginning after this time.
        stop:       ISO 8601 stop  — restrict to events beginning before this time.
        max_events: cap on events to download (default 20 — each is one speasy call slot).

    Returns per-event statistics and saves the raw data to the workspace for run_python.
    """
    spz = _get_spz()
    if spz is None:
        return {"error": "speasy is not installed"}

    # --- resolve catalog index ---
    uid = catalog_id.removeprefix("amda/")
    try:
        flat = spz.inventories.flat_inventories.amda
        cats = getattr(flat, "catalogs", {}) or {}
        tts = getattr(flat, "timetables", {}) or {}
        index = cats.get(uid) or tts.get(uid)
        if index is None:
            return {"error": f"Catalog {catalog_id!r} not found"}
        cat = spz.get_data(index)
    except Exception as e:
        return {"error": f"Failed to download catalog {catalog_id!r}: {e}"}

    if cat is None:
        return {"error": f"No data for catalog {catalog_id!r}"}

    # --- filter events ---
    try:
        events = list(cat)
    except Exception as e:
        return {"error": f"Cannot iterate catalog: {e}"}

    filtered = []
    for ev in events:
        ev_start = str(getattr(ev, "start_time", "") or getattr(ev, "start", "") or "")[:19]
        if ev_start < start or ev_start > stop:
            continue
        filtered.append(ev)

    if not filtered:
        return {
            "warning": f"No events found in [{start}, {stop}] for {catalog_id!r}.",
            "suggestion": "Widen the time window or use get_catalog() to inspect the survey range.",
        }

    selected = filtered[:max_events]

    # --- batch download: ONE speasy call for all events ---
    try:
        timeseries_list = spz.get_data(param_id, selected)
    except Exception as e:
        return {"error": f"speasy.get_data({param_id!r}, events) failed: {e}"}

    if timeseries_list is None:
        return {"error": f"No data returned for {param_id!r} over {len(selected)} events"}

    if not isinstance(timeseries_list, list):
        timeseries_list = [timeseries_list]

    # --- per-event statistics ---
    import numpy as np

    stats: list[dict] = []
    for i, (ev, ts) in enumerate(zip(selected, timeseries_list)):
        ev_start = str(getattr(ev, "start_time", "") or getattr(ev, "start", "") or "")[:19]
        ev_stop = str(getattr(ev, "stop_time", "") or getattr(ev, "stop", "") or "")[:19]
        if ts is None or len(ts.time) == 0:
            stats.append({"event": i, "start": ev_start, "stop": ev_stop, "status": "no_data"})
            continue
        vals = ts.values.astype(float)
        vals[~np.isfinite(vals)] = np.nan
        vals[np.abs(vals) >= 1e30] = np.nan
        with np.errstate(all="ignore"):
            stats.append(
                {
                    "event": i,
                    "start": ev_start,
                    "stop": ev_stop,
                    "n_points": int(len(ts.time)),
                    "mean": _fmt(np.nanmean(vals)),
                    "std": _fmt(np.nanstd(vals)),
                    "min": _fmt(np.nanmin(vals)),
                    "max": _fmt(np.nanmax(vals)),
                }
            )

    good = [s for s in stats if s.get("status") != "no_data"]
    units = str(getattr(timeseries_list[0], "unit", "") or "") if timeseries_list else ""

    return {
        "catalog_id": catalog_id,
        "param_id": param_id,
        "time_window": [start, stop],
        "n_events_found": len(filtered),
        "n_events_downloaded": len(selected),
        "n_events_with_data": len(good),
        "units": units,
        "per_event_stats": stats,
        "note": (
            "Use run_python with spz.get_data(param_id, events) for custom plots. "
            "The catalog events are speasy DateTimeRange objects iterable from the catalog."
        ),
    }


def _fmt(val) -> float | None:
    try:
        v = float(val)
        return round(v, 4) if abs(v) < 1e10 else None
    except Exception:
        return None
