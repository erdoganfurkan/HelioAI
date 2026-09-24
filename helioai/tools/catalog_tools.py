"""Catalog and timetable tools for HelioAI.

Exposes the 29 CatalogIndex + 188 TimetableIndex from the AMDA speasy
inventory as first-class agent tools.  The key capability is
get_events_timeseries: download a parameter for every event in a catalog
in one speasy call, opening the door to superposed epoch analysis.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from helioai.tools.offload import run_blocking, speasy_gate

log = logging.getLogger(__name__)


def _ev_iso(ev, attr_time: str, attr_plain: str) -> str:
    raw = str(getattr(ev, attr_time, "") or getattr(ev, attr_plain, "") or "")[:19]
    return raw.replace(" ", "T")


def _starts_within(ev, start: str | None, stop: str | None) -> bool:
    """Whether an event's *start* lies in `[start, stop]`; an open side is unbounded.

    Both catalog tools select events by where they begin, deliberately: a superposed
    epoch analysis aligns events on their onset, and "every ICME of 2015" means the
    ICMEs that arrived in 2015. An event that started before the window and ended
    inside it is therefore out, one that started inside and ran past the end is in.
    The comparison is on 19-character ISO strings, which sort chronologically.

    Args:
        ev: A speasy event (`start_time`) or a reconstructed one (`start`).
        start: Inclusive ISO lower bound, or None.
        stop: Inclusive ISO upper bound, or None.
    """
    ev_start = _ev_iso(ev, "start_time", "start")
    if start and ev_start < start:
        return False
    if stop and ev_start > stop:
        return False
    return True


def _event_value(ev, column: str):
    """Extract a column value from an event (start/stop are virtual columns)."""
    if column == "start":
        return _ev_iso(ev, "start_time", "start")
    if column == "stop":
        return _ev_iso(ev, "stop_time", "stop")
    meta = getattr(ev, "meta", None)
    if meta and isinstance(meta, dict):
        return meta.get(column)
    return None


WHERE_OPS: tuple[str, ...] = ("eq", "ne", "gt", "gte", "lt", "lte", "contains")
"""The operators `_match` can apply, and the enum `get_catalog`'s schema offers.

One tuple rather than two lists: the schema advertised what the model may send and the
dispatch decided what actually ran, with nothing holding them together."""


def _match(op: str, a, b) -> bool:
    """Apply comparison operator op between event value a and filter value b.

    Raises:
        ValueError: `op` is not one of `WHERE_OPS`. It used to fall through to False,
            which filtered every event out, so a typo read as an empty window.
    """
    if op not in WHERE_OPS:
        raise ValueError(f"unknown where operator {op!r}; use one of {', '.join(WHERE_OPS)}")
    try:
        a_f, b_f = float(a), float(b)
        a, b = a_f, b_f
    except (TypeError, ValueError):
        pass
    if op == "eq":
        return a == b
    if op == "ne":
        return a != b
    if op == "gt":
        return a is not None and a > b
    if op == "gte":
        return a is not None and a >= b
    if op == "lt":
        return a is not None and a < b
    if op == "lte":
        return a is not None and a <= b
    if op == "contains":
        return b.lower() in str(a).lower() if a is not None else False
    raise AssertionError(f"{op!r} is listed in WHERE_OPS but has no branch here")


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


# ── remote community catalogs (HELIO4CAST) ────────────────────────────────────

_HELIO4CAST = {
    "icmecat": {
        "url": "https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v23.csv",
        "start_col": "icme_start_time",
        "stop_col": "mo_end_time",
        "name": "HELIO4CAST ICMECAT v2.3",
        "description": (
            "Interplanetary CME catalog 1990-2025, multi-spacecraft in-situ (Wind, "
            "STEREO-A/B, PSP, Solar Orbiter, BepiColombo, MAVEN, Juno, Ulysses, "
            "MESSENGER, VEX). start=ICME start, stop=magnetic obstacle end; filter "
            "spacecraft with where={column: sc_insitu}. Cite Moestl et al."
        ),
        "citation": (
            "HELIO4CAST ICMECAT v2.3 (https://helioforecast.space/icmecat) — "
            "Moestl et al. (2017), Space Weather 15, doi:10.1002/2017SW001614"
        ),
        "approx_events": 1976,
        "survey": ("1990", "2025"),
    },
}
_HELIO4CAST_TTL_S = 7 * 86400


def _helio4cast_cache_path(name: str) -> Path:
    from helioai.config import settings

    d = settings.data_dir / "helio4cast"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.csv"


def _fetch_helio4cast_csv(name: str) -> Path | None:
    import urllib.request

    spec = _HELIO4CAST[name]
    path = _helio4cast_cache_path(name)
    if path.exists() and (time.time() - path.stat().st_mtime) < _HELIO4CAST_TTL_S:
        return path
    try:
        with urllib.request.urlopen(spec["url"], timeout=30) as resp:
            data = resp.read()
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    except Exception as e:
        if path.exists():
            log.warning("helio4cast %s: refresh failed (%s) — using stale cache", name, e)
            return path
        log.warning("helio4cast %s: download failed: %s", name, e)
        return None
    return path


def _h4c_iso(raw: str) -> str:
    t = (raw or "").strip().rstrip("Z").replace(" ", "T")
    return t + ":00" if len(t) == 16 else t


def _load_helio4cast_catalog(name: str):
    import csv

    from speasy.products import Catalog, Event

    spec = _HELIO4CAST.get(name)
    if spec is None:
        return None
    path = _fetch_helio4cast_csv(name)
    if path is None:
        return None
    time_cols = (spec["start_col"], spec["stop_col"])
    events, skipped = [], 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            start = _h4c_iso(row.get(spec["start_col"], ""))
            stop = _h4c_iso(row.get(spec["stop_col"], ""))
            if not start or not stop:
                skipped += 1
                continue
            meta = {k: v for k, v in row.items() if k and v and k not in time_cols}
            events.append(Event(start, stop, meta=meta))
    if skipped:
        log.info("helio4cast %s: skipped %d rows without start/stop", name, skipped)
    return Catalog(name=spec["name"], meta={}, events=events)


# ── tools ─────────────────────────────────────────────────────────────────────


async def list_catalogs(
    type: str = "all",
    region: str | None = None,
    query: str | None = None,
) -> dict:
    """List available AMDA event catalogs and timetables.

    Args:
        type: 'catalog', 'timetable', or 'all' (default).
        region: optional keyword filter on name/description (e.g. 'ICME', 'bow shock', 'MMS').
        query: optional free-text description of the events wanted; the list is then
            ordered by semantic relevance to it instead of by size.

    Returns a list of entries with id, name, type, nb_events, survey range and description.
    Use the `id` field with get_catalog() and get_events_timeseries().

    Example:
        >>> await list_catalogs(type="catalog", region="ICME")
        {'total': 3, 'type_filter': 'catalog', 'region_filter': 'ICME', 'catalogs': [
         {'id': 'amda/sharedcatalog_41', 'name': 'ICME_multi-catalog', 'type': 'catalog',
          'nb_events': 2003, 'survey_start': '1975-01-08', 'survey_stop': '2022-10-21',
          'description': '...'}, ...]}
    """
    return await run_blocking(_list_catalogs_sync, type=type, region=region, query=query)


def _list_catalogs_sync(
    type: str = "all",
    region: str | None = None,
    query: str | None = None,
) -> dict:
    """Synchronous body of `list_catalogs`, run off the event loop by its wrapper."""
    spz = _get_spz()
    if spz is None:
        return {"error": "speasy is not installed"}

    entries = _walk_catalogs(spz)

    # Append local/ catalogs (direct disk read, bypasses TTL cache)
    try:
        for p in sorted(_catalogs_dir_of_bound_user().glob("*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            nb = len(data.get("events", []))
            entries.append(
                {
                    "id": f"local/{p.stem}",
                    "name": data.get("name", p.stem),
                    "type": "catalog",
                    "nb_events": nb,
                    "survey_start": "",
                    "survey_stop": "",
                    "description": data.get("description", "")[:200],
                }
            )
    except Exception as e:
        log.warning("list_catalogs: local catalog scan failed: %s", e)

    for key, spec in _HELIO4CAST.items():
        nb = spec["approx_events"]
        cache = _helio4cast_cache_path(key)
        if cache.exists():
            try:
                with cache.open(encoding="utf-8") as f:
                    nb = max(sum(1 for _ in f) - 1, 0)
            except OSError:
                pass
        entries.append(
            {
                "id": f"helio4cast/{key}",
                "name": spec["name"],
                "type": "catalog",
                "nb_events": nb,
                "survey_start": spec["survey"][0],
                "survey_stop": spec["survey"][1],
                "description": spec["description"][:200],
            }
        )

    if type in ("catalog", "timetable"):
        entries = [e for e in entries if e["type"] == type]

    if region:
        kw = region.lower()
        entries = [e for e in entries if kw in e["name"].lower() or kw in e["description"].lower()]

    # Biggest first is a proxy for relevance and nothing more: with 217 entries and a
    # substring filter, "shock" put a 2 797-event bow-shock list above the interplanetary
    # shock catalogue the question was about. With a query, the catalogue index that
    # `helioai index` builds — 221 embeddings, read by nothing until now — orders the
    # list; entries the index does not know (local, Helio4Cast) keep their place below,
    # by size. Without one, or without the index, the order is what it always was.
    ranked = _semantic_order(query, entries) if query else None
    if ranked is not None:
        entries = ranked
    else:
        entries.sort(key=lambda e: e["nb_events"], reverse=True)

    return {
        "total": len(entries),
        "type_filter": type,
        "region_filter": region,
        **({"query": query, "order": "relevance"} if ranked is not None else {}),
        "catalogs": entries,
    }


def _semantic_order(query: str, entries: list[dict]) -> list[dict] | None:
    """`entries` ordered by relevance to `query`, or None when the index cannot say."""
    try:
        from helioai.tools.rag import search_catalogs

        hits = search_catalogs(query, top_k=max(len(entries), 1))
    except Exception as e:
        log.debug("list_catalogs: semantic order unavailable (%s)", e)
        return None
    if not hits:
        return None
    rank = {h["id"]: i for i, h in enumerate(hits)}
    known = sorted((e for e in entries if e["id"] in rank), key=lambda e: rank[e["id"]])
    rest = sorted((e for e in entries if e["id"] not in rank), key=lambda e: -e["nb_events"])
    return known + rest


async def get_catalog(
    catalog_id: str,
    start: str | None = None,
    stop: str | None = None,
    max_events: int = 10,
    columns: list[str] | None = None,
    where: dict | None = None,
    sort_by: str | None = None,
    descending: bool = False,
    offset: int = 0,
) -> dict:
    """Download and summarize an AMDA event catalog or timetable.

    Args:
        catalog_id: speasy uid from list_catalogs (e.g. 'amda/sharedcatalog_41').
        start:      optional ISO 8601 start — filter events beginning after this time.
        stop:       optional ISO 8601 stop  — filter events beginning before this time.
        max_events: maximum events to include in the sample (default 10).
        columns:    restrict the metadata columns returned per event.
        where:      server-side row filter — {"column": str, "op": "eq|ne|gt|gte|lt|lte|contains", "value": any}.
        sort_by:    column name to sort events by before slicing.
        descending: sort direction (default ascending).
        offset:     pagination offset into the filtered+sorted events.

    Returns catalog metadata + a sample of events (start, stop, key columns).
    Use get_events_timeseries() to download a parameter over all events.

    Example:
        >>> await get_catalog("amda/sharedcatalog_41", start="2015-01-01", stop="2016-01-01",
        ...                   max_events=5, sort_by="start")
        {'_kind': 'catalog_preview', 'catalog_id': 'amda/sharedcatalog_41',
         'name': 'ICME_multi-catalog', 'nb_events_total': 2003, 'nb_events_filtered': ...,
         'returned': 5, 'columns': [...], 'sample': [{'start': ..., 'stop': ..., ...}, ...],
         'survey_start': '1975-01-08', 'survey_stop': '2022-10-21'}
    """
    return await run_blocking(
        _get_catalog_sync,
        catalog_id=catalog_id,
        start=start,
        stop=stop,
        max_events=max_events,
        columns=columns,
        where=where,
        sort_by=sort_by,
        descending=descending,
        offset=offset,
    )


def _get_catalog_sync(
    catalog_id: str,
    start: str | None = None,
    stop: str | None = None,
    max_events: int = 10,
    columns: list[str] | None = None,
    where: dict | None = None,
    sort_by: str | None = None,
    descending: bool = False,
    offset: int = 0,
) -> dict:
    """Synchronous body of `get_catalog`, run off the event loop by its wrapper."""
    spz = _get_spz()
    if spz is None:
        return {"error": "speasy is not installed"}

    try:
        cat, index = _resolve_catalog(catalog_id, spz)
    except Exception as e:
        return {"error": f"Failed to download catalog {catalog_id!r}: {e}"}

    if cat is None:
        return {"error": f"Catalog {catalog_id!r} not found"}

    try:
        events = list(cat)
    except Exception as e:
        return {"error": f"Failed to iterate catalog events: {e}"}

    nb_total = len(events)

    # 1. Time window filter — by event start, see _starts_within
    if start or stop:
        events = [ev for ev in events if _starts_within(ev, start, stop)]

    # 2. where filter
    if where and isinstance(where, dict):
        col = where.get("column", "")
        op = where.get("op", "eq")
        val = where.get("value")
        if col and op and val is not None:
            # Refused rather than applied: an operator with no branch used to filter
            # every event out, and a count of zero reads the same whether the filter
            # was wrong or the window is genuinely empty.
            try:
                events = [ev for ev in events if _match(op, _event_value(ev, col), val)]
            except ValueError as e:
                return {"error": str(e)}

    nb_filtered = len(events)

    # 3. sort
    if sort_by:

        def _sort_key(ev):
            v = _event_value(ev, sort_by)
            try:
                return (0, float(v))
            except (TypeError, ValueError):
                return (1, str(v) if v is not None else "")

        events = sorted(events, key=_sort_key, reverse=descending)

    # 4. pagination + slice
    offset = max(0, offset)
    page = events[offset : offset + max_events]

    # 5. Build event rows
    rows: list[dict] = []
    for ev in page:
        ev_start = _ev_iso(ev, "start_time", "start")
        ev_stop = _ev_iso(ev, "stop_time", "stop")
        row: dict[str, Any] = {"start": ev_start, "stop": ev_stop}
        meta = getattr(ev, "meta", None)
        if meta and isinstance(meta, dict):
            if columns:
                for k in columns:
                    row[k] = meta.get(k)
            else:
                for k, v in list(meta.items())[:8]:
                    row[k] = v
        if sort_by and sort_by not in row:
            row[sort_by] = _event_value(ev, sort_by)
        rows.append(row)

    all_columns = list(rows[0].keys()) if rows else (["start", "stop"] + (columns or []))

    survey_start, survey_stop = _survey(index) if index is not None else ("", "")
    cat_name = _name(index) if index is not None else catalog_id.split("/")[-1]
    cat_type = _spz_type(index) if index is not None else "catalog"
    returned = len(rows)
    return {
        "_kind": "catalog_preview",
        "catalog_id": catalog_id,
        "name": cat_name,
        "type": cat_type,
        "nb_events_total": nb_total,
        "nb_events_filtered": nb_filtered,
        "offset": offset,
        "returned": returned,
        "columns": all_columns,
        "sample": rows,
        "survey_start": survey_start,
        "survey_stop": survey_stop,
        "note": (
            f"Showing rows {offset}–{offset + returned} of {nb_filtered} filtered "
            f"({nb_total} total). "
            + (
                f"Use offset={offset + returned} for the next page. "
                if offset + returned < nb_filtered
                else ""
            )
            + "Use get_events_timeseries() to download a parameter over all events."
        ),
    }


async def get_events_timeseries(
    catalog_id: str,
    param_id: str,
    start: str,
    stop: str,
    max_events: int = 50,
    _data_dir: str | None = None,
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
        _data_dir: injected by the runtime (`tool_exec.trusted_args`) — the session's data
            directory the collection is persisted in. Not exposed in the LLM tool schema.

    Returns per-event statistics and saves the raw data to the workspace for run_python.

    Example:
        >>> await get_events_timeseries("amda/sharedcatalog_41", "amda/imf",
        ...                             "2015-01-01", "2016-01-01", max_events=10)
        {'catalog_id': 'amda/sharedcatalog_41', 'param_id': 'amda/imf', 'stats': [
         {'event': 0, 'start': '2015-01-03T...', 'stop': '2015-01-04T...',
          'n_points': ..., ...}, ...], ...}
    """
    return await run_blocking(
        _get_events_timeseries_sync,
        catalog_id=catalog_id,
        param_id=param_id,
        start=start,
        stop=stop,
        max_events=max_events,
        _data_dir=_data_dir,
    )


def _get_events_timeseries_sync(
    catalog_id: str,
    param_id: str,
    start: str,
    stop: str,
    max_events: int = 50,
    _data_dir: str | None = None,
) -> dict:
    """Synchronous body of `get_events_timeseries`, run off the event loop by its wrapper."""
    spz = _get_spz()
    if spz is None:
        return {"error": "speasy is not installed"}

    # --- resolve catalog ---
    try:
        cat, _ = _resolve_catalog(catalog_id, spz)
    except Exception as e:
        return {"error": f"Failed to download catalog {catalog_id!r}: {e}"}

    if cat is None:
        return {"error": f"Catalog {catalog_id!r} not found"}

    # --- filter events ---
    try:
        events = list(cat)
    except Exception as e:
        return {"error": f"Cannot iterate catalog: {e}"}

    filtered = [ev for ev in events if _starts_within(ev, start, stop)]

    if not filtered:
        return {
            "warning": f"No events found in [{start}, {stop}] for {catalog_id!r}.",
            "suggestion": "Widen the time window or use get_catalog() to inspect the survey range.",
        }

    selected = filtered[:max_events]
    cap_warning = (
        f"Showing first {max_events}/{len(filtered)} events. "
        f"Pass max_events={len(filtered)} for full SEA."
        if len(filtered) > max_events
        else None
    )

    # --- batch download: ONE speasy call for all events ---
    try:
        with speasy_gate:
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
    for i, (ev, ts) in enumerate(zip(selected, timeseries_list, strict=False)):
        ev_start = _ev_iso(ev, "start_time", "start")
        ev_stop = _ev_iso(ev, "stop_time", "stop")
        if ts is None or len(ts.time) == 0:
            stats.append({"event": i, "start": ev_start, "stop": ev_stop, "status": "no_data"})
            continue
        from helioai.datastore import blank_fill

        vals, _ = blank_fill(ts.values, (getattr(ts, "meta", {}) or {}).get("FILLVAL"))
        with np.errstate(all="ignore"):
            entry: dict = {
                "event": i,
                "start": ev_start,
                "stop": ev_stop,
                "n_points": int(len(ts.time)),
            }
            if vals.ndim == 1 or vals.shape[1] == 1:
                flat = vals.ravel()
                entry.update(
                    mean=_fmt(np.nanmean(flat)),
                    std=_fmt(np.nanstd(flat)),
                    min=_fmt(np.nanmin(flat)),
                    max=_fmt(np.nanmax(flat)),
                )
            else:
                col_names = list(getattr(ts, "columns", None) or [])
                if len(col_names) != vals.shape[1]:
                    col_names = [f"c{j}" for j in range(vals.shape[1])]
                components = {}
                for j, cname in enumerate(col_names):
                    col = vals[:, j]
                    components[cname] = {
                        "mean": _fmt(np.nanmean(col)),
                        "std": _fmt(np.nanstd(col)),
                        "min": _fmt(np.nanmin(col)),
                        "max": _fmt(np.nanmax(col)),
                    }
                entry["components"] = components
                n_mag = min(vals.shape[1], 3)
                mag = np.linalg.norm(vals[:, :n_mag], axis=1)
                mag[~np.isfinite(mag)] = np.nan
                entry["magnitude"] = {
                    "mean": _fmt(np.nanmean(mag)),
                    "std": _fmt(np.nanstd(mag)),
                    "min": _fmt(np.nanmin(mag)),
                    "max": _fmt(np.nanmax(mag)),
                }
            stats.append(entry)

    good = [s for s in stats if s.get("status") != "no_data"]
    units = str(getattr(timeseries_list[0], "unit", "") or "") if timeseries_list else ""

    if len(stats) > 10:
        per_event_stats = stats[:5] + stats[-5:]
        stats_note = (
            f"per_event_stats shows first 5 + last 5 of {len(stats)} events. "
            "Full data available via load_data()."
        )
    else:
        per_event_stats = stats
        stats_note = None

    # Persist event collection for reuse in run_python via load_data()
    from helioai.datastore import save_event_collection

    series = []
    for ev, ts in zip(selected, timeseries_list, strict=False):
        ev_start = _ev_iso(ev, "start_time", "start")
        ev_stop = _ev_iso(ev, "stop_time", "stop")
        series.append((ev_start, ev_stop, ts if (ts is not None and len(ts.time) > 0) else None))

    saved = save_event_collection(
        param_id,
        series=series,
        param_id=param_id,
        units=units,
        source="get_events_timeseries",
        data_dir=Path(_data_dir) if _data_dir else None,
    )

    result: dict = {
        "catalog_id": catalog_id,
        "param_id": param_id,
        "time_window": [start, stop],
        "n_events_found": len(filtered),
        "n_events_downloaded": len(selected),
        "n_events_with_data": len(good),
        "units": units,
        "per_event_stats": per_event_stats,
    }
    if cap_warning:
        result["cap_warning"] = cap_warning
    if stats_note:
        result["stats_note"] = stats_note
    if saved:
        ds_name = saved["dataset"]
        result["dataset"] = ds_name
        result["note"] = (
            f"In run_python: events = load_data({ds_name!r}) — "
            "a list of objects with .time, .values, .start, .stop, .units per event. "
            "Use the superposed_epoch recipe: load_recipe('superposed_epoch')."
        )
    else:
        result["note"] = (
            "Use run_python with spz.get_data(param_id, events) for custom plots. "
            "The catalog events are speasy DateTimeRange objects iterable from the catalog."
        )
    return result


def _fmt(val) -> float | None:
    try:
        v = float(val)
        return round(v, 4) if abs(v) < 1e10 else None
    except Exception:
        return None


# ── local catalog storage (local/<name>) ──────────────────────────────────────

_LOCAL_NAME_RE = re.compile(r"^[a-z0-9_\-]{1,40}$")
_MAX_EVENTS_LOCAL = 5000


def _catalogs_dir_of_bound_user() -> Path:
    from helioai.workspace import current_user, user_home

    d = user_home(current_user()) / "catalogs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_local_catalog(name: str):
    """Load a local catalog from JSON and reconstruct a speasy Catalog."""
    from speasy.products import Catalog, Event

    path = _catalogs_dir_of_bound_user() / f"{name}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    events = [
        Event(ev["start"], ev["stop"], meta=ev.get("meta") or {}) for ev in data.get("events", [])
    ]
    return Catalog(name=data.get("name", name), meta={}, events=events)


def _resolve_catalog(catalog_id: str, spz):
    """Return (speasy_catalog_object, index_or_None) for amda/, local/ or helio4cast/ prefixes."""
    if catalog_id.startswith("local/"):
        name = catalog_id[len("local/") :]
        cat = _load_local_catalog(name)
        if cat is None:
            return None, None
        return cat, None

    if catalog_id.startswith("helio4cast/"):
        name = catalog_id[len("helio4cast/") :]
        cat = _load_helio4cast_catalog(name)
        if cat is None:
            return None, None
        return cat, None

    uid = catalog_id.removeprefix("amda/")
    flat = spz.inventories.flat_inventories.amda
    cats = getattr(flat, "catalogs", {}) or {}
    tts = getattr(flat, "timetables", {}) or {}
    index = cats.get(uid) or tts.get(uid)
    if index is None:
        return None, None
    with speasy_gate:
        cat = spz.get_data(index)
    return cat, index


async def save_catalog(
    name: str,
    events: list[dict],
    description: str = "",
    _catalogs_dir: str | None = None,
) -> dict:
    """Save a list of events as a local catalog under the local/<name> prefix.

    Args:
        name:        Catalog name — lowercase letters, digits, hyphens, underscores (1-40 chars).
        events:      List of dicts with 'start' and 'stop' ISO 8601 strings plus optional extra keys.
        description: Short description (optional).
        _catalogs_dir: injected by the runtime (`tool_exec.trusted_args`) — the user's
            catalogue directory. Not exposed in the LLM tool schema.

    Returns {"catalog_id": "local/<name>", "nb_events": N, "note": "..."}.
    Overwrites an existing catalog with the same name.
    Use list_catalogs() then get_catalog("local/<name>") to inspect it.

    Example:
        >>> await save_catalog("my-shocks",
        ...                    [{"start": "2015-03-17T04:01:00", "stop": "2015-03-17T05:00:00",
        ...                      "note": "St. Patrick's Day storm shock"}])
        {'catalog_id': 'local/my-shocks', 'nb_events': 1, 'overwritten': False, 'note': '...'}
    """
    return await run_blocking(
        _save_catalog_sync,
        name=name,
        events=events,
        description=description,
        _catalogs_dir=_catalogs_dir,
    )


def _save_catalog_sync(
    name: str,
    events: list[dict],
    description: str = "",
    _catalogs_dir: str | None = None,
) -> dict:
    """Synchronous body of `save_catalog`, run off the event loop by its wrapper."""
    if not _LOCAL_NAME_RE.fullmatch(name):
        return {
            "error": (
                f"Invalid catalog name {name!r} — "
                "use 1-40 lowercase letters, digits, hyphens or underscores"
            )
        }
    if not events:
        return {"error": "events list is empty — provide at least one event"}
    if len(events) > _MAX_EVENTS_LOCAL:
        return {"error": f"Too many events ({len(events)} > {_MAX_EVENTS_LOCAL} cap)"}

    validated: list[dict] = []
    for i, ev in enumerate(events):
        s = str(ev.get("start", "")).strip()
        e = str(ev.get("stop", "")).strip()
        if not s or not e:
            return {"error": f"Event {i}: 'start' and 'stop' are required"}
        if s >= e:
            return {"error": f"Event {i}: start >= stop ({s!r} >= {e!r})"}
        meta = {k: v for k, v in ev.items() if k not in ("start", "stop")}
        validated.append({"start": s, "stop": e, "meta": meta})

    import datetime

    payload = {
        "name": name,
        "description": description,
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "events": validated,
    }
    target = Path(_catalogs_dir) if _catalogs_dir else _catalogs_dir_of_bound_user()
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{name}.json"
    overwritten = path.exists()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "catalog_id": f"local/{name}",
        "nb_events": len(validated),
        "overwritten": overwritten,
        "note": (
            f"Saved {len(validated)} events as local/{name}. "
            "Use get_catalog('local/" + name + "') to inspect or "
            "get_events_timeseries('local/" + name + "', param_id, ...) to analyse."
        ),
    }
