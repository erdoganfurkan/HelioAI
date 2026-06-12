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

log = logging.getLogger(__name__)


def _ev_iso(ev, attr_time: str, attr_plain: str) -> str:
    raw = str(getattr(ev, attr_time, "") or getattr(ev, attr_plain, "") or "")[:19]
    return raw.replace(" ", "T")


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


def _match(op: str, a, b) -> bool:
    """Apply comparison operator op between event value a and filter value b."""
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
    return False


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

    # Append local/ catalogs (direct disk read, bypasses TTL cache)
    try:
        for p in sorted(_catalogs_dir().glob("*.json")):
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
    """
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

    # 1. Time window filter
    if start or stop:
        filtered: list = []
        for ev in events:
            ev_start = _ev_iso(ev, "start_time", "start")
            if start and ev_start < start:
                continue
            if stop and ev_start > stop:
                continue
            filtered.append(ev)
        events = filtered

    # 2. where filter
    if where and isinstance(where, dict):
        col = where.get("column", "")
        op = where.get("op", "eq")
        val = where.get("value")
        if col and op and val is not None:
            events = [ev for ev in events if _match(op, _event_value(ev, col), val)]

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

    filtered = []
    for ev in events:
        ev_start = _ev_iso(ev, "start_time", "start")
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
        ev_start = _ev_iso(ev, "start_time", "start")
        ev_stop = _ev_iso(ev, "stop_time", "stop")
        if ts is None or len(ts.time) == 0:
            stats.append({"event": i, "start": ev_start, "stop": ev_stop, "status": "no_data"})
            continue
        vals = ts.values.astype(float)
        vals[~np.isfinite(vals)] = np.nan
        vals[np.abs(vals) >= 1e30] = np.nan
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

    # Persist event collection for reuse in run_python via load_data()
    from helioai.datastore import save_event_collection

    series = []
    for ev, ts in zip(selected, timeseries_list):
        ev_start = _ev_iso(ev, "start_time", "start")
        ev_stop = _ev_iso(ev, "stop_time", "stop")
        series.append((ev_start, ev_stop, ts if (ts is not None and len(ts.time) > 0) else None))

    saved = save_event_collection(
        param_id,
        series=series,
        param_id=param_id,
        units=units,
        source="get_events_timeseries",
    )

    result: dict = {
        "catalog_id": catalog_id,
        "param_id": param_id,
        "time_window": [start, stop],
        "n_events_found": len(filtered),
        "n_events_downloaded": len(selected),
        "n_events_with_data": len(good),
        "units": units,
        "per_event_stats": stats,
    }
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


def _catalogs_dir() -> Path:
    from helioai.config import settings

    d = Path(settings.catalogs.catalogs_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_local_catalog(name: str):
    """Load a local catalog from JSON and reconstruct a speasy Catalog."""
    from speasy.products import Catalog, Event

    path = _catalogs_dir() / f"{name}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    events = [
        Event(ev["start"], ev["stop"], meta=ev.get("meta") or {}) for ev in data.get("events", [])
    ]
    return Catalog(name=data.get("name", name), meta={}, events=events)


def _resolve_catalog(catalog_id: str, spz):
    """Return (speasy_catalog_object, index_or_None) for amda/ or local/ prefixes."""
    if catalog_id.startswith("local/"):
        name = catalog_id[len("local/") :]
        cat = _load_local_catalog(name)
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
    cat = spz.get_data(index)
    return cat, index


async def save_catalog(
    name: str,
    events: list[dict],
    description: str = "",
) -> dict:
    """Save a list of events as a local catalog under the local/<name> prefix.

    Args:
        name:        Catalog name — lowercase letters, digits, hyphens, underscores (1-40 chars).
        events:      List of dicts with 'start' and 'stop' ISO 8601 strings plus optional extra keys.
        description: Short description (optional).

    Returns {"catalog_id": "local/<name>", "nb_events": N, "note": "..."}.
    Overwrites an existing catalog with the same name.
    Use list_catalogs() then get_catalog("local/<name>") to inspect it.
    """
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
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "events": validated,
    }
    path = _catalogs_dir() / f"{name}.json"
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
