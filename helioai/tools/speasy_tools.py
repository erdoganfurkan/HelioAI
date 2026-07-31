"""speasy tools: data access layer wrapping the speasy library.

speasy provides unified access to 70+ missions and 65k+ products from
CDAWeb, AMDA, CSA, SSC and others. These tools are the helioai equivalent
of AMDA's download_timeseries and list_parameters.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


async def get_timeseries(
    param_id: str,
    start: str,
    stop: str,
    max_points: int = 5000,
) -> dict:
    """Download a time series from any speasy provider.

    Args:
        param_id: speasy parameter id (e.g. 'amda/imf', 'cdaweb/AC_H0_MFI/BGSEc')
        start: ISO 8601 start time (e.g. '2024-01-01T00:00:00')
        stop:  ISO 8601 stop time
        max_points: max samples to return (downsampled if needed)

    The param_id should be in speasy format: "{provider}/{xmlid}"
    e.g. "amda/ace_epam_ca60_he", "cda/ACE_H0_MFI/BGSEc"
    (returned by search_parameters).

    Returns dict with: param_id, start, stop, units, shape, n_points, preview (first 10 rows as CSV)
    """
    try:
        import numpy as np
        import speasy as spz
    except ImportError:
        return {"error": "speasy is not installed. Run: pip install speasy"}

    blocking, coverage_note = _coverage_check(spz, np, param_id, start, stop)
    if blocking is not None:
        return blocking

    try:
        var = spz.get_data(param_id, start, stop)
    except Exception as e:
        log.warning("speasy.get_data failed: %s", e)
        return {"error": f"Failed to retrieve {param_id!r}: {e}"}

    if var is None:
        return {"error": f"No data returned for {param_id!r} between {start} and {stop}"}

    times = var.time
    values = var.values

    n_points = len(times)
    if n_points == 0:
        return {"error": f"Empty dataset for {param_id!r}"}

    # Replace fill values with NaN HERE, once, before anything else sees the array.
    # Everything downstream — the preview shown to the agent, the .npz the sandbox
    # loads, the exported notebook — then works on data where "no measurement" is
    # NaN rather than -1e31 or 99999.9. Leaving it to the caller meant the sandbox
    # had to remember to call clean(), which cannot see FILLVAL anyway, so a
    # forgotten call put a 99999.9 "speed" straight into a plot and an average.
    from helioai.datastore import blank_fill

    values, fill_mask = blank_fill(values, (getattr(var, "meta", {}) or {}).get("FILLVAL"))
    quality = _data_quality(times, values, np, fill_mask=fill_mask)

    # An all-fill series is functionally as empty as a zero-length one: the rows
    # exist but not one of them holds a measurement. Reporting it as a successful
    # download of n points is a lie the agent then plots — the ACE/SWEPAM
    # saturation during the 2003 Halloween storm returns 1913 such rows. Fail
    # here instead, and do not persist the garbage.
    if quality.get("missing_pct") == 100.0:
        return {
            "error": (
                f"{param_id!r} returned {n_points} rows between {start} and {stop} but "
                f"every value is a fill value — the instrument has no valid data for "
                f"this window."
            ),
            "suggestion": (
                "Try another instrument or spacecraft for this quantity; the parameter "
                "id is fine, the coverage is not."
            ),
            "n_points": n_points,
            "missing_pct": 100.0,
        }

    # Persist full-resolution data before downsampling
    from helioai.datastore import save_timeseries

    saved = save_timeseries(
        param_id,
        time=times,
        values=values,
        param_id=param_id,
        units=str(getattr(var, "unit", "") or ""),
        start=start,
        stop=stop,
        columns=list(getattr(var, "columns", None) or []),
        source="get_timeseries",
    )

    # Downsample if needed
    if n_points > max_points:
        step = n_points // max_points
        times = times[::step]
        values = values[::step]
        n_points = len(times)

    # Build a brief preview (first 10 rows as CSV text)
    preview_lines: list[str] = []
    for i in range(min(10, n_points)):
        t_str = str(times[i])
        v = values[i]
        if hasattr(v, "__len__"):
            v_str = ", ".join(f"{x:.4g}" for x in v)
        else:
            v_str = f"{float(v):.6g}"
        preview_lines.append(f"{t_str}  {v_str}")
    preview = "\n".join(preview_lines)

    shape = list(values.shape)
    units = getattr(var, "unit", "") or ""
    name = getattr(var, "name", "") or ""
    components = list(getattr(var, "columns", None) or [])

    # Cadence: median of time deltas (provider-agnostic)
    cadence = ""
    try:
        if n_points > 1:
            deltas = np.diff(times.astype("datetime64[ms]").astype(float))
            med_ms = float(np.median(deltas))
            if med_ms >= 3_600_000:
                cadence = f"{med_ms / 3_600_000:.4g} h"
            elif med_ms >= 60_000:
                cadence = f"{med_ms / 60_000:.4g} min"
            elif med_ms >= 1_000:
                cadence = f"{med_ms / 1_000:.4g} s"
            else:
                cadence = f"{med_ms:.4g} ms"
    except Exception:
        pass

    # Mission / instrument: best-effort from param_id prefix + var.meta
    mission = ""
    instrument = ""
    try:
        parts = param_id.split("/")
        dataset = parts[1] if len(parts) > 1 else parts[0]
        # First token before '_'/'-' is the mission (cda/ACE_H0_MFI/.. → ACE, amda/ace_imf_all → ace)
        mission = dataset.split("_")[0].split("-")[0]
    except Exception:
        pass
    try:
        meta = getattr(var, "meta", {}) or {}
        instrument = str(meta.get("FIELDNAM", "") or meta.get("VAR_NOTES", "") or "")[:80]
    except Exception:
        pass

    result: dict = {
        "param_id": param_id,
        "name": name,
        "start": start,
        "stop": stop,
        "units": str(units),
        "components": components,
        "cadence": cadence,
        "mission": mission,
        "instrument": instrument,
        "shape": shape,
        "n_points": n_points,
        "preview": preview,
    }
    if quality:
        result["quality"] = quality
    if coverage_note:
        result["coverage_note"] = coverage_note
    if saved:
        ds_name = saved["dataset"]
        result["dataset"] = ds_name
        result["dataset_note"] = f"use load_data({ds_name!r}) in run_python — never spz.get_data"
    return result


_PROVIDERS_WITH_RANGE = ("amda", "cda", "csa", "ssc")


def _coverage_check(
    spz, np, param_id: str, start: str, stop: str
) -> tuple[dict | None, str | None]:
    """Compare the requested window against the parameter's published coverage.

    Returns (blocking_result, note). The blocking result is returned to the agent
    instead of downloading; the note rides along with a successful download.

    Only a request that does not overlap the coverage AT ALL is refused. A partial
    overlap still downloads, because refusing it throws away real data: the guard
    used to reject whenever the window merely extended past either edge, which for
    a parameter ending last month means an ordinary "the last few weeks" request
    returns nothing. That mattered little while this ran for AMDA alone; it now
    covers CDA's ~68k parameters too.

    Answering "no data" without the coverage is what let the agent invent a Wind
    position for 2015 after `WI_OR_DEF/GSE_POS` (1994-1997) and `WI_K0_3DP`
    (2019 onward) both came back empty — the dates would have named the problem.
    """
    provider = param_id.split("/", 1)[0] if "/" in param_id else ""
    if provider not in _PROVIDERS_WITH_RANGE:
        return None, None
    try:
        rng = getattr(spz, provider).parameter_range(param_id.split("/", 1)[1])
        if rng is None:
            return None, None
        # speasy's DateTimeRange exposes start_time/stop_time as tz-aware datetimes.
        # isoformat, not str(): str() renders a space separator that datetime64 rejects.
        avail_start = rng.start_time.isoformat()[:19]
        avail_stop = rng.stop_time.isoformat()[:19]
        req_start = np.datetime64(start[:19])
        req_stop = np.datetime64(stop[:19])
        cov_start = np.datetime64(avail_start)
        cov_stop = np.datetime64(avail_stop)
    except Exception as e:
        # Never block a download over a coverage lookup — but log it, because
        # swallowing this silently is exactly how the previous version stayed
        # broken: it read rng.start, which does not exist, and the bare except
        # made the guard a no-op that its mocked test still passed.
        log.warning("coverage check failed for %r: %s", param_id, e)
        return None, None

    if req_stop < cov_start or req_start > cov_stop:
        return {
            "error": (
                f"{param_id!r} has no data in [{start}, {stop}] — it covers "
                f"[{avail_start}, {avail_stop}]."
            ),
            "suggestion": (
                "The window and the parameter do not overlap at all. Pick a different "
                "parameter for this event rather than a different window, unless you "
                "meant to study a date inside the coverage."
            ),
            "available_start": avail_start,
            "available_stop": avail_stop,
        }, None

    if req_start < cov_start or req_stop > cov_stop:
        return None, (
            f"Requested [{start}, {stop}] extends past the coverage of {param_id!r} "
            f"([{avail_start}, {avail_stop}]) — the series is clipped to the overlap."
        )
    return None, None


def _data_quality(times, values, np, fillval=None, fill_mask=None) -> dict:
    """Deterministic data-quality checks on full-resolution arrays.

    Returns {} when the data is not numeric/datetime (provider-agnostic, like the
    cadence block). The `notable` flag gates whether the agent and the web card
    surface it: missing > 5%, any gap, or any 5σ outlier.

    Args:
        fillval: the dataset's declared CDF FILLVAL. Ignored when `fill_mask` is
            given.
        fill_mask: a mask already computed by `_blank_fill_values`. Passed by
            `get_timeseries`, which blanks fill to NaN before this runs — without
            it the count would be recomputed from values whose fill is now NaN,
            which still works but repeats the comparison for nothing.
    """
    try:
        from helioai.datastore import fill_mask as _mask_of

        bad = _mask_of(values, fillval) if fill_mask is None else fill_mask
        total = values.size
        missing_pct = round(100 * float(bad.sum()) / total, 1) if total else 0.0

        gaps: list[dict] = []
        if len(times) > 2:
            deltas = np.diff(times.astype("datetime64[ms]").astype(float))
            med = float(np.median(deltas))
            if med > 0:
                for i in np.where(deltas > 3 * med)[0][:10]:
                    gaps.append(
                        {
                            "start": str(times[i])[:19],
                            "dur_h": round(float(deltas[i]) / 3_600_000, 2),
                        }
                    )

        outliers = 0
        clean = values[~bad]
        if clean.size > 1:
            mean = float(clean.mean())
            std = float(clean.std())
            if std > 0:
                outliers = int((np.abs(clean - mean) > 5 * std).sum())

        notable = bool(missing_pct > 5 or gaps or outliers > 0)
        return {
            "missing_pct": missing_pct,
            "gaps": gaps,
            "outliers_5sigma": outliers,
            "notable": notable,
        }
    except Exception:
        return {}


async def list_missions() -> dict:
    """List available speasy data providers and their top-level missions.

    Returns a summary dict with provider names and approximate product counts.
    """
    try:
        import speasy as spz
    except ImportError:
        return {"error": "speasy is not installed. Run: pip install speasy"}

    providers: dict[str, int] = {}
    try:
        tree = spz.inventories.tree
        for attr in dir(tree):
            if attr.startswith("_"):
                continue
            node = getattr(tree, attr, None)
            if node is not None:
                providers[attr] = "available"
    except Exception as e:
        log.warning("Failed to walk speasy inventory: %s", e)
        return {"error": str(e)}

    return {
        "providers": list(providers.keys()),
        "note": (
            "Use search_parameters to find specific parameters. "
            "Common provider prefixes: amda/, cdaweb/, csa/, ssc/"
        ),
    }


async def search_parameters(
    query: str | None = None,
    top_k: int = 5,
    provider: str | None = None,
    queries: list[str] | None = None,
) -> dict:
    """Semantic search over the speasy catalog (65k+ products).

    Requires the index to be built first (run: helioai index).
    Falls back to a direct speasy text match if no index is found.

    Args:
        query: free-text English query for a SINGLE parameter.
        queries: list of queries to resolve SEVERAL parameters in ONE call
                 (preferred when 2+ parameters are needed — much cheaper).
        top_k: number of results per query.
        provider: optional — restrict to one provider (amda/cda/csa/ssc).

    Returns either {query, provider, results} (single) or
    {provider, groups: [{query, results}]} (batch).
    """
    if queries:
        try:
            from helioai.tools.rag import search_batch as rag_search_batch

            batch = rag_search_batch(queries, top_k=top_k, provider=provider)
            return {
                "provider": provider,
                "groups": [
                    {"query": q, "results": r} for q, r in zip(queries, batch, strict=False)
                ],
            }
        except Exception as e:
            log.warning("RAG batch search failed (%s), falling back to text scan", e)
        try:
            import speasy as spz

            groups = [{"query": q, "results": _fallback_search(spz, q, top_k)} for q in queries]
            return {
                "provider": provider,
                "groups": groups,
                "note": "RAG index not built — using text fallback (provider filter ignored)",
            }
        except Exception as e2:
            return {"error": f"Search failed: {e2}"}

    if not query:
        return {"error": "provide query (string) or queries (list of strings)"}

    try:
        from helioai.tools.rag import search as rag_search

        results = rag_search(query, top_k=top_k, provider=provider)
        return {"query": query, "provider": provider, "results": results}
    except Exception as e:
        log.warning("RAG search failed (%s), falling back to speasy inventory scan", e)

    # Fallback: naive text search on speasy inventory (provider filter ignored — best effort)
    try:
        import speasy as spz

        results = _fallback_search(spz, query, top_k)
        return {
            "query": query,
            "results": results,
            "note": "RAG index not built — using text fallback (provider filter ignored)",
        }
    except Exception as e2:
        return {"error": f"Search failed: {e2}"}


def _fallback_search(spz, query: str, top_k: int) -> list[dict]:
    """Naive case-insensitive text match on speasy inventory when RAG is unavailable."""
    q = query.lower()
    results: list[dict] = []

    def _walk(node, depth: int = 0) -> None:
        if depth > 8 or len(results) >= top_k * 3:
            return
        for attr in dir(node):
            if attr.startswith("_"):
                continue
            child = getattr(node, attr, None)
            if child is None:
                continue
            name = getattr(child, "name", attr) or attr
            desc = getattr(child, "desc", "") or ""
            uid = getattr(child, "uid", "") or ""
            if q in name.lower() or q in desc.lower() or q in attr.lower():
                results.append(
                    {
                        "id": str(uid) or attr,
                        "name": str(name),
                        "description": str(desc)[:120],
                        "score": 0.5,
                    }
                )
            if not hasattr(child, "uid") and depth < 6:
                _walk(child, depth + 1)

    try:
        _walk(spz.inventories.tree)
    except Exception:
        pass

    return results[:top_k]
