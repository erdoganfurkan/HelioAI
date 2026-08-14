"""speasy tools: data access layer wrapping the speasy library.

speasy provides unified access to 70+ missions and 65k+ products from
CDAWeb, AMDA, CSA, SSC and others. These tools are the helioai equivalent
of AMDA's download_timeseries and list_parameters.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def _sample_cadence(times, values) -> tuple[str, int]:
    """Median gap between samples that actually carry a measurement, and how many.

    Measured over the whole time grid instead, this reports the file's epoch spacing,
    which on an ISTP file padded with fill rows is not the rate anything was sampled
    at: `WI_PM_3DP` reported "8 ms" for protons measured every 3.08 s, a factor of 385.

    Wrong is bad; wrong in this direction is worse, because cadence is exactly what the
    agent is asked to choose on. A run briefed "give me a few seconds, not the
    sub-second product" reads "8 ms" and rejects the very product it wanted — talked
    out of the right answer by our own metadata rather than by the data.

    `n_valid` ships alongside because the two numbers only make sense together: 61 %
    missing next to a 3 s cadence reads as a broken product until you can see that
    46 157 of the 119 168 rows are real and evenly spaced.

    Args:
        times: Sample times, full resolution (never the thinned preview copy).
        values: Matching values, already fill-blanked to NaN.

    Returns:
        (human-readable cadence, number of samples carrying a measurement).
    """
    import numpy as np

    n_valid = len(times)
    sample_times = times
    try:
        finite = np.isfinite(np.asarray(values, dtype="float64"))
        if finite.ndim > 1:
            finite = finite.any(axis=tuple(range(1, finite.ndim)))
        n_valid = int(finite.sum())
        if n_valid > 1:
            sample_times = times[finite]
    except (TypeError, ValueError):
        pass  # non-numeric variable: the grid is all there is to measure

    if len(sample_times) < 2:
        return "", n_valid
    try:
        med_ms = float(np.median(np.diff(sample_times.astype("datetime64[ms]").astype(float))))
    except Exception:
        return "", n_valid
    if med_ms >= 3_600_000:
        return f"{med_ms / 3_600_000:.4g} h", n_valid
    if med_ms >= 60_000:
        return f"{med_ms / 60_000:.4g} min", n_valid
    if med_ms >= 1_000:
        return f"{med_ms / 1_000:.4g} s", n_valid
    return f"{med_ms:.4g} ms", n_valid


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

    Example:
        >>> await get_timeseries("amda/imf", "2010-01-01T00:00:00", "2010-01-01T06:00:00")
        {'dataset': 'imf', 'param_id': 'amda/imf', 'units': 'nT',
         'components': ['bx', 'by', 'bz'], 'cadence': '16 s', 'shape': [1350, 3],
         'n_points': 1350, 'n_valid': 1350, 'quality': {'missing_pct': 0.0, ...},
         'preview': '2010-01-01T00:00:09.000000000  -1.839, 2.308, 0.108\\n...', ...}
    """
    try:
        import numpy as np
        import speasy as spz
    except ImportError:
        return {"error": "speasy is not installed. Run: pip install speasy"}

    from helioai.datastore import find_existing

    cached = find_existing(param_id, start, stop)
    if cached is not None:
        return {
            "dataset": cached,
            "dataset_note": f"use load_data({cached!r}) in run_python — never spz.get_data",
            "param_id": param_id,
            "start": start,
            "stop": stop,
            "already_downloaded": True,
            "note": "This session already holds this parameter for this exact interval. "
            "Nothing was re-fetched; read it with load_data().",
        }

    blocking, coverage_note = _coverage_check(spz, np, param_id, start, stop)
    if blocking is not None:
        return blocking

    try:
        var = spz.get_data(param_id, start, stop)
    except Exception as e:
        log.warning("speasy.get_data failed: %s", e)
        # `str(e)` alone can be a bare "tuple index out of range", which tells the agent
        # nothing about what to do next. Name the exception, and recognise the shape a
        # non-data variable (a CDF time axis) produces so the agent looks for a real
        # product instead of retrying the same id.
        detail = f"{type(e).__name__}: {e}"
        if isinstance(e, IndexError | TypeError):
            detail += (
                " — this often means the id is not a plottable data variable "
                "(a CDF time axis or support variable). Search for the measured "
                "quantity itself and use that id."
            )
        return {"error": f"Failed to retrieve {param_id!r}: {detail}"}

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

    # Cadence and valid-sample count describe the PERSISTED series, so they are taken
    # before the preview is thinned: `load_data()` hands back the full resolution, and
    # a cadence measured on the thinned copy would describe something the agent never
    # works with.
    cadence, n_valid = _sample_cadence(times, values)

    # Downsample if needed
    if n_points > max_points:
        step = n_points // max_points
        times = times[::step]
        values = values[::step]
        n_points = len(times)

    # Build a brief preview (first 10 rows as CSV text). Nothing here may raise: a
    # non-numeric variable (datetime64 values, string labels) used to kill the whole
    # call from inside this loop, with no try/except above it.
    preview_lines: list[str] = []
    for i in range(min(10, n_points)):
        v = values[i]
        try:
            if hasattr(v, "__len__"):
                v_str = ", ".join(f"{x:.4g}" for x in v)
            else:
                v_str = f"{float(v):.6g}"
        except (TypeError, ValueError):
            v_str = str(v)[:60]
        preview_lines.append(f"{times[i]}  {v_str}")
    preview = "\n".join(preview_lines)

    shape = list(values.shape)
    units = getattr(var, "unit", "") or ""
    name = getattr(var, "name", "") or ""
    components = list(getattr(var, "columns", None) or [])

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

    # `dataset` leads the payload on purpose. Stale tool results are summarised by
    # serialising this dict and cutting it to a fixed length, so trailing keys are the
    # ones that vanish — and losing the handle is what makes the agent re-download a
    # parameter it already has instead of calling load_data().
    result: dict = {}
    if saved:
        ds_name = saved["dataset"]
        result["dataset"] = ds_name
        result["dataset_note"] = f"use load_data({ds_name!r}) in run_python — never spz.get_data"
    result |= {
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
        "n_valid": n_valid,
        "preview": preview,
    }
    if quality:
        result["quality"] = quality
    if coverage_note:
        result["coverage_note"] = coverage_note
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

    Example:
        >>> await list_missions()
        {'providers': ['amda', 'archive', 'cda', 'csa', 'ssc', 'uiowaephtool'],
         'note': 'Use search_parameters to find specific parameters. ...'}
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


def _apply_window(results: list[dict], window: tuple[str, str] | None) -> list[dict]:
    """Flag and demote products whose published coverage misses the requested window.

    Demoted rather than dropped: the index can lag the archive, and `get_timeseries`
    keeps its own coverage guard for exactly that case.
    """
    if not window or not results:
        return results
    from helioai.tools.rag import _covers

    for r in results:
        if not _covers(r.get("coverage", ""), window):
            r["covers_window"] = False
    return sorted(results, key=lambda r: r.get("covers_window", True) is False)


async def search_parameters(
    query: str | None = None,
    top_k: int = 5,
    provider: str | None = None,
    queries: list[str] | None = None,
    start: str | None = None,
    stop: str | None = None,
) -> dict:
    """Semantic search over the speasy catalog (83k+ products).

    Requires the index to be built first (run: helioai index).
    Falls back to a direct speasy text match if no index is found.

    Every result carries the product's published `coverage`. Passing the window you
    intend to download sorts products that cannot cover it to the bottom and flags them,
    which is cheaper than discovering it one failed download at a time — resolving a
    2015 ephemeris used to cost four turns against products that stop in 1997.

    Args:
        query: free-text English query for a SINGLE parameter.
        queries: list of queries to resolve SEVERAL parameters in ONE call
                 (preferred when 2+ parameters are needed — much cheaper).
        top_k: number of results per query.
        provider: optional — restrict to one provider (amda/cda/csa/ssc).
        start: optional ISO start of the interval you intend to download.
        stop: optional ISO stop. Both are needed for the filter to apply.

    Returns either {query, provider, results} (single) or
    {provider, groups: [{query, results}]} (batch).

    Example:
        >>> await search_parameters(query="ACE solar wind proton density", top_k=3)
        {'query': 'ACE solar wind proton density', 'provider': None, 'results': [
         {'id': 'cda/AC_H2_SWE/Np',
          'description': 'Proton No. density. Solar Wind Proton Number Density, scalar. '
                         'ACE/SWEPAM ... 1-Hour Level 2 Data ... Units: #/cc. ...',
          'coverage': '1998-02-04 → 2024-07-09', 'score': 1.0}, ...]}
    """
    window = (start, stop) if start and stop else None
    if queries:
        try:
            from helioai.tools.rag import search_batch as rag_search_batch

            batch = rag_search_batch(queries, top_k=top_k, provider=provider)
            return {
                "provider": provider,
                "groups": [
                    {"query": q, "results": _apply_window(r, window)}
                    for q, r in zip(queries, batch, strict=False)
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
        return {"query": query, "provider": provider, "results": _apply_window(results, window)}
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
