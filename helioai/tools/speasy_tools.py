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
        import speasy as spz
        import numpy as np
    except ImportError:
        return {"error": "speasy is not installed. Run: pip install speasy"}

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

    return {
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
                "groups": [{"query": q, "results": r} for q, r in zip(queries, batch)],
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
