"""NASA ADS literature search — find_papers tool."""

from __future__ import annotations

import httpx

from helioai.config import settings
from helioai.logging_config import get_logger

log = get_logger(__name__)

_ADS_URL = "https://api.adsabs.harvard.edu/v1/search/query"
_FIELDS = "title,author,year,bibcode,doi,citation_count,abstract"
_MAX_ROWS = 10
_ABSTRACT_CHARS = 300
_SORTS = {"relevance": "score desc", "date": "date desc", "citations": "citation_count desc"}


async def find_papers(
    query: str,
    max_results: int = 5,
    year_start: int | None = None,
    year_end: int | None = None,
    sort: str = "relevance",
    _transport: httpx.AsyncBaseTransport | None = None,
) -> dict:
    token = settings.literature.ads_token
    if not token:
        return {
            "error": (
                "ADS_API_TOKEN is not set — get a free key at "
                "https://ui.adsabs.harvard.edu/user/settings/token and add it to .env"
            )
        }

    q = query
    if year_start or year_end:
        q = f"{query} year:{year_start or ''}-{year_end or ''}"
    params = {
        "q": q,
        "fl": _FIELDS,
        "rows": min(max(max_results, 1), _MAX_ROWS),
        "sort": _SORTS.get(sort, _SORTS["relevance"]),
    }

    async with httpx.AsyncClient(timeout=15, transport=_transport) as client:
        try:
            resp = await client.get(
                _ADS_URL, params=params, headers={"Authorization": f"Bearer {token}"}
            )
        except httpx.HTTPError as e:
            return {"error": f"ADS request failed: {e}"}

    if resp.status_code != 200:
        return {"error": f"ADS returned HTTP {resp.status_code}: {resp.text[:200]}"}

    docs = resp.json().get("response", {}).get("docs", [])
    papers = [_slim(d) for d in docs]
    log.info("find_papers", query=query, n_results=len(papers))
    return {
        "query": q,
        "papers": papers,
        "note": (
            "Cite as: Authors (year), bibcode. "
            "Full record: https://ui.adsabs.harvard.edu/abs/<bibcode>"
        ),
    }


def _first(value) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def _slim(doc: dict) -> dict:
    authors = doc.get("author") or []
    first_author = authors[0] if authors else ""
    return {
        "title": _first(doc.get("title")),
        "authors": f"{first_author} et al." if len(authors) > 1 else first_author,
        "year": doc.get("year", ""),
        "bibcode": doc.get("bibcode", ""),
        "doi": _first(doc.get("doi")),
        "citations": doc.get("citation_count", 0),
        "abstract": (doc.get("abstract") or "")[:_ABSTRACT_CHARS],
    }
