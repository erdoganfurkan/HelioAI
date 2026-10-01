"""NASA ADS literature search — find_papers tool."""

from __future__ import annotations

import re

import httpx

from helioai.config import settings
from helioai.logging_config import get_logger

log = get_logger(__name__)

_ADS_URL = "https://api.adsabs.harvard.edu/v1/search/query"
_FIELDS = "title,author,year,bibcode,doi,citation_count,abstract"
_MAX_ROWS = 10
_ABSTRACT_CHARS = 300
_SORTS = {"relevance": "score desc", "date": "date desc", "citations": "citation_count desc"}
_TERM = re.compile(r'-?(?:\w+:)?"[^"]*"|-?(?:\w+:)?\([^)]*\)|\S+')
_OPERATOR = re.compile(r"(?:^|\s)(?:AND|OR|NOT)(?:\s|$)|^\(|\s\(")
_WIDENED = "property:refereed database:astronomy"


async def find_papers(
    query: str,
    max_results: int = 5,
    year_start: int | None = None,
    year_end: int | None = None,
    sort: str = "relevance",
    _transport: httpx.AsyncBaseTransport | None = None,
) -> dict:
    """Search NASA ADS for papers relevant to an event, parameter or method.

    Requires `ADS_API_TOKEN`; without it the tool returns an error rather than
    raising, so the agent can tell the user what is missing.

    Args:
        query: Free-text ADS query — event, parameter, method or author.
        max_results: How many papers to return.
        year_start: Earliest publication year, inclusive. None leaves it open.
        year_end: Latest publication year, inclusive. None leaves it open.
        sort: `relevance`, `citation_count` or `date`, passed to ADS as given.
        _transport: Test seam for injecting an httpx transport. Not part of the
            tool schema the model sees.

    Returns:
        `{"query", "papers", "note"}`, where each paper carries title, authors,
        year, bibcode, doi, citations and abstract. On a missing token or an ADS
        failure, an `error` key instead — never an exception, because the agent
        has to be able to report the cause.

    Example:
        >>> await find_papers("interplanetary shock Rankine-Hugoniot multi-spacecraft",
        ...                   max_results=2)
        {'query': '...', 'papers': [
         {'title': 'Multiple spacecraft observations of interplanetary shocks: ...',
          'authors': 'Russell, C. T. et al.', 'year': '1983',
          'bibcode': '1983JGR....88.9941R', 'doi': '10.1029/JA088iA12p09941',
          'citations': 72, 'abstract': '...'}, ...], 'note': '...'}
    """
    token = settings.literature.ads_token
    if not token:
        return {
            "error": (
                "ADS_API_TOKEN is not set — get a free key at "
                "https://ui.adsabs.harvard.edu/user/settings/token and add it to .env"
            )
        }

    years = f" year:{year_start or ''}-{year_end or ''}" if (year_start or year_end) else ""
    q = f"{query}{years}"
    rows = min(max(max_results, 1), _MAX_ROWS)

    async with httpx.AsyncClient(timeout=15, transport=_transport) as client:
        try:
            resp = await _search(client, token, q, rows, _SORTS.get(sort, _SORTS["relevance"]))
        except httpx.HTTPError as e:
            return {"error": f"ADS request failed: {e}"}
        if resp.status_code != 200:
            return {"error": f"ADS returned HTTP {resp.status_code}: {resp.text[:200]}"}
        docs = resp.json().get("response", {}).get("docs", [])

        widened = _widen(query)
        relaxed_query = f"{widened} {_WIDENED}{years}" if widened else None
        more: list[dict] = []
        if relaxed_query and len(docs) < rows:
            try:
                extra = await _search(client, token, relaxed_query, rows, _SORTS["relevance"])
            except httpx.HTTPError:
                extra = None
            if extra is not None and extra.status_code == 200:
                more = extra.json().get("response", {}).get("docs", [])
            else:
                relaxed_query = None
        else:
            relaxed_query = None

    seen = {d.get("bibcode") for d in docs}
    docs = docs + [d for d in more if d.get("bibcode") not in seen][: rows - len(docs)]
    papers = [_slim(d) for d in docs]
    log.info("find_papers", query=query, n_results=len(papers), relaxed=relaxed_query is not None)
    out = {
        "query": q,
        "papers": papers,
        "note": (
            "Cite as: Authors (year), bibcode. "
            "Full record: https://ui.adsabs.harvard.edu/abs/<bibcode>"
        ),
        "relaxed": relaxed_query is not None,
    }
    if relaxed_query is not None:
        out["relaxed_query"] = relaxed_query
    return out


async def _search(client: httpx.AsyncClient, token: str, q: str, rows: int, sort: str):
    return await client.get(
        _ADS_URL,
        params={"q": q, "fl": _FIELDS, "rows": rows, "sort": sort},
        headers={"Authorization": f"Bearer {token}"},
    )


def _widen(query: str) -> str | None:
    """The same query with its bare words made optional, or None when there is nothing
    to widen.

    ADS requires every bare word of a query, and a model writes eight to twelve of them:
    on a live turn of 2026-09-25, ten of twelve queries about the 2015 St. Patrick's Day
    shock returned 0–3 papers, each extra word ("driver", "in situ", "ACE") removing
    some. Fielded terms (`author:`, `title:`, `abs:`), quoted phrases and negations stay
    required — they are what the query is about — and the bare words become one `OR`
    group, so ADS ranks by how many a paper matches instead of demanding all. A query
    with explicit operators is the model's own structure and is left alone.
    """
    if _OPERATOR.search(query):
        return None
    required, optional = [], []
    for term in _TERM.findall(query):
        bucket = required if (":" in term or '"' in term or term.startswith("-")) else optional
        bucket.append(term)
    if not optional:
        return None
    return " ".join([*required, "(" + " OR ".join(optional) + ")"])


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
