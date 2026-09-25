"""Tests for the find_papers NASA ADS tool."""

from __future__ import annotations

import json

import httpx
import pytest

from helioai.config import settings
from helioai.tools import literature

_DOC = {
    "title": ["Electron-scale measurements of magnetic reconnection in space"],
    "author": ["Burch, J. L.", "Torbert, R. B.", "Phan, T. D."],
    "year": "2016",
    "bibcode": "2016Sci...352.2939B",
    "doi": ["10.1126/science.aaf2939"],
    "citation_count": 1500,
    "abstract": "x" * 900,
}


@pytest.fixture
def ads_token(monkeypatch):
    monkeypatch.setattr(settings.literature, "ads_token", "test-token")


def _transport(payload: dict | None = None, status: int = 200, capture: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture["url"] = str(request.url)
            capture["auth"] = request.headers.get("authorization", "")
        body = payload if payload is not None else {"response": {"docs": [_DOC]}}
        return httpx.Response(status, content=json.dumps(body))

    return httpx.MockTransport(handler)


async def test_missing_token_returns_error(monkeypatch):
    monkeypatch.setattr(settings.literature, "ads_token", "")
    out = await literature.find_papers("shock")
    assert "ADS_API_TOKEN" in out["error"]


async def test_papers_mapping(ads_token):
    capture: dict = {}
    out = await literature.find_papers("reconnection", _transport=_transport(capture=capture))
    paper = out["papers"][0]
    assert paper["title"].startswith("Electron-scale")
    assert paper["authors"] == "Burch, J. L. et al."
    assert paper["bibcode"] == "2016Sci...352.2939B"
    assert paper["doi"] == "10.1126/science.aaf2939"
    assert len(paper["abstract"]) == 300
    assert capture["auth"] == "Bearer test-token"


async def test_rows_capped_at_10(ads_token):
    capture: dict = {}
    await literature.find_papers("shock", max_results=50, _transport=_transport(capture=capture))
    assert "rows=10" in capture["url"]


async def test_year_filter_in_query(ads_token):
    capture: dict = {}
    out = await literature.find_papers(
        "shock", year_start=2015, year_end=2020, _transport=_transport(capture=capture)
    )
    assert "year:2015-2020" in out["query"]


async def test_http_error_status(ads_token):
    out = await literature.find_papers("shock", _transport=_transport(payload={}, status=401))
    assert "401" in out["error"]


async def test_network_error(ads_token):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    out = await literature.find_papers("shock", _transport=httpx.MockTransport(handler))
    assert "ADS request failed" in out["error"]


async def test_single_author_no_et_al(ads_token):
    doc = dict(_DOC, author=["Shue, J.-H."])
    payload = {"response": {"docs": [doc]}}
    out = await literature.find_papers("magnetopause", _transport=_transport(payload=payload))
    assert out["papers"][0]["authors"] == "Shue, J.-H."


def test_registered_in_registry():
    import helioai.tools.setup  # noqa: F401
    from helioai.tools.registry import registry

    assert "find_papers" in registry


# ── a query too narrow for ADS's implicit AND is widened once ───────────────────────


def _doc(bibcode: str) -> dict:
    return dict(_DOC, bibcode=bibcode, title=[f"Paper {bibcode}"])


def _two_step(
    first: list[dict], second: list[dict], *, status2: int = 200, log: list | None = None
):
    calls: list[dict] = [] if log is None else log

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        if len(calls) == 1:
            return httpx.Response(200, content=json.dumps({"response": {"docs": first}}))
        return httpx.Response(status2, content=json.dumps({"response": {"docs": second}}))

    return httpx.MockTransport(handler)


async def test_a_query_that_starves_is_widened_to_refereed_papers_matching_most_words(ads_token):
    """Live web run, 2026-09-25: ten of the twelve find_papers queries of one turn came
    back with 0–3 papers — ADS requires every bare word, and the model writes eight to
    twelve ("... solar wind driver Dst -223"). Replayed against ADS, the same queries
    widened this way returned five refereed papers each, the two the answer cited among
    them. The exact hits come first; the widened ones fill the rest, ranked by relevance
    — by citations, "any of these words" is headed by Deep learning."""
    calls: list[dict] = []
    transport = _two_step([_doc("A")], [_doc("A"), _doc("B"), _doc("C")], log=calls)

    out = await literature.find_papers(
        'abs:"St. Patrick" 2015 storm sheath -223',
        max_results=5,
        sort="citations",
        year_start=2015,
        _transport=transport,
    )

    assert [p["bibcode"] for p in out["papers"]] == ["A", "B", "C"]
    assert out["relaxed"] is True
    widened = calls[1]["q"]
    assert out["relaxed_query"] == widened
    assert widened.startswith('abs:"St. Patrick" -223 (2015 OR storm OR sheath)')
    assert "property:refereed" in widened and "database:astronomy" in widened
    assert "year:2015-" in widened
    assert calls[1]["sort"] == "score desc"
    assert calls[0]["q"] == 'abs:"St. Patrick" 2015 storm sheath -223 year:2015-'


async def test_a_query_that_returns_enough_is_sent_once_and_left_as_it_is(ads_token):
    calls: list[dict] = []
    docs = [_doc(str(i)) for i in range(5)]
    out = await literature.find_papers(
        "interplanetary shock", max_results=5, _transport=_two_step(docs, [], log=calls)
    )
    assert len(calls) == 1
    assert out["relaxed"] is False and "relaxed_query" not in out


@pytest.mark.parametrize(
    "query", ["shock AND (Wind OR ACE)", "shock NOT magnetopause", '"interplanetary shock"']
)
async def test_a_query_with_nothing_to_relax_is_not_rewritten(ads_token, query):
    """Explicit operators are the model's own structure, and a query of required terms
    only has no optional word to widen: both are sent once, as written."""
    calls: list[dict] = []
    out = await literature.find_papers(query, _transport=_two_step([], [_doc("X")], log=calls))
    assert len(calls) == 1
    assert out["papers"] == [] and out["relaxed"] is False


async def test_a_failed_widening_keeps_the_exact_hits(ads_token):
    out = await literature.find_papers(
        "shock Wind 2015 driver", _transport=_two_step([_doc("A")], [], status2=503)
    )
    assert [p["bibcode"] for p in out["papers"]] == ["A"]
    assert "error" not in out and out["relaxed"] is False
