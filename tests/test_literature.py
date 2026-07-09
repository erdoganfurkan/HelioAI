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
