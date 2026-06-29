"""Tests for helioai.tools.speasy_tools — mocked speasy + rag."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

import helioai.tools.rag as rag_module
from helioai.tools.speasy_tools import (
    _data_quality,
    get_timeseries,
    list_missions,
    search_parameters,
)


# ─────────────────────────────── _data_quality ──────────────────────────────


def _times(n, step_s=60):
    return np.datetime64("2024-03-01T00:00:00") + np.arange(n) * np.timedelta64(step_s, "s")


def test_data_quality_clean_data_not_notable() -> None:
    t = _times(100)
    v = np.ones(100)
    q = _data_quality(t, v, np)
    assert q["missing_pct"] == 0.0
    assert q["gaps"] == []
    assert q["outliers_5sigma"] == 0
    assert q["notable"] is False


def test_data_quality_counts_nan_and_fill() -> None:
    t = _times(100)
    v = np.ones(100)
    v[:10] = np.nan
    v[10] = 1e31  # CDF fill value
    q = _data_quality(t, v, np)
    assert q["missing_pct"] == 11.0
    assert q["notable"] is True


def test_data_quality_detects_gap() -> None:
    t = _times(50)
    t[25:] += np.timedelta64(3, "h")  # large hole before sample 25
    v = np.ones(50)
    q = _data_quality(t, v, np)
    assert len(q["gaps"]) == 1
    assert q["gaps"][0]["dur_h"] == pytest.approx(3.0 + 1 / 60, abs=0.05)
    assert q["notable"] is True


def test_data_quality_detects_outlier() -> None:
    t = _times(200)
    v = np.ones(200)
    v[100] = 50.0  # far beyond 5 sigma
    q = _data_quality(t, v, np)
    assert q["outliers_5sigma"] >= 1
    assert q["notable"] is True


def test_data_quality_non_numeric_returns_empty() -> None:
    t = _times(3)
    v = np.array(["a", "b", "c"], dtype=object)
    assert _data_quality(t, v, np) == {}


# ─────────────────────────────── list_missions ──────────────────────────────


async def test_list_missions_returns_dict(monkeypatch) -> None:
    mock_tree = MagicMock()
    type(mock_tree).__iter__ = MagicMock(return_value=iter([]))

    mock_spz = MagicMock()
    mock_spz.inventories.tree = mock_tree

    monkeypatch.setitem(sys.modules, "speasy", mock_spz)
    result = await list_missions()
    assert isinstance(result, dict)
    assert "providers" in result or "error" in result


# ─────────────────────────────── search_parameters ──────────────────────────


async def test_search_parameters_calls_rag_search(monkeypatch) -> None:
    fake_results = [
        {"id": "amda/ace_b_gse", "name": "Bx", "description": "ACE B field", "score": 0.92},
        {"id": "amda/ace_b_y", "name": "By", "description": "ACE B field Y", "score": 0.88},
    ]
    monkeypatch.setattr(rag_module, "search", lambda q, top_k=5, provider=None: fake_results)

    result = await search_parameters("ACE magnetic field")
    assert isinstance(result, dict)
    assert result["results"][0]["id"] == "amda/ace_b_gse"
    assert result["results"][0]["score"] == pytest.approx(0.92)


async def test_search_parameters_returns_query_field(monkeypatch) -> None:
    monkeypatch.setattr(rag_module, "search", lambda q, top_k=5, provider=None: [])
    result = await search_parameters("solar wind density", top_k=3)
    assert result["query"] == "solar wind density"
    assert result["results"] == []


async def test_search_parameters_batch_returns_groups(monkeypatch) -> None:
    def fake_batch(queries, top_k=5, provider=None):
        return [
            [{"id": f"amda/p{i}", "name": q, "description": "", "score": 0.9}]
            for i, q in enumerate(queries)
        ]

    monkeypatch.setattr(rag_module, "search_batch", fake_batch)

    result = await search_parameters(queries=["imf bz ace", "sw density ace"])
    assert "groups" in result
    assert len(result["groups"]) == 2
    assert result["groups"][0]["query"] == "imf bz ace"
    assert result["groups"][0]["results"][0]["id"] == "amda/p0"


async def test_search_parameters_requires_query_or_queries() -> None:
    result = await search_parameters()
    assert "error" in result


# ─────────────────────────────── get_timeseries ─────────────────────────────


def _make_fake_var(n: int = 5) -> MagicMock:
    fake_var = MagicMock()
    times = np.array([f"2005-01-17T12:0{i}:00" for i in range(n)], dtype="datetime64[s]")
    fake_var.time = times
    fake_var.values = np.random.rand(n, 1).astype("float32")
    fake_var.unit = "nT"
    return fake_var


async def test_get_timeseries_returns_preview(monkeypatch) -> None:
    fake_var = _make_fake_var(5)
    mock_spz = MagicMock()
    mock_spz.get_data = MagicMock(return_value=fake_var)
    mock_spz.amda.parameter_range.return_value = None  # skip coverage guard
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await get_timeseries("amda/ace_b_gse", "2005-01-17T12:00:00", "2005-01-17T12:10:00")
    assert "error" not in result
    assert "n_points" in result
    assert "preview" in result


async def test_get_timeseries_no_data_returns_error(monkeypatch) -> None:
    mock_spz = MagicMock()
    mock_spz.get_data = MagicMock(return_value=None)
    mock_spz.amda.parameter_range.return_value = None  # skip coverage guard
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await get_timeseries("amda/fake", "2024-01-01T00:00:00", "2024-01-01T01:00:00")
    assert "error" in result


async def test_get_timeseries_exception_returns_error(monkeypatch) -> None:
    mock_spz = MagicMock()
    mock_spz.get_data = MagicMock(side_effect=RuntimeError("network error"))
    mock_spz.amda.parameter_range.return_value = None  # skip coverage guard
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await get_timeseries("amda/fake", "2024-01-01T00:00:00", "2024-01-01T01:00:00")
    assert "error" in result


async def test_get_timeseries_parameter_range_guard(monkeypatch) -> None:
    mock_spz = MagicMock()
    mock_rng = MagicMock()
    mock_rng.start = "2005-01-01T00:00:00"
    mock_rng.stop = "2005-12-31T23:59:59"
    mock_spz.amda.parameter_range.return_value = mock_rng
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    # request entirely outside the available range → warning, no download attempted
    result = await get_timeseries("amda/imf", "2020-01-01T00:00:00", "2020-01-02T00:00:00")
    assert "warning" in result
    assert "suggestion" in result
    assert "available_start" in result
    mock_spz.get_data.assert_not_called()
