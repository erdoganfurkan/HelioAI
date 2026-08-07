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


def _range(start_iso: str, stop_iso: str):
    """A stand-in for speasy's DateTimeRange, with ITS attribute names.

    This matters more than it looks: the previous version of this test set
    `.start`/`.stop` on a MagicMock, which the real class does not have. The guard
    read those, raised AttributeError into a bare `except`, and was a silent no-op
    in production while this test passed — a mock that agrees with the code instead
    of with the library proves nothing.
    """
    from datetime import UTC, datetime

    class _R:
        start_time = datetime.fromisoformat(start_iso).replace(tzinfo=UTC)
        stop_time = datetime.fromisoformat(stop_iso).replace(tzinfo=UTC)

    return _R()


def test_speasy_daterange_still_uses_start_time(monkeypatch) -> None:
    """Guards against the library renaming what the guard reads.

    If speasy ever returns a range whose attributes are not start_time/stop_time,
    the coverage guard degrades to a no-op again — this fails loudly instead.
    """
    import speasy as spz

    rng = spz.cda.parameter_range("AC_H0_MFI/BGSM")
    assert hasattr(rng, "start_time") and hasattr(rng, "stop_time")


async def test_get_timeseries_refuses_a_window_with_no_overlap(monkeypatch) -> None:
    mock_spz = MagicMock()
    mock_spz.amda.parameter_range.return_value = _range(
        "2005-01-01T00:00:00", "2005-12-31T23:59:59"
    )
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await get_timeseries("amda/imf", "2020-01-01T00:00:00", "2020-01-02T00:00:00")

    assert "error" in result
    assert "2005-01-01" in result["error"], "the error must name the coverage"
    assert result["available_start"].startswith("2005-01-01")
    mock_spz.get_data.assert_not_called()


async def test_get_timeseries_downloads_a_partial_overlap(monkeypatch) -> None:
    """A window overhanging the coverage must still fetch what exists.

    The guard used to refuse whenever the window crossed either edge, which throws
    away real data — for a parameter ending last month, an ordinary "the last few
    weeks" request returned nothing at all.
    """
    mock_spz = MagicMock()
    mock_spz.amda.parameter_range.return_value = _range(
        "2005-01-01T00:00:00", "2005-12-31T23:59:59"
    )
    mock_spz.get_data = MagicMock(return_value=_make_fake_var(5))
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await get_timeseries("amda/imf", "2005-12-30T00:00:00", "2006-01-05T00:00:00")

    assert "error" not in result
    mock_spz.get_data.assert_called_once()
    assert "coverage_note" in result
    assert "clipped" in result["coverage_note"]


async def test_get_timeseries_is_silent_when_fully_covered(monkeypatch) -> None:
    mock_spz = MagicMock()
    mock_spz.amda.parameter_range.return_value = _range(
        "2005-01-01T00:00:00", "2005-12-31T23:59:59"
    )
    mock_spz.get_data = MagicMock(return_value=_make_fake_var(5))
    monkeypatch.setitem(sys.modules, "speasy", mock_spz)

    result = await get_timeseries("amda/imf", "2005-06-01T00:00:00", "2005-06-02T00:00:00")

    assert "coverage_note" not in result
    assert "error" not in result


@pytest.mark.asyncio
async def test_get_timeseries_rejects_an_all_fill_series(monkeypatch, tmp_path):
    """100% CDF fill is as empty as zero rows — reporting success gets it plotted.

    Regression: ACE/SWEPAM during the 2003 Halloween storm returns 1913 rows of
    -1e31, which used to come back as a successful download with a quality note.
    """
    import numpy as np

    import helioai.tools.speasy_tools as st

    n = 40
    times = np.arange("2003-10-29T00:00", n, dtype="datetime64[m]")
    values = np.full(n, -9.9999998e30)

    class _Var:
        unit = "cm^-3"
        columns = ["Np"]
        meta: dict = {}
        name = "Np"

        def __init__(self):
            self.time = times
            self.values = values

        def __len__(self):
            return n

    import speasy

    monkeypatch.setattr(speasy, "get_data", lambda *a, **k: _Var(), raising=False)

    res = await st.get_timeseries("cda/AC_H0_SWE/Np", "2003-10-29", "2003-10-29T00:40")

    assert "error" in res
    assert res["missing_pct"] == 100.0
    assert res["n_points"] == n
    assert "dataset" not in res, "garbage must not be persisted"


def test_data_quality_honours_the_declared_fillval():
    """Not every mission fills with ~1e31 — Wind/SWE uses 99999.9.

    That value passed the magnitude test and reached the agent as a plausible
    solar-wind speed. Only the declared FILLVAL can catch it; a blanket
    "reject >= 99999" rule would throw away real data such as an OMNI proton
    temperature of 99093 K.
    """
    import numpy as np

    from helioai.tools.speasy_tools import _data_quality

    # The sentinel as Wind actually stores it: float32, hence not exactly 99999.9.
    fill = np.float32(99999.9)
    times = np.arange("2003-10-29T00:00", 10, dtype="datetime64[m]")
    values = np.array([400.0, 420.0, fill, fill, 430.0, 440.0, 450.0, 460.0, 470.0, 480.0])

    without = _data_quality(times, values, np)
    assert without["missing_pct"] == 0.0, "documents the old blind spot"

    with_fv = _data_quality(times, values, np, fill)
    assert with_fv["missing_pct"] == 20.0
    assert with_fv["notable"] is True


def test_data_quality_keeps_real_values_near_the_sentinel():
    """An OMNI proton temperature of 99093 K is real data, not a fill value."""
    import numpy as np

    from helioai.tools.speasy_tools import _data_quality

    times = np.arange("2003-10-29T00:00", 4, dtype="datetime64[m]")
    values = np.array([77695.0, 99093.0, 88000.0, 91000.0])

    q = _data_quality(times, values, np, 99999.8984375)

    assert q["missing_pct"] == 0.0


@pytest.mark.asyncio
async def test_get_timeseries_persists_nan_not_the_sentinel(monkeypatch, tmp_path):
    """The whole point: what reaches the sandbox must already be NaN.

    Wind/SWE fills with 99999.9. Persisting that raw meant a plot with a spike to
    99999.9 km/s and a mean wrecked by it, because the reader has to remember to
    clean — and clean() cannot see FILLVAL anyway.
    """
    import numpy as np

    import helioai.datastore as ds
    import helioai.tools.speasy_tools as st

    fill = np.float32(99999.9)
    values = np.array([400.0, fill, 420.0, fill, 440.0])
    times = np.arange("2015-03-17T00:00", 5, dtype="datetime64[m]")

    class _Var:
        unit = "km/s"
        columns = ["V"]
        meta = {"FILLVAL": fill}
        name = "V"

        def __init__(self):
            self.time = times
            self.values = values

        def __len__(self):
            return len(times)

    import speasy

    monkeypatch.setattr(speasy, "get_data", lambda *a, **k: _Var(), raising=False)
    monkeypatch.setattr(ds, "_session_data_dir", lambda: tmp_path)

    res = await st.get_timeseries("cda/WI_H1_SWE/V", "2015-03-17", "2015-03-17T00:05")

    assert "error" not in res
    assert res["quality"]["missing_pct"] == 40.0
    assert "99999" not in res["preview"], "the agent must not be shown the sentinel"

    stored = np.load(tmp_path / f"{res['dataset']}.npz")["values"]
    assert np.isnan(stored).sum() == 2
    assert not (stored > 99000).any()
    assert np.nanmax(stored) == 440.0


@pytest.mark.asyncio
async def test_retrieval_failure_names_the_exception_type(monkeypatch):
    """A bare "tuple index out of range" gave the agent nothing to act on."""
    import speasy

    def boom(*a, **k):
        raise IndexError("tuple index out of range")

    monkeypatch.setattr(speasy, "get_data", boom)
    result = await get_timeseries("cda/AC_OR_SSC/Epoch", "2015-03-17", "2015-03-18")
    assert "IndexError" in result["error"]
    assert "not a plottable data variable" in result["error"]


@pytest.mark.asyncio
async def test_preview_survives_non_numeric_values(monkeypatch):
    """datetime64 values used to raise inside the preview loop, killing the call."""
    from types import SimpleNamespace

    import speasy

    times = np.array(["2015-03-17T00:00:00", "2015-03-17T00:01:00"], dtype="datetime64[s]")
    var = SimpleNamespace(
        time=times, values=times.copy(), unit="", name="Epoch", columns=[], meta={}
    )
    monkeypatch.setattr(speasy, "get_data", lambda *a, **k: var)
    result = await get_timeseries("cda/AC_OR_SSC/Epoch", "2015-03-17", "2015-03-18")
    assert "error" not in result or "preview" in result


@pytest.mark.asyncio
async def test_repeat_download_short_circuits_before_the_network(tmp_path, monkeypatch):
    """A second request for the same param+window must not hit speasy at all."""
    import speasy

    import helioai.workspace as ws
    from helioai.datastore import save_timeseries

    monkeypatch.setattr(ws, "_root", lambda: tmp_path)
    tok = ws.set_label("sess")
    calls = {"n": 0}

    def counting_get_data(*a, **k):
        calls["n"] += 1
        raise AssertionError("speasy must not be reached for an already-held dataset")

    try:
        save_timeseries(
            "b3gsm",
            time=np.array(["2015-03-16T18:00:00"], dtype="datetime64[s]"),
            values=np.array([[1.0, 2.0, 3.0]]),
            param_id="cda/WI_H0_MFI/B3GSM",
            units="nT",
            start="2015-03-16T18:00:00",
            stop="2015-03-18T12:00:00",
            columns=["Bx", "By", "Bz"],
            source="get_timeseries",
        )
        monkeypatch.setattr(speasy, "get_data", counting_get_data)
        result = await get_timeseries(
            "cda/WI_H0_MFI/B3GSM", "2015-03-16T18:00:00", "2015-03-18T12:00:00"
        )
        assert result["dataset"] == "b3gsm"
        assert result["already_downloaded"] is True
        assert calls["n"] == 0
    finally:
        ws.reset_label(tok)
