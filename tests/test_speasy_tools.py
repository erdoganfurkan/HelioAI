"""Tests for helioai.tools.speasy_tools — fake speasy + rag."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from support.fake_speasy import FakeRange, FakeSpeasy, FakeVariable

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
    fake_spz = FakeSpeasy(tree=object())

    monkeypatch.setitem(sys.modules, "speasy", fake_spz)
    result = await list_missions()
    assert isinstance(result, dict)
    assert "providers" in result or "error" in result


# ─────────────────────────────── search_parameters ──────────────────────────


async def test_search_parameters_calls_rag_search(monkeypatch) -> None:
    fake_results = [
        {"id": "amda/ace_b_gse", "name": "Bx", "description": "ACE B field", "score": 0.92},
        {"id": "amda/ace_b_y", "name": "By", "description": "ACE B field Y", "score": 0.88},
    ]
    monkeypatch.setattr(
        rag_module, "search", lambda q, top_k=5, provider=None, window=None: fake_results
    )

    result = await search_parameters("ACE magnetic field")
    assert isinstance(result, dict)
    assert result["results"][0]["id"] == "amda/ace_b_gse"
    assert all("score" not in r for r in result["results"]), (
        "score is written before two later reorderings and contradicts the rank it "
        "sits on — the tool boundary is where it stops"
    )
    assert fake_results[0]["score"] == pytest.approx(0.92), (
        "rag.search keeps its score; stripping must not mutate the cached dicts"
    )


async def test_search_parameters_returns_query_field(monkeypatch) -> None:
    monkeypatch.setattr(rag_module, "search", lambda q, top_k=5, provider=None, window=None: [])
    result = await search_parameters("solar wind density", top_k=3)
    assert result["query"] == "solar wind density"
    assert result["results"] == []


async def test_search_parameters_batch_returns_groups(monkeypatch) -> None:
    def fake_batch(queries, top_k=5, provider=None, window=None):
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
    assert all("score" not in r for g in result["groups"] for r in g["results"])


async def test_the_download_window_reaches_the_ranking(monkeypatch) -> None:
    """A product that cannot cover the interval used to be flagged inside the top-k, after
    the cut; the ranking itself now knows the window, so it is demoted before the cut."""
    seen: dict = {}

    def fake_search(q, top_k=5, provider=None, window=None):
        seen["window"] = window
        return []

    monkeypatch.setattr(rag_module, "search", fake_search)
    await search_parameters("Wind MFI", start="2015-03-17T00:00:00", stop="2015-03-18T00:00:00")
    assert seen["window"] == ("2015-03-17T00:00:00", "2015-03-18T00:00:00")
    await search_parameters("Wind MFI", start="2015-03-17T00:00:00")
    assert seen["window"] is None, "both ends are needed for the filter to apply"


async def test_search_parameters_requires_query_or_queries() -> None:
    result = await search_parameters()
    assert "error" in result


async def test_fallback_note_names_what_actually_failed(monkeypatch) -> None:
    """The installed 0.3.0 candidate had an index and a torch too old for transformers:
    the dense search raised at import and the note said "RAG index not built". A client
    model with none of our context would have told the user to build an index that
    existed. The exception is the diagnosis, so the note carries it."""

    def boom(q, top_k=5, provider=None, window=None):
        raise NameError("name 'nn' is not defined")

    monkeypatch.setattr(rag_module, "search", boom)
    fake_spz = FakeSpeasy(tree=None)
    monkeypatch.setattr(
        "helioai.tools.speasy_tools._fallback_search", lambda spz, q, k: [{"id": "x"}]
    )
    monkeypatch.setitem(__import__("sys").modules, "speasy", fake_spz)

    result = await search_parameters("Wind MFI GSM")

    assert result["results"] == [{"id": "x"}]
    assert "NameError: name 'nn' is not defined" in result["note"]
    assert "index not built" not in result["note"]


async def test_batch_fallback_note_names_what_actually_failed(monkeypatch) -> None:
    def boom(queries, top_k=5, provider=None, window=None):
        raise RuntimeError("chroma unreachable")

    monkeypatch.setattr(rag_module, "search_batch", boom)
    monkeypatch.setattr("helioai.tools.speasy_tools._fallback_search", lambda spz, q, k: [])
    monkeypatch.setitem(__import__("sys").modules, "speasy", FakeSpeasy())

    result = await search_parameters(queries=["a", "b"])

    assert len(result["groups"]) == 2
    assert "RuntimeError: chroma unreachable" in result["note"]


# ─────────────────────────────── get_timeseries ─────────────────────────────


def _make_fake_var(n: int = 5) -> FakeVariable:
    times = np.array([f"2005-01-17T12:0{i}:00" for i in range(n)], dtype="datetime64[s]")
    return FakeVariable(time=times, values=np.random.rand(n, 1).astype("float32"), unit="nT")


async def test_get_timeseries_returns_preview(monkeypatch) -> None:
    fake_var = _make_fake_var(5)
    fake_spz = FakeSpeasy(get_data=fake_var)
    monkeypatch.setitem(sys.modules, "speasy", fake_spz)

    result = await get_timeseries("amda/ace_b_gse", "2005-01-17T12:00:00", "2005-01-17T12:10:00")
    assert "error" not in result
    assert "n_points" in result
    assert "preview" in result


async def test_get_timeseries_no_data_returns_error(monkeypatch) -> None:
    fake_spz = FakeSpeasy(get_data=None)
    monkeypatch.setitem(sys.modules, "speasy", fake_spz)

    result = await get_timeseries("amda/fake", "2024-01-01T00:00:00", "2024-01-01T01:00:00")
    assert "error" in result


async def test_get_timeseries_exception_returns_error(monkeypatch) -> None:
    fake_spz = FakeSpeasy(get_data_error=RuntimeError("network error"))
    monkeypatch.setitem(sys.modules, "speasy", fake_spz)

    result = await get_timeseries("amda/fake", "2024-01-01T00:00:00", "2024-01-01T01:00:00")
    assert "error" in result


def _range(start_iso: str, stop_iso: str) -> FakeRange:
    return FakeRange(start_iso, stop_iso)


async def test_get_timeseries_refuses_a_window_with_no_overlap(monkeypatch) -> None:
    fake_spz = FakeSpeasy(ranges={"amda": _range("2005-01-01T00:00:00", "2005-12-31T23:59:59")})
    monkeypatch.setitem(sys.modules, "speasy", fake_spz)

    result = await get_timeseries("amda/imf", "2020-01-01T00:00:00", "2020-01-02T00:00:00")

    assert "error" in result
    assert "2005-01-01" in result["error"], "the error must name the coverage"
    assert result["available_start"].startswith("2005-01-01")
    assert fake_spz.get_data_calls == []


async def test_get_timeseries_downloads_a_partial_overlap(monkeypatch) -> None:
    """A window overhanging the coverage must still fetch what exists.

    The guard used to refuse whenever the window crossed either edge, which throws
    away real data — for a parameter ending last month, an ordinary "the last few
    weeks" request returned nothing at all.
    """
    fake_spz = FakeSpeasy(
        get_data=_make_fake_var(5),
        ranges={"amda": _range("2005-01-01T00:00:00", "2005-12-31T23:59:59")},
    )
    monkeypatch.setitem(sys.modules, "speasy", fake_spz)

    result = await get_timeseries("amda/imf", "2005-12-30T00:00:00", "2006-01-05T00:00:00")

    assert "error" not in result
    assert len(fake_spz.get_data_calls) == 1
    assert "coverage_note" in result
    assert "clipped" in result["coverage_note"]
    assert result["available_start"] == "2005-01-01T00:00:00"
    assert result["available_stop"] == "2005-12-31T23:59:59", "bounds as keys, not only prose"


async def test_get_timeseries_says_what_it_obtained_beside_what_was_asked(monkeypatch) -> None:
    """The series' own first and last timestamps, so a window clipped to the archive or to
    a gap is visible as two dates, not inferred from a sentence."""
    fake_spz = FakeSpeasy(get_data=_make_fake_var(5))
    monkeypatch.setitem(sys.modules, "speasy", fake_spz)

    result = await get_timeseries("amda/imf", "2005-01-17T00:00:00", "2005-01-18T00:00:00")

    assert result["start"] == "2005-01-17T00:00:00" and result["stop"] == "2005-01-18T00:00:00"
    assert result["obtained_start"] == "2005-01-17T12:00:00"
    assert result["obtained_stop"] == "2005-01-17T12:04:00"
    keys = list(result)
    assert keys.index("obtained_start") == keys.index("stop") + 1, "beside what was asked"


async def test_get_timeseries_is_silent_when_fully_covered(monkeypatch) -> None:
    fake_spz = FakeSpeasy(
        get_data=_make_fake_var(5),
        ranges={"amda": _range("2005-01-01T00:00:00", "2005-12-31T23:59:59")},
    )
    monkeypatch.setitem(sys.modules, "speasy", fake_spz)

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

    fake_spz = FakeSpeasy(
        get_data=FakeVariable(time=times, values=values, unit="cm^-3", columns=["Np"], name="Np")
    )
    monkeypatch.setitem(sys.modules, "speasy", fake_spz)

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

    fake_spz = FakeSpeasy(
        get_data=FakeVariable(
            time=times, values=values, unit="km/s", columns=["V"], meta={"FILLVAL": fill}, name="V"
        )
    )
    monkeypatch.setitem(sys.modules, "speasy", fake_spz)
    monkeypatch.setattr(ds, "_session_data_dir", lambda data_dir=None: tmp_path)

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
    fake_spz = FakeSpeasy(get_data_error=IndexError("tuple index out of range"))
    monkeypatch.setitem(sys.modules, "speasy", fake_spz)
    result = await get_timeseries("cda/AC_OR_SSC/Epoch", "2015-03-17", "2015-03-18")
    assert "IndexError" in result["error"]
    assert "not a plottable data variable" in result["error"]


@pytest.mark.asyncio
async def test_preview_survives_non_numeric_values(monkeypatch):
    """datetime64 values used to raise inside the preview loop, killing the call."""
    times = np.array(["2015-03-17T00:00:00", "2015-03-17T00:01:00"], dtype="datetime64[s]")
    var = FakeVariable(time=times, values=times.copy(), name="Epoch")
    fake_spz = FakeSpeasy(get_data=var)
    monkeypatch.setitem(sys.modules, "speasy", fake_spz)
    result = await get_timeseries("cda/AC_OR_SSC/Epoch", "2015-03-17", "2015-03-18")
    assert "error" not in result or "preview" in result


@pytest.mark.asyncio
async def test_repeat_download_short_circuits_before_the_network(tmp_path, monkeypatch):
    """A second request for the same param+window must not hit speasy at all."""
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
        monkeypatch.setitem(sys.modules, "speasy", FakeSpeasy(get_data=counting_get_data))
        result = await get_timeseries(
            "cda/WI_H0_MFI/B3GSM", "2015-03-16T18:00:00", "2015-03-18T12:00:00"
        )
        assert result["dataset"] == "b3gsm"
        assert result["already_downloaded"] is True
        assert calls["n"] == 0
    finally:
        ws.reset_label(tok)


def test_cadence_measures_samples_not_the_file_grid():
    """Regression: `WI_PM_3DP` reported "8 ms" for protons sampled every 3 s.

    The ISTP file carries an 8 ms epoch grid and pads it with fill rows, so a median
    taken over the whole grid describes the file, not the instrument — off by 385x.
    Cadence is what the agent picks a product on, and a run briefed "a few seconds,
    not the sub-second product" rejects the right product on that label alone.
    """
    import numpy as np

    from helioai.tools.speasy_tools import _sample_cadence

    # 8 ms grid, one real sample every 3 s (every 375th row), the rest fill-blanked.
    times = np.datetime64("2015-03-17T00:00:00") + np.arange(
        0, 375 * 20, dtype="int64"
    ) * np.timedelta64(8, "ms")
    values = np.full(len(times), np.nan)
    values[::375] = 42.0

    cadence, n_valid = _sample_cadence(times, values)
    assert cadence == "3 s", cadence
    assert n_valid == 20


def test_cadence_of_a_gapless_product_is_unchanged():
    """The fix must not move a product whose grid already is its sampling."""
    import numpy as np

    from helioai.tools.speasy_tools import _sample_cadence

    times = np.datetime64("2015-03-17T00:00:00") + np.arange(100, dtype="int64") * np.timedelta64(
        16, "s"
    )
    cadence, n_valid = _sample_cadence(times, np.arange(100.0))
    assert cadence == "16 s"
    assert n_valid == 100


def test_cadence_survives_a_non_numeric_variable():
    """A label/time variable has nothing to test for finiteness; the grid is all there is."""
    import numpy as np

    from helioai.tools.speasy_tools import _sample_cadence

    times = np.datetime64("2015-03-17T00:00:00") + np.arange(5, dtype="int64") * np.timedelta64(
        1, "m"
    )
    cadence, n_valid = _sample_cadence(times, np.array(["a", "b", "c", "d", "e"]))
    assert cadence == "1 min"
    assert n_valid == 5


# ── the data tools run their library calls off the event loop ─────────────────


async def test_a_slow_download_does_not_freeze_the_event_loop(monkeypatch):
    """`get_timeseries` is `async def` but speasy is synchronous: called inline, a
    download froze the loop — and every other user's stream on the web server — for
    its whole duration. A ticker coroutine measures the longest gap between two of its
    own ticks while a fake download sleeps half a second."""
    import asyncio
    import sys
    import time

    from helioai.tools.speasy_tools import get_timeseries

    def slow_get_data(*a, **k):
        time.sleep(0.5)
        raise RuntimeError("no data, on purpose")

    fake_spz = FakeSpeasy(get_data=slow_get_data)
    monkeypatch.setitem(sys.modules, "speasy", fake_spz)
    monkeypatch.setattr(
        "helioai.tools.speasy_tools._coverage_check", lambda *a, **k: (None, None, None)
    )

    gaps: list[float] = []

    async def ticker(stop: asyncio.Event):
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(0.01)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    stop = asyncio.Event()
    tick = asyncio.create_task(ticker(stop))
    result = await get_timeseries("amda/imf", "2015-03-17T00:00:00", "2015-03-17T01:00:00")
    stop.set()
    await tick

    assert "error" in result
    assert max(gaps) < 0.25, f"the loop stalled for {max(gaps):.2f}s during the download"


async def test_the_worker_thread_sees_the_callers_session(monkeypatch, tmp_path):
    """`to_thread` copies the context, so the download lands in the caller's session
    directory. This is the property a raw executor would silently lose."""
    import helioai.workspace as ws
    from helioai.tools.offload import run_blocking

    token = ws.set_label("ticker-session")
    try:
        expected = ws.get_session_dir()
        seen = await run_blocking(ws.get_session_dir)
    finally:
        ws.reset_label(token)
    assert seen == expected
