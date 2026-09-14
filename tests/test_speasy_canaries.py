"""Offline canaries for the real speasy attributes mirrored by test fakes."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib import import_module

import numpy as np
from support.fake_speasy import (
    FakeCatalog,
    FakeCatalogIndex,
    FakeEvent,
    FakeParameterIndex,
    FakeRange,
    FakeVariable,
)


def _assert_attributes(real, fake) -> None:
    missing = [attr for attr in fake.exposed_attributes if not hasattr(real, attr)]
    assert missing == []


def test_speasy_variable_exposes_fake_variable_attributes() -> None:
    """Guards the data-tool fakes against SpeasyVariable attribute drift."""
    containers = import_module("speasy.core.data_containers")
    variable_mod = import_module("speasy.products.variable")
    times = np.array(["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64[ns]")
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    real = variable_mod.SpeasyVariable(
        axes=[containers.VariableTimeAxis(times)],
        values=containers.DataContainer(values, meta={"UNITS": "nT", "FILLVAL": -1e31}, name="B"),
        columns=["Bx", "By"],
    )
    fake = FakeVariable(time=times, values=values, unit="nT", columns=["Bx", "By"], name="B")

    _assert_attributes(real, fake)


def test_speasy_event_exposes_fake_event_attributes() -> None:
    """Guards catalog event tests against Event attribute drift."""
    catalog_mod = import_module("speasy.products.catalog")
    real = catalog_mod.Event(
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
        meta={"kind": "shock"},
    )
    fake = FakeEvent("2024-01-01T00:00:00", "2024-01-02T00:00:00")

    _assert_attributes(real, fake)


def test_speasy_catalog_exposes_fake_catalog_attributes() -> None:
    """Guards iterable catalog tests against Catalog attribute drift."""
    catalog_mod = import_module("speasy.products.catalog")
    real = catalog_mod.Catalog("events", meta={}, events=[])
    fake = FakeCatalog(name="events")

    _assert_attributes(real, fake)


def test_speasy_catalog_index_exposes_fake_catalog_index_attributes() -> None:
    """Guards AMDA catalog listing tests against CatalogIndex attribute drift."""
    indexes = import_module("speasy.core.inventory.indexes")
    meta = {
        "desc": "catalog",
        "nbIntervals": 2,
        "surveyStart": "2024-01-01",
        "surveyStop": "2024-01-02",
    }
    real = indexes.CatalogIndex("Catalog", "amda", "sharedcatalog_1", meta=meta)
    fake = FakeCatalogIndex("sharedcatalog_1", "Catalog", "catalog", 2, "2024-01-01", "2024-01-02")

    _assert_attributes(real, fake)


def test_speasy_timetable_index_exposes_fake_catalog_index_attributes() -> None:
    """Guards AMDA timetable listing tests against TimetableIndex attribute drift."""
    indexes = import_module("speasy.core.inventory.indexes")
    meta = {
        "desc": "timetable",
        "nbIntervals": 2,
        "surveyStart": "2024-01-01",
        "surveyStop": "2024-01-02",
    }
    real = indexes.TimetableIndex("Timetable", "amda", "sharedtimetable_1", meta=meta)
    fake = FakeCatalogIndex(
        "sharedtimetable_1",
        "Timetable",
        "timetable",
        2,
        "2024-01-01",
        "2024-01-02",
        "TimetableIndex",
    )

    _assert_attributes(real, fake)


def test_speasy_parameter_index_exposes_fake_parameter_index_attributes() -> None:
    """Guards fallback search tests against ParameterIndex attribute drift."""
    indexes = import_module("speasy.core.inventory.indexes")
    real = indexes.ParameterIndex("B", "amda", "amda/b", meta={"desc": "field"})
    fake = FakeParameterIndex("amda/b", "B", "field")

    _assert_attributes(real, fake)


def test_speasy_datetime_range_exposes_fake_range_attributes() -> None:
    """Guards coverage-window tests against DateTimeRange attribute drift."""
    range_mod = import_module("speasy.core.datetime_range")
    real = range_mod.DateTimeRange(
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
    )
    fake = FakeRange("2024-01-01T00:00:00", "2024-01-02T00:00:00")

    _assert_attributes(real, fake)
