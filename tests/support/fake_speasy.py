"""Duck-typed speasy doubles anchored to the attributes HelioAI reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import numpy as np


def _dt(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


@dataclass
class FakeRange:
    """Keep coverage tests on speasy's start_time/stop_time contract."""

    start_time: datetime | str
    stop_time: datetime | str

    exposed_attributes = ("start_time", "stop_time")

    def __post_init__(self) -> None:
        self.start_time = _dt(self.start_time)
        self.stop_time = _dt(self.stop_time)


@dataclass
class FakeEvent:
    """Keep catalog tests on speasy Event rather than ad-hoc start/stop mocks."""

    start_time: datetime | str
    stop_time: datetime | str
    meta: dict[str, Any] = field(default_factory=dict)

    exposed_attributes = ("start_time", "stop_time", "meta")


@dataclass
class FakeCatalog:
    """Model the iterable speasy Catalog surface used by catalog tools."""

    name: str = "catalog"
    meta: dict[str, Any] = field(default_factory=dict)
    events: list[FakeEvent] = field(default_factory=list)

    exposed_attributes = ("name", "meta")

    def __iter__(self):
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)


@dataclass
class FakeCatalogIndex:
    """Carry speasy inventory names exactly so tests fail on attribute drift."""

    __spz_uid__: str
    __spz_name__: str
    desc: str = ""
    nbIntervals: int = 0
    surveyStart: str = ""
    surveyStop: str = ""
    __spz_type__: str = "CatalogIndex"

    exposed_attributes = (
        "__spz_uid__",
        "__spz_name__",
        "__spz_type__",
        "desc",
        "nbIntervals",
        "surveyStart",
        "surveyStop",
    )


@dataclass
class FakeParameterIndex:
    """Carry the inventory fields fallback search reads from real parameters."""

    __spz_uid__: str
    __spz_name__: str
    desc: str = ""
    __spz_type__: str = "ParameterIndex"

    exposed_attributes = ("__spz_uid__", "__spz_name__", "__spz_type__", "desc")

    @property
    def uid(self) -> str:
        return self.__spz_uid__

    @property
    def name(self) -> str:
        return self.__spz_name__


@dataclass
class FakeVariable:
    """Expose only the SpeasyVariable data fields HelioAI consumes."""

    time: np.ndarray
    values: np.ndarray
    unit: str = ""
    columns: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    name: str = ""

    exposed_attributes = ("time", "values", "meta", "columns", "name", "unit")

    def __len__(self) -> int:
        return len(self.time)


@dataclass
class FakeGetDataCall:
    """Record speasy.get_data calls without tying tests to mock internals."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class FakeProvider:
    """Represent provider namespaces that expose parameter_range."""

    def __init__(self, range_result: Any = None, range_error: Exception | None = None) -> None:
        self.range_result = range_result
        self.range_error = range_error
        self.parameter_range_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def parameter_range(self, *args, **kwargs):
        self.parameter_range_calls.append((args, kwargs))
        if self.range_error is not None:
            raise self.range_error
        if callable(self.range_result):
            return self.range_result(*args, **kwargs)
        return self.range_result


class FakeSpeasy:
    """Model the speasy module shape imported lazily by the tools."""

    def __init__(
        self,
        *,
        catalogs: dict[str, FakeCatalogIndex] | None = None,
        timetables: dict[str, FakeCatalogIndex] | None = None,
        tree: Any | None = None,
        get_data: Any = None,
        get_data_error: Exception | None = None,
        ranges: dict[str, Any] | None = None,
    ) -> None:
        amda_flat = SimpleNamespace(catalogs=catalogs or {}, timetables=timetables or {})
        self.inventories = SimpleNamespace(
            flat_inventories=SimpleNamespace(amda=amda_flat),
            tree=tree if tree is not None else SimpleNamespace(),
        )
        self.get_data_result = get_data
        self.get_data_error = get_data_error
        self.get_data_calls: list[FakeGetDataCall] = []
        self.amda = FakeProvider()
        self.cda = FakeProvider()
        self.csa = FakeProvider()
        self.ssc = FakeProvider()
        for provider, result in (ranges or {}).items():
            setattr(self, provider, FakeProvider(result))

    def get_data(self, *args, **kwargs):
        self.get_data_calls.append(FakeGetDataCall(args, kwargs))
        if self.get_data_error is not None:
            raise self.get_data_error
        if callable(self.get_data_result):
            return self.get_data_result(*args, **kwargs)
        return self.get_data_result
