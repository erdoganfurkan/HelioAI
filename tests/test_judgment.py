"""The judgment seam: abstention is `None`, and `None` is exactly today's loop.

Every guarantee the runtime relies on is a test here: the default backend costs nothing
and touches nothing; a site whose experiment is off never asks; a judge that fails or
hangs abstains and is recorded; thresholds turn probabilities into answers or `None`;
and an answer is one JSON line with the state in full. The SDK is imported in one test
only, skipped without the extra — what it pins is the mapping of our questions onto its
types, which is all an offline test can honestly pin.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from helioai import config
from helioai.config import settings
from helioai.core import judgment
from helioai.core.judgment import Answers, Choice, Noul, ask


class _Refuses:
    """A backend that must never be reached."""

    async def ask(self, state, questions):  # pragma: no cover — reaching it is the failure
        raise AssertionError("the backend was consulted")


class _Answers:
    def __init__(self, answers, model="jev-test", request_id="req_test"):
        self.response = SimpleNamespace(answers=answers, model=model, request_id=request_id)
        self.calls: list[tuple[dict, dict]] = []

    async def ask(self, state, questions):
        self.calls.append((state, questions))
        return self.response


class _Raises:
    async def ask(self, state, questions):
        raise ConnectionError("no route to judge")


class _Hangs:
    async def ask(self, state, questions):
        await asyncio.sleep(10)


QUESTIONS = {
    "relevant": Noul("Does this product measure what the query asks for?"),
    "mtype": Choice("Which measurement type?", ("MagneticField", "Ephemeris"), floor=0.6),
}
STATE = {"query": "MMS1 position 2019", "product": "IMAP sc_position_GSE"}


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(judgment, "_record_dir", lambda: tmp_path)
    monkeypatch.setattr(judgment, "_warned", set())
    return tmp_path / judgment.RECORD_FILE


@pytest.fixture
def jev_on(monkeypatch):
    monkeypatch.setattr(settings.judgment, "backend", "jev")
    monkeypatch.setattr(settings.judgment, "timeout_s", 0.2)
    monkeypatch.setattr(settings.agent, "experiments", frozenset({"judgment_search"}))


def test_the_default_backend_abstains_with_no_io_and_no_record(recorded, monkeypatch):
    monkeypatch.setattr(settings.judgment, "backend", "null")
    monkeypatch.setattr(settings.agent, "experiments", frozenset({"judgment_search"}))
    monkeypatch.setattr(judgment, "_backend", lambda: _Refuses())

    assert asyncio.run(ask("search", STATE, QUESTIONS)) is None
    assert not recorded.exists()


def test_a_site_whose_experiment_is_off_never_asks_even_with_a_judge(recorded, monkeypatch):
    monkeypatch.setattr(settings.judgment, "backend", "jev")
    monkeypatch.setattr(settings.agent, "experiments", frozenset())
    monkeypatch.setattr(judgment, "_backend", lambda: _Refuses())

    assert asyncio.run(ask("search", STATE, QUESTIONS)) is None
    assert not judgment.enabled("search")
    assert not recorded.exists()


def test_an_answer_is_decided_per_question_and_recorded_with_the_state_in_full(
    recorded, jev_on, monkeypatch
):
    backend = _Answers(
        {
            "relevant": SimpleNamespace(noul=0.03),
            "mtype": SimpleNamespace(
                choice="Ephemeris",
                confidence=0.91,
                probabilities={"MagneticField": 0.09, "Ephemeris": 0.91},
            ),
        }
    )
    monkeypatch.setattr(judgment, "_backend", lambda: backend)

    answers = asyncio.run(ask("search", STATE, QUESTIONS, decided={"rank": 1}))

    assert isinstance(answers, Answers)
    assert answers["relevant"] is False and answers["mtype"] == "Ephemeris"
    assert answers.get("relevant", "abstained") is False
    assert answers.model == "jev-test" and answers.request_id == "req_test"
    assert backend.calls[0][0] == STATE
    lines = [json.loads(row) for row in recorded.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    line = lines[0]
    assert line["site"] == "search" and line["state"] == STATE and line["decided"] == {"rank": 1}
    assert line["answers"] == {"relevant": False, "mtype": "Ephemeris"}
    assert line["raw"]["mtype"]["probabilities"]["Ephemeris"] == 0.91
    assert line["questions"]["mtype"] == {
        "kind": "choice",
        "instructions": "Which measurement type?",
        "options": ["MagneticField", "Ephemeris"],
        "floor": 0.6,
    }
    assert line["error"] is None and line["latency_ms"] >= 0


def test_thresholds_turn_probabilities_into_answers_or_none():
    q = {
        "yes": Noul("?"),
        "no": Noul("?"),
        "unsure": Noul("?", margin=0.3),
        "pick": Choice("?", ("a", "b"), floor=0.8),
    }
    raw = {
        "yes": SimpleNamespace(noul=0.71),
        "no": SimpleNamespace(noul=0.29),
        "unsure": SimpleNamespace(noul=0.75),
        "pick": SimpleNamespace(choice="a", confidence=0.79, probabilities={"a": 0.79, "b": 0.21}),
    }

    assert judgment._decide(q, raw) == {"yes": True, "no": False, "unsure": None, "pick": None}
    assert judgment._decide(q, {})["yes"] is None


def test_a_failing_judge_abstains_and_the_failure_is_recorded(recorded, jev_on, monkeypatch):
    monkeypatch.setattr(judgment, "_backend", lambda: _Raises())

    assert asyncio.run(ask("search", STATE, QUESTIONS)) is None
    line = json.loads(recorded.read_text(encoding="utf-8").splitlines()[0])
    assert line["answers"] is None and line["error"].startswith("ConnectionError")


def test_a_slow_judge_abstains_within_the_budget(recorded, jev_on, monkeypatch):
    monkeypatch.setattr(judgment, "_backend", lambda: _Hangs())

    async def timed():
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        result = await ask("search", STATE, QUESTIONS)
        return result, loop.time() - t0

    result, elapsed = asyncio.run(timed())
    assert result is None and elapsed < 2.0
    line = json.loads(recorded.read_text(encoding="utf-8").splitlines()[0])
    assert line["error"].startswith("TimeoutError")


def test_a_missing_session_leaves_no_file_and_still_answers(jev_on, monkeypatch):
    monkeypatch.setattr(judgment, "_record_dir", lambda: None)
    monkeypatch.setattr(
        judgment, "_backend", lambda: _Answers({"relevant": SimpleNamespace(noul=0.9)})
    )

    answers = asyncio.run(ask("search", STATE, {"relevant": QUESTIONS["relevant"]}))
    assert answers is not None and answers["relevant"] is True


def test_an_unknown_backend_is_refused_where_the_key_is_checked(monkeypatch):
    """`import helioai.config` must not fail on a typo (the MCP server and the web app
    start from it); the refusal lives in `build_llm_client` and in `doctor`, like an
    unknown experiment."""
    from helioai.core.llm.factory import build_llm_client
    from helioai.doctor import check_judgment

    monkeypatch.setenv("HELIOAI_JUDGMENT_BACKEND", "Local")
    assert config._load().judgment.backend == "local"
    with pytest.raises(RuntimeError, match="local"):
        config.validate_judgment(config.JudgmentConfig(backend="local"))

    monkeypatch.setattr(settings.judgment, "backend", "local")
    with pytest.raises(RuntimeError, match="local"):
        build_llm_client("groq")
    assert check_judgment().status == "fail"

    monkeypatch.setattr(settings.judgment, "backend", "null")
    assert check_judgment().status == "ok"
    monkeypatch.setattr(settings.judgment, "backend", "jev")
    monkeypatch.setattr(settings.judgment, "api_key", "")
    assert check_judgment().status == "fail" and "TYPESAFE_API_KEY" in check_judgment().detail


def test_the_key_is_read_from_the_environment_and_never_required(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("HELIOAI_JUDGMENT_BACKEND", raising=False)
    loaded = config._load()
    assert loaded.judgment.backend == "null" and loaded.judgment.api_key == ""
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setenv("HELIOAI_JUDGMENT_TIMEOUT_S", "0.5")
    loaded = config._load()
    assert loaded.judgment.api_key == "k" and loaded.judgment.timeout_s == 0.5


def test_our_questions_map_onto_the_sdk_types():
    pytest.importorskip("typesafe_sdk")
    sdk = judgment.to_sdk(QUESTIONS)
    assert type(sdk["relevant"]).__name__ == "Noul"
    assert sdk["relevant"].instructions == QUESTIONS["relevant"].instructions
    assert type(sdk["mtype"]).__name__ == "Choice"
    assert dict(sdk["mtype"].criteria) == {"MagneticField": None, "Ephemeris": None}


def test_the_jev_backend_needs_a_key_before_it_imports_anything(monkeypatch):
    monkeypatch.setattr(settings.judgment, "api_key", "")
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        judgment._JevBackend()._get_client()
