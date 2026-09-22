"""One seam between HelioAI and a System One judge — every question and every threshold.

Why one file: HelioAI is PyHC-listed and open source, and a reviewer who asks "what does
this agent delegate to a proprietary model?" must be able to answer by reading one module.
Every site that consults the judge builds its questions here-shaped (`Noul`, `Choice`),
calls `ask`, and reads an `Answers` — or `None`.

Why abstention is `None` and not a sentinel: a sentinel can be compared, summed and sorted
into a plausible wrong answer; `None > 0.5` raises. The judge abstains when the backend is
`null` (the default), when the site's experiment is off, when the call fails or times out,
and per question when the answer does not clear that question's own threshold. Every call
site reads `if answers is None: <today's code>` — so a build without a key, without the
extra, or with the default backend is bit-for-bit the loop that exists today. That is the
property the tests hold this module to.

Why observation only: the 2026-09-15 lesson — nothing that changes what the model sees
ships without a replay on both sides of the same question — applies to every answer here.
So `ask` records one JSON line per call under the session workspace (`judgment.jsonl`),
with the state in full: a disagreement that cannot be adjudicated later without the state
is not a measurement. What a site does with an answer is the site's business, and until a
site has earned it, that is: annotate, never correct.

Why the model call is bounded: a judgment that can hold a turn is a judgment that can
hang it. The whole call sits under `settings.judgment.timeout_s` (round trip measured
from France on 2026-09-22: median 266 ms, p95 373 ms), and a timeout abstains.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from helioai.config import settings
from helioai.logging_config import get_logger

log = get_logger(__name__)

RECORD_FILE = "judgment.jsonl"


@dataclass(frozen=True)
class Noul:
    """A yes/no question.

    The judge returns a probability; the answer is `True` at or above `0.5 + margin`,
    `False` at or below `0.5 - margin`, and `None` in between. The default margin makes
    0.7 the least confident "yes" — a site that needs more certainty raises it.

    Attributes:
        instructions: The question, in plain English, as the judge reads it.
        margin: Half-width of the abstention band around 0.5.
    """

    instructions: str
    margin: float = 0.2


@dataclass(frozen=True)
class Choice:
    """One option out of a closed set.

    The judge returns the chosen option and a confidence; below `floor` the answer is
    `None`. The options are the vocabulary the site already speaks — for a measurement
    type, the index's own `measurement_type` values — so an answer is usable as an exact
    filter, never as prose to interpret.

    Attributes:
        instructions: The question, in plain English.
        options: The closed set, in the order the site wants them presented.
        floor: Minimum confidence for the choice to count as an answer.
    """

    instructions: str
    options: tuple[str, ...]
    floor: float = 0.5


Question = Noul | Choice


@dataclass(frozen=True)
class Answers:
    """What the judge said for one call, already decided question by question.

    `values` holds `bool` for a `Noul`, `str` for a `Choice`, `None` where the answer did
    not clear its threshold; `raw` keeps the probabilities so the record can be re-judged
    at another threshold later without another call.

    Attributes:
        values: Decided answers, keyed like the questions.
        raw: The judge's probabilities, keyed like the questions.
        model: The judge model that answered.
        latency_ms: Wall time of the call as this process saw it.
        request_id: The provider's id for the call, for support and audit.
    """

    values: Mapping[str, bool | str | None]
    raw: Mapping[str, Any]
    model: str
    latency_ms: float
    request_id: str | None = None

    def __getitem__(self, name: str) -> bool | str | None:
        return self.values[name]

    def get(self, name: str, default: Any = None) -> Any:
        """The decided answer for `name`, or `default` when absent or abstained."""
        value = self.values.get(name)
        return default if value is None else value


def enabled(site: str) -> bool:
    """Whether `site` may ask at all: a judging backend AND the site's experiment name.

    Two axes on purpose. With the backend alone, "Jev does not help at this site" and "the
    layer costs something" could not be told apart; with the experiment alone, a site
    could be on with nobody to answer.
    """
    return settings.judgment.backend != "null" and f"judgment_{site}" in settings.agent.experiments


async def ask(
    site: str,
    state: Mapping[str, Any],
    questions: Mapping[str, Question],
    *,
    decided: Any = None,
) -> Answers | None:
    """Ask the judge; `None` means abstain, and the caller runs today's code.

    Args:
        site: The call site's name; `judgment_<site>` is its experiment.
        state: What the judge reads — JSON-serialisable, and recorded in full.
        questions: The questions, keyed by the names the site will read back.
        decided: What the deterministic path decided, when the site knows it before
            asking; recorded beside the answer so the two can be compared offline.

    Returns:
        The decided answers, or `None` when the judge abstained as a whole.
    """
    if not enabled(site):
        return None
    t0 = time.perf_counter()
    try:
        response = await asyncio.wait_for(
            _backend().ask(dict(state), dict(questions)), timeout=settings.judgment.timeout_s
        )
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        _warn_once(site, e)
        _record(site, state, questions, None, decided, latency, error=f"{type(e).__name__}: {e}")
        return None
    latency = (time.perf_counter() - t0) * 1000
    answers = Answers(
        values=_decide(questions, response.answers),
        raw=_raw(questions, response.answers),
        model=response.model,
        latency_ms=latency,
        request_id=response.request_id,
    )
    _record(site, state, questions, answers, decided, latency)
    return answers


def _decide(questions: Mapping[str, Question], raw: Mapping[str, Any]) -> dict[str, Any]:
    values: dict[str, bool | str | None] = {}
    for name, q in questions.items():
        answer = raw.get(name)
        if answer is None:
            values[name] = None
        elif isinstance(q, Noul):
            p = float(answer.noul)
            values[name] = True if p >= 0.5 + q.margin else False if p <= 0.5 - q.margin else None
        else:
            values[name] = answer.choice if float(answer.confidence) >= q.floor else None
    return values


def _raw(questions: Mapping[str, Question], raw: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, q in questions.items():
        answer = raw.get(name)
        if answer is None:
            out[name] = None
        elif isinstance(q, Noul):
            out[name] = {"noul": float(answer.noul)}
        else:
            out[name] = {
                "choice": answer.choice,
                "confidence": float(answer.confidence),
                "probabilities": dict(answer.probabilities),
            }
    return out


def _describe(q: Question) -> dict[str, Any]:
    if isinstance(q, Noul):
        return {"kind": "noul", "instructions": q.instructions, "margin": q.margin}
    return {
        "kind": "choice",
        "instructions": q.instructions,
        "options": list(q.options),
        "floor": q.floor,
    }


def _record_dir() -> Path | None:
    from helioai.runtime.context import RunContext

    ctx = RunContext.current()
    return ctx.session_dir if ctx is not None else None


async def batch(
    site: str,
    states: list[Mapping[str, Any]],
    questions: Mapping[str, Question],
    *,
    concurrency: int = 8,
    record_to: Path | None = None,
) -> list[Answers | None]:
    """Ask the same questions of many states, off the agent loop — for a job, not a turn.

    The runtime's `ask` is gated by a site's experiment name because it runs inside a
    conversation nobody asked to be judged. A job such as `helioai index --classify` is an
    explicit request: it needs only a judging backend, and it records every call to a file
    of its own (`record_to`) rather than to a session that does not exist. The same
    thresholds decide the answers, so what a job writes and what a turn would read agree.

    Args:
        site: The job's name, in the records.
        states: One state per item, JSON-serialisable.
        questions: The questions, asked of every state.
        concurrency: How many calls in flight at once (37 ms per item at eight, measured).
        record_to: The JSON-lines file every call is appended to; None records nothing.

    Returns:
        One `Answers` or `None` (abstained, failed, timed out) per state, in order.
    """
    if settings.judgment.backend == "null" or not states:
        return [None] * len(states)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(state: Mapping[str, Any]) -> Answers | None:
        async with sem:
            t0 = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    _backend().ask(dict(state), dict(questions)),
                    timeout=settings.judgment.timeout_s,
                )
            except Exception as e:
                latency = (time.perf_counter() - t0) * 1000
                _warn_once(site, e)
                _record(site, state, questions, None, None, latency, error=str(e), path=record_to)
                return None
            latency = (time.perf_counter() - t0) * 1000
            answers = Answers(
                values=_decide(questions, response.answers),
                raw=_raw(questions, response.answers),
                model=response.model,
                latency_ms=latency,
                request_id=response.request_id,
            )
            _record(site, state, questions, answers, None, latency, path=record_to)
            return answers

    return list(await asyncio.gather(*(one(st) for st in states)))


def _record(
    site: str,
    state: Mapping[str, Any],
    questions: Mapping[str, Question],
    answers: Answers | None,
    decided: Any,
    latency_ms: float,
    *,
    error: str | None = None,
    path: Path | None = None,
) -> None:
    line = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "site": site,
        "backend": settings.judgment.backend,
        "model": answers.model if answers else settings.judgment.model,
        "latency_ms": round(latency_ms, 1),
        "request_id": answers.request_id if answers else None,
        "state": dict(state),
        "questions": {n: _describe(q) for n, q in questions.items()},
        "raw": dict(answers.raw) if answers else None,
        "answers": dict(answers.values) if answers else None,
        "decided": decided,
        "error": error,
    }
    try:
        if path is None:
            directory = _record_dir()
            if directory is None:
                log.debug("judgment_unrecorded_no_session", site=site)
                return
            path = directory / RECORD_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        log.warning("judgment_record_failed", site=site, error=str(e))


_warned: set[str] = set()


def _warn_once(site: str, error: Exception) -> None:
    key = f"{site}:{type(error).__name__}"
    if key not in _warned:
        _warned.add(key)
        log.warning("judgment_abstained", site=site, error=f"{type(error).__name__}: {error}")


# ── The intent contract ──────────────────────────────────────────────────────────────
#
# What a question commits the answer to, read once from the question alone. Every field
# is a Choice over a closed set or a yes/no: the judge never writes free text into the
# system, and a date arrives as its named components — the code assembles the datetime,
# because ordering dates is arithmetic and arithmetic is not what a reading model is for.
# The quantity is a Choice over the index's own measurement-type vocabulary, so a later
# join between "what was asked" and "what was retrieved" is an exact comparison of two
# strings, not a judgement.

_YEARS = tuple(str(y) for y in range(1990, 2028))
_MONTHS = tuple(f"{m:02d}" for m in range(1, 13))
_DAYS = tuple(f"{d:02d}" for d in range(1, 32))
NONE = "none"

INTENT_QUESTIONS: dict[str, Question] = {
    "deliverable": Choice(
        "What does the request want delivered? A measured value or number; a figure or plot; "
        "a catalogue or list of events; an explanation with no computation; a procedure or "
        "code to run elsewhere; or something else.",
        ("value", "figure", "catalogue", "explanation", "procedure", "other"),
        floor=0.6,
    ),
    "quantity": Choice(
        "Which SPASE measurement type is the physical quantity the request is about? "
        "MagneticField for B, IMF, field components or magnitude; ThermalPlasma for density, "
        "temperature, velocity, plasma beta of the bulk plasma; EnergeticParticles for fluxes "
        "above thermal energies, SEP, cosmic rays; Ephemeris for a position, orbit, trajectory; "
        "ElectricField, Waves, IonComposition as named; none when the request names no quantity "
        "(a catalogue lookup, an explanation).",
        (
            "MagneticField",
            "ElectricField",
            "ThermalPlasma",
            "EnergeticParticles",
            "IonComposition",
            "Ephemeris",
            "Waves",
            "Spectrum",
            NONE,
        ),
        floor=0.6,
    ),
    "frame": Choice(
        "Which coordinate frame does the request name for the data, if any?",
        ("GSE", "GSM", "RTN", "GEO", "SM", "HEE", "HCI", "other", NONE),
        floor=0.6,
    ),
    "year": Choice("Which year does the request name, if any?", _YEARS + (NONE,), floor=0.6),
    "month": Choice(
        "Which month (01–12) does the request name, if any?", _MONTHS + (NONE,), floor=0.6
    ),
    "day": Choice(
        "Which day of the month (01–31) does the request name, if any?", _DAYS + (NONE,), floor=0.6
    ),
    "event_named": Noul("Does the request name a specific event, date or time interval?"),
    "uncertainty_required": Noul(
        "Does the request ask for an uncertainty, an error bar, a confidence or a spread?"
    ),
    "method_named": Noul(
        "Does the request name a specific method, recipe, formula or reference to use?"
    ),
    "two_spacecraft": Noul(
        "Does the request compare or combine two or more spacecraft or instruments?"
    ),
}


async def intent_contract(query: str) -> Answers | None:
    """The contract for one question, or None — see `INTENT_QUESTIONS`.

    Args:
        query: The user's question, verbatim.
    """
    if not query or not query.strip():
        return None
    return await ask("intent", {"question": query}, INTENT_QUESTIONS)


def contract_fields(answers: Answers) -> dict[str, Any]:
    """The `intent` event's payload: the decided answers, the date assembled by the code.

    A Choice of `none`, or an abstention, is `None` in the payload — the reader must not
    mistake "the judge did not say" for "the request named nothing". `date` is the ISO day,
    month or year the components make, with `date_precision` saying which.
    """
    out: dict[str, Any] = {}
    for name in INTENT_QUESTIONS:
        value = answers.get(name)
        out[name] = None if value in (None, NONE) else value
    year, month, day = out.pop("year"), out.pop("month"), out.pop("day")
    if year and month and day:
        out["date"], out["date_precision"] = f"{year}-{month}-{day}", "day"
    elif year and month:
        out["date"], out["date_precision"] = f"{year}-{month}", "month"
    elif year:
        out["date"], out["date_precision"] = year, "year"
    else:
        out["date"], out["date_precision"] = None, None
    out["model"] = answers.model
    out["latency_ms"] = round(answers.latency_ms, 1)
    return out


async def collect(task: asyncio.Task) -> dict[str, Any] | None:
    """Read a contract task the turn started; abstain if it has not answered in time.

    The task ran concurrently with the model; by the time the answer is out it has almost
    always finished. A turn never waits on the judge longer than one judgment budget.
    """
    try:
        answers = await asyncio.wait_for(task, timeout=settings.judgment.timeout_s)
    except Exception as e:
        _warn_once("intent", e)
        return None
    return contract_fields(answers) if answers is not None else None


class _JevBackend:
    """TypeSafe's System One through `typesafe-sdk`, imported only when first asked.

    One client per process, no SDK-level retry: the whole call is already bounded by
    `ask`, and a retry inside a 2 s budget would only make the timeout the common case.
    """

    def __init__(self) -> None:
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not settings.judgment.api_key:
                raise RuntimeError("TYPESAFE_API_KEY is not set")
            from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

            self._client = AsyncTypeSafeClient(
                api_key=settings.judgment.api_key,
                model=settings.judgment.model,
                retry=RetryPolicy(max_retries=0, timeout=settings.judgment.timeout_s),
            )
        return self._client

    async def ask(self, state: dict[str, Any], questions: dict[str, Question]) -> Any:
        client = self._get_client()
        return await client.system_one(state, to_sdk(questions))


def to_sdk(questions: Mapping[str, Question]) -> dict[str, Any]:
    """Our question shapes as `typesafe_sdk` ones — the only place the SDK's names appear.

    Options of a `Choice` become criteria without descriptions: the vocabulary is the
    site's own (an index field, a closed list of regions), and describing it in prose
    would put a second, unmeasured question in front of the judge.
    """
    from typesafe_sdk import Choice as SdkChoice
    from typesafe_sdk import Noul as SdkNoul

    out: dict[str, Any] = {}
    for name, q in questions.items():
        if isinstance(q, Noul):
            out[name] = SdkNoul(instructions=q.instructions)
        else:
            out[name] = SdkChoice(
                instructions=q.instructions, criteria=dict.fromkeys(q.options, None)
            )
    return out


_jev: _JevBackend | None = None


def _backend() -> _JevBackend:
    global _jev
    if _jev is None:
        _jev = _JevBackend()
    return _jev


async def aclose() -> None:
    """Close the judge's connection pool at shutdown; safe when it was never opened."""
    global _jev
    if _jev is not None and _jev._client is not None:
        from helioai.core.llm.base import close_sdk_client

        await close_sdk_client(_jev._client)
    _jev = None
