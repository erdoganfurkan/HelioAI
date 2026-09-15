"""One verdict on a finished answer: its ids, its recipes, its figures, its numbers.

Four checks judged the lead's answer from four places — `_flag_unknown_ids` and
`_flag_recipe_bypass` in `tool_exec`, `provenance_check.check_reply` for the numbers in
the prose, `vision.maybe_review` for the figures — each with its own event and none aware
of the others. `validate` runs them in one place and returns a `Verdict` the wrapper turns
into the events the interfaces already render, plus one `verdict` event that carries the
whole judgement.

What is new is the judgement of *claims*: when the model closed with `final_answer`, it
named each number and where it came from. Those are compared to the provenance ledger by
**name**, with a unit-aware tolerance — `2.5` states a recorded `2.519`, `57.2 deg` states
`57.16 deg`, `0.0025 pT` states `2.5 nT` — instead of being found again in the prose and
attributed by the words around them. The regex check stays as the net under the prose:
a number the model stated without claiming it is still judged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from helioai import provenance
from helioai.core.provenance_check import RTOL, _is_scalar, _states, check_reply
from helioai.core.tool_exec import _flag_recipe_bypass, _flag_unknown_ids
from helioai.logging_config import get_logger

log = get_logger(__name__)

_SESSION_SOURCES_ONLY = frozenset({"literature", "asserted"})


@dataclass
class Verdict:
    """How an answer holds up, in every respect the runtime can check.

    Attributes:
        matched: Claims a ledger entry of the same name states (unit-aware).
        contradicted: Claims naming a recorded scalar that holds another value.
        unsourced: Claims nothing in the session computed — including those the model
            itself marked `literature` or `asserted`, which are never contradicted.
        unknown_ids: Parameter ids the answer quotes that exist in no catalogue.
        recipe_flags: Exports that look like a calibrated recipe's output while the
            recipe was never loaded, or loaded and never called.
        figure_reviews: The vision verdicts on the run's figures.
        prose: The regex-based provenance report on the free text, or None when the
            session computed nothing or the text states no number.
    """

    matched: list[dict] = field(default_factory=list)
    contradicted: list[dict] = field(default_factory=list)
    unsourced: list[dict] = field(default_factory=list)
    unknown_ids: list[str] = field(default_factory=list)
    recipe_flags: list[dict] = field(default_factory=list)
    figure_reviews: list[str] = field(default_factory=list)
    prose: dict | None = None

    @property
    def has_claims(self) -> bool:
        """Whether the answer named its numbers at all."""
        return bool(self.matched or self.contradicted or self.unsourced)

    def as_event(self) -> dict:
        """The `verdict` event payload: counts up front, every detail behind them."""
        return {
            "matched": len(self.matched),
            "contradicted": len(self.contradicted),
            "unsourced": len(self.unsourced),
            "unknown_ids": list(self.unknown_ids),
            "recipe_flags": list(self.recipe_flags),
            "figure_reviews": list(self.figure_reviews),
            "claims": [
                {"status": status, **c}
                for status, group in (
                    ("contradicted", self.contradicted),
                    ("unsourced", self.unsourced),
                    ("matched", self.matched),
                )
                for c in group
            ],
        }


def validate(
    text: str,
    claims: list[dict],
    *,
    history: list,
    artifacts: list[dict],
    session_dir: Path,
    figure_reviews: list[str] | None = None,
) -> tuple[str, Verdict]:
    """Judge a finished answer.

    Args:
        text: The answer, as the model wrote it.
        claims: The numbers it named (`final_answer`), each `{name, value, units,
            source}`; empty for a prose answer.
        history: The turn's messages, read for evidence a recipe was loaded.
        artifacts: What the run exported, to tell a computed value from a quoted one.
        session_dir: Whose ledger to check against.
        figure_reviews: The vision verdicts collected during the run.

    Returns:
        The text — annotated when a check appends a correction, exactly as the two
        checks always annotated it — and the verdict.
    """
    text, unknown_ids = _flag_unknown_ids(text)
    text, recipe_flags = _flag_recipe_bypass(text, history, artifacts)
    verdict = Verdict(
        unknown_ids=unknown_ids,
        recipe_flags=recipe_flags,
        figure_reviews=list(figure_reviews or []),
        prose=_prose_report(text, session_dir),
    )
    if claims:
        entries = provenance.read_ledger(session_dir).get("values") or []
        for claim in claims:
            status, detail = judge_claim(claim, entries)
            getattr(verdict, status).append(detail)
        verdict.prose = _without_claimed_numbers(verdict.prose, claims)
    return text, verdict


def _without_claimed_numbers(prose: dict | None, claims: list[dict]) -> dict | None:
    """The prose report minus the numbers the claims already judged.

    The regex check attributes a number to an export by the words around it; the
    claims name the export. On the MMS1 live run the same "21.36 R_E" was matched to
    `MMS1_X_GSM_Re` by its claim and contradicted against `MMS1_Z_GSM_Re` by the regex,
    which had read "Z" nearby — two verdicts on one number, the wrong one in red. A
    number a claim covers keeps the claim's verdict alone; the prose net stays for the
    numbers the model stated without claiming.
    """
    if not prose:
        return prose
    values = [v for v in (_number(c.get("value")) for c in claims) if v is not None]
    if not values:
        return prose

    def claimed(x: object) -> bool:
        v = _number(x)
        return v is not None and any(
            abs(v - c) <= RTOL * max(abs(v), abs(c), 1e-12) for c in values
        )

    kept = [d for d in prose.get("details") or [] if not claimed(d.get("value"))]
    dropped = [d for d in prose.get("details") or [] if claimed(d.get("value"))]
    out = dict(prose, details=kept)
    for d in dropped:
        key = d.get("status")
        if key in out and isinstance(out[key], int) and out[key] > 0:
            out[key] -= 1
    return out


def _prose_report(text: str, session_dir: Path) -> dict | None:
    try:
        return check_reply(text, session_dir)
    except Exception:
        log.debug("prose_provenance_failed", exc_info=True)
        return None


def judge_claim(claim: dict, entries: list[dict]) -> tuple[str, dict]:
    """Place one claim against the ledger: `matched`, `contradicted` or `unsourced`.

    The entries considered are those whose name is the claim's `source` or `name` — the
    export it says it came from. Any run that produced the value sources it (a session
    that exported `compression_ratio` twice, 3.045 then 2.538, computed both), so the
    entries are read newest first and the first that states the value wins, within the
    prose checker's tolerance (`RTOL`) or the rounding the claim shows. A named scalar
    that states another value contradicts the claim; an entry holding several values
    cannot accuse, since a reply legitimately quotes one component of a vector; nor can
    a dimensioned scalar accuse a claim that gives no units — the claim may be another
    quantity of the same run, and on the first live run it was (a normal's components
    filed under the angle's export). A claim the model marked `literature` or
    `asserted` is never contradicted: it did not say the session computed it.

    Args:
        claim: `{name, value, units, source}` as `final_answer` delivered it.
        entries: `provenance.read_ledger(...)["values"]`.

    Returns:
        The status and the detail dict shown to a reader.
    """
    detail = {
        "name": claim.get("name"),
        "value": claim.get("value"),
        "units": claim.get("units") or "",
        "source": claim.get("source") or "asserted",
    }
    value = _number(claim.get("value"))
    source = str(detail["source"])
    if value is None or source in _SESSION_SOURCES_ONLY:
        return "unsourced", detail
    names = {source, str(detail["name"] or "")}
    named = [e for e in reversed(entries) if e.get("name") in names]
    if not named:
        return "unsourced", detail
    comparable = False
    for entry in named:
        converted = _in_ledger_units(value, str(detail["units"]), str(entry.get("units") or ""))
        if converted is None:
            continue
        comparable = True
        if converted is _INCOMPATIBLE:
            continue
        if _states(entry, converted, RTOL) or _within_rounding(entry, converted):
            detail["ledger"] = _ledger_value(entry)
            detail["code_path"] = entry.get("code_path")
            return "matched", detail
    if not comparable:
        detail["note"] = "units could not be reconciled with the ledger's"
        return "unsourced", detail
    scalars = [e for e in named if _is_scalar(e)]
    if not scalars:
        return "unsourced", detail
    recorded = scalars[0]
    recorded_units = str(recorded.get("units") or "")
    if not str(detail["units"]).strip() and recorded_units:
        # The first live run named the shock normal's components `source: theta_bn` —
        # the run that printed them — with no units; the ledger's `theta_bn` is the
        # angle. "n_x stated -0.509, the session computed 54.85 deg" accuses nothing a
        # reader can act on: a claim without units may be another quantity of the same
        # run, and here it was. Only a claim that says what it measures can be wrong.
        detail["note"] = (
            f"no units given; the export named holds {_ledger_value(recorded)} {recorded_units}"
        )
        return "unsourced", detail
    detail["ledger"] = _ledger_value(recorded)
    detail["ledger_units"] = recorded_units
    detail["code_path"] = recorded.get("code_path")
    return "contradicted", detail


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


_INCOMPATIBLE = object()


def _in_ledger_units(value: float, claim_units: str, ledger_units: str) -> float | object | None:
    """The claim's value expressed in the ledger's units.

    Two blanks, or two identical strings, need no conversion; anything else goes through
    astropy. A blank ledger unit means the export did not say — a `run_python` export
    without `units=`, or a ratio — so the claim's value is compared as is, whatever it
    was labelled: the SEA live run claimed `peak_tau = 0.606 "dimensionless (normalized
    epoch)"` against an export recorded blank and was told the units could not be
    reconciled. A claim unit that does not parse against a ledger unit that does leaves
    the claim unjudged (None) rather than accused — a spelling is not a contradiction.
    Two units that parse and cannot be converted into each other (`deg` against `km/s`)
    are one, and return `_INCOMPATIBLE`.
    """
    a, b = claim_units.strip(), ledger_units.strip()
    if a.lower() == b.lower() or not a or not b:
        return value
    try:
        import astropy.units as u

        ua, ub = u.Unit(a), u.Unit(b)
    except Exception:
        return None
    try:
        return float((value * ua).to(ub).value)
    except Exception:
        return _INCOMPATIBLE


def _within_rounding(entry: dict, value: float) -> bool:
    """Whether the claim is the entry's mean rounded to the digits the claim shows, or —
    for an entry that summarises a series — within its spread. `2.5` states a recorded
    `2.519`; `60` states `59.95`; a mean quoted inside one standard deviation of a time
    series is not a different number."""
    mean = entry.get("mean")
    if not isinstance(mean, (int, float)) or isinstance(mean, bool):
        return False
    tol = 0.5 * 10 ** (-_decimals(value))
    if abs(abs(mean) - abs(value)) <= tol:
        return True
    std = entry.get("std")
    if not _is_scalar(entry) and isinstance(std, (int, float)) and std > 0:
        return abs(abs(mean) - abs(value)) <= std
    return False


def _decimals(value: float) -> int:
    text = repr(float(value))
    if "e" in text or "E" in text:
        return 0
    return len(text.split(".")[1].rstrip("0")) if "." in text else 0


def _ledger_value(entry: dict) -> object:
    return entry.get("mean")
