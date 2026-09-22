"""What the turn did about what was asked — four joins, no model, observation only.

The intent contract (`judgment.intent_contract`) says what a question committed the answer
to. This module places that contract against what the turn actually produced — the
parameter cards of every product loaded, the claims of `final_answer`, the figures — and
records the comparison on the `intent` event. Each join is an exact operation on typed
fields: a set membership, an interval intersection, a count. None calls a model, none
reads prose, and none changes the answer; the `intent` event is what a reader of the
journal, or a later correction path behind its own experiment, will consult.

The four joins are the failures the benches actually produced, one each:

- **frame** — the frame the question named against the `coord_sys` of the cards the
  sandbox filled from the archive's metadata: "GSM asked, `BGSE` plotted" was caught by
  nothing.
- **window** — the date the question named against the bounds the downloads obtained,
  and the cards whose series stopped short of the window asked for (`coverage_note`).
  The 2026-09-18 θ_Bn of 12° for a 54° shock had a window that ended at the shock; every
  check downstream was green because none looked at the bounds.
- **quantity** — the measurement type the question named against the indexed type of
  the products loaded: twelve searches and a confident answer built on no product that
  measured the quantity (2026-09-15) is a count, once the field is filled.
- **responsiveness** — what the question required against what the answer delivered:
  an uncertainty asked and no claim about a spread, a figure asked and none produced.

A join that has nothing to compare — no frame named, no card with a bound, no type on
any loaded product — reports `None`, never a verdict: "the join could not say" and "the
turn got it right" are kept apart, as they are in the contract itself.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from helioai.tools import rag

UNCERTAINTY_WORDS = re.compile(
    r"(?:^|[_\s-])(?:std|stdev|sigma|spread|uncertaint\w*|error|err|stderr|sem|ci|iqr|mad)"
    r"(?:[_\s-]|$)|±",
    re.I,
)

_FIGURE_DELIVERABLES = frozenset({"figure"})
_VALUE_DELIVERABLES = frozenset({"value"})
_CATALOGUE_DELIVERABLES = frozenset({"catalogue"})


def checks(contract: dict, artifacts: list[dict], claims: list[dict]) -> dict[str, Any]:
    """The four joins for one turn, as the `checks` field of the `intent` event.

    Args:
        contract: `judgment.contract_fields` output — the decided intent, `None` where
            the judge abstained or the request named nothing.
        artifacts: `RunEnd.artifacts` — every parameter card, figure, export and
            catalogue preview the run produced, sub-agents included.
        claims: `RunEnd.claims` — the numbers the answer named through `final_answer`.
    """
    cards = [a for a in artifacts if a.get("kind") == "parameter_card"]
    return {
        "frame": _frame(contract.get("frame"), cards),
        "window": _window(contract.get("date"), contract.get("date_precision"), cards),
        "quantity": _quantity(contract.get("quantity"), cards),
        "responsiveness": _responsiveness(contract, artifacts, claims),
    }


def _frame(asked: str | None, cards: list[dict]) -> dict[str, Any] | None:
    loaded = sorted({str(c["coord_sys"]).upper() for c in cards if c.get("coord_sys")})
    if not asked or asked == "other" or not loaded:
        return None
    return {"asked": asked, "loaded": loaded, "match": asked.upper() in loaded}


def _window(date: str | None, precision: str | None, cards: list[dict]) -> dict[str, Any] | None:
    short = sorted({c["param_id"] for c in cards if c.get("coverage_note") and c.get("param_id")})
    bounds = [b for b in (_card_bounds(c) for c in cards) if b is not None]
    covered: bool | None = None
    if date and precision and bounds:
        asked = _interval(date, precision)
        if asked is not None:
            covered = any(lo < asked[1] and hi > asked[0] for lo, hi in bounds)
    if not short and covered is None:
        return None
    return {"date": date, "covered": covered, "short": short}


def _card_bounds(card: dict) -> tuple[datetime, datetime] | None:
    lo = _parse(card.get("obtained_start") or card.get("start"))
    hi = _parse(card.get("obtained_stop") or card.get("stop"))
    if lo is None or hi is None or hi < lo:
        return None
    return lo, hi


def _interval(date: str, precision: str) -> tuple[datetime, datetime] | None:
    try:
        if precision == "day":
            lo = datetime.strptime(date, "%Y-%m-%d")
            return lo, lo + timedelta(days=1)
        if precision == "month":
            lo = datetime.strptime(date, "%Y-%m")
            nxt = (
                lo.replace(year=lo.year + 1, month=1)
                if lo.month == 12
                else lo.replace(month=lo.month + 1)
            )
            return lo, nxt
        if precision == "year":
            lo = datetime.strptime(date, "%Y")
            return lo, lo.replace(year=lo.year + 1)
    except ValueError:
        return None
    return None


def _parse(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "")[:19]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _quantity(asked: str | None, cards: list[dict]) -> dict[str, Any] | None:
    ids = sorted({c["param_id"] for c in cards if c.get("param_id")})
    if not asked or not ids:
        return None
    family = rag.TYPE_FAMILIES.get(asked)
    if family is None:
        return None
    types = rag.measurement_types_of(ids)
    loaded: dict[str, int] = {}
    untyped = 0
    matching = 0
    for pid in ids:
        raw = types.get(pid)
        if not raw:
            untyped += 1
            continue
        loaded[raw] = loaded.get(raw, 0) + 1
        if rag._types_of(raw) & family:
            matching += 1
    match: bool | None = None if not loaded else matching > 0
    return {"asked": asked, "loaded": loaded, "untyped": untyped, "match": match}


def _responsiveness(
    contract: dict, artifacts: list[dict], claims: list[dict]
) -> dict[str, Any] | None:
    figures = sum(len(a.get("figure_paths") or []) for a in artifacts if a.get("kind") == "image")
    exports = [k for a in artifacts if a.get("kind") == "exports" for k in (a.get("values") or {})]
    catalogues = sum(1 for a in artifacts if a.get("kind") == "catalog_preview")
    out: dict[str, Any] = {}

    required = contract.get("uncertainty_required")
    if required is not None:
        named = [str(c.get("name") or "") for c in claims] + [str(k) for k in exports]
        claimed = any(UNCERTAINTY_WORDS.search(n) for n in named)
        out["uncertainty"] = {"required": required, "claimed": claimed}

    deliverable = contract.get("deliverable")
    if deliverable in _FIGURE_DELIVERABLES:
        out["deliverable"] = {"asked": deliverable, "figures": figures, "match": figures > 0}
    elif deliverable in _VALUE_DELIVERABLES:
        n = len(claims) + len(exports)
        out["deliverable"] = {
            "asked": deliverable,
            "claims": len(claims),
            "exports": len(exports),
            "match": n > 0,
        }
    elif deliverable in _CATALOGUE_DELIVERABLES:
        out["deliverable"] = {
            "asked": deliverable,
            "catalogues": catalogues,
            "match": catalogues > 0,
        }
    elif deliverable:
        out["deliverable"] = {"asked": deliverable, "match": None}

    return out or None


def summary(check: dict[str, Any] | None) -> list[str]:
    """The mismatches, as short phrases for a renderer; empty when every join agreed or
    had nothing to compare."""
    if not check:
        return []
    out: list[str] = []
    frame = check.get("frame")
    if frame and frame.get("match") is False:
        out.append(f"frame {frame['asked']} asked, {'/'.join(frame['loaded'])} loaded")
    window = check.get("window")
    if window:
        if window.get("covered") is False:
            out.append(f"{window['date']} not in any loaded series")
        if window.get("short"):
            out.append(f"{len(window['short'])} series short of the window asked")
    quantity = check.get("quantity")
    if quantity and quantity.get("match") is False:
        loaded = ", ".join(sorted(quantity["loaded"]))
        out.append(f"{quantity['asked']} asked, loaded {loaded}")
    resp = check.get("responsiveness") or {}
    unc = resp.get("uncertainty")
    if unc and unc.get("required") and not unc.get("claimed"):
        out.append("uncertainty asked, none claimed")
    deliv = resp.get("deliverable")
    if deliv and deliv.get("match") is False:
        out.append(f"{deliv['asked']} asked, none produced")
    return out
