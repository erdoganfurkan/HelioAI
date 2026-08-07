"""Confront a written answer with the values the session actually computed.

The provenance ledger records every `export()` of a session (see `helioai.provenance`).
This module reads a reply, pulls out the numbers it states, and asks the ledger where
each one came from. The result is emitted as a `provenance` event next to the reply.

**It annotates, it never blocks.** An approximate numeric match cannot decide what is
legitimate: "about 29% of the separation" is derived from measurements without being one
of them, and rejecting it would make the tool worse than silent. What was missing was not
a gate but visibility — the failure it exists to expose (a reply publishing 13.02 nT that
no script in the session computes) stayed invisible for two days of debugging.

It certifies provenance, not correctness: a wrong averaging window produces a traceable,
recorded, and wrong number, and nothing here will say so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from helioai import provenance

_UNITS = (
    r"nT|pT|km/s|km s-1|cm\^?-3|cm-3|cm⁻³|/cm3|nPa|eV|keV|MeV|kK|MK|K|Hz|mHz|kHz|"
    r"RE|R_E|Re|deg|°|%|s|min|h"
)

_TOKEN = re.compile(rf"(?<![\w.])(-?\d+(?:\.\d+)?)(?:\s*({_UNITS})(?![\w/^]))?")

# A bare integer under this is almost always "3 panels" or "2 spacecraft"; matching them
# floods the report with noise that hides the one number that matters. Written with a
# decimal point it is a measurement — "M_A ~ 4.0" is a claim, "4 panels" is not.
_MIN_BARE_INT = 10.0
_YEARS = range(1994, 2031)

_NAME_WINDOW = 40


@dataclass
class Claim:
    """A number stated in a reply, with the wording immediately around it."""

    value: float
    units: str
    text: str
    context: str
    status: str = "unsourced"
    name: str | None = None
    code_path: str | None = None


@dataclass
class Report:
    matched: int = 0
    contradicted: int = 0
    derived: int = 0
    unsourced: int = 0
    details: list[dict] = field(default_factory=list)

    def as_event(self) -> dict:
        return {
            "matched": self.matched,
            "contradicted": self.contradicted,
            "derived": self.derived,
            "unsourced": self.unsourced,
            "details": self.details,
        }


def extract_claims(text: str) -> list[Claim]:
    """Numbers a reply states as facts, with their unit and surrounding wording.

    Deliberately conservative. Years, clock times, ISO dates, version numbers and small
    bare integers are dropped, because a report where the real problem sits at rank 30
    is a report nobody reads.
    """
    if not text:
        return []
    norm = text.replace("−", "-").replace("≈", "~")
    claims: list[Claim] = []
    for m in _TOKEN.finditer(norm):
        raw, unit = m.group(1), (m.group(2) or "")
        before = norm[m.start() - 1 : m.start()]
        after = norm[m.end() : m.end() + 1]
        if before in ("-", ":", "/", ".") or after in (":", "/", "-"):
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if not unit:
            written_as_int = "." not in raw
            if written_as_int and (abs(value) < _MIN_BARE_INT or int(value) in _YEARS):
                continue
        claims.append(
            Claim(
                value=value,
                units=unit,
                text=f"{raw} {unit}".strip(),
                context=norm[max(0, m.start() - _NAME_WINDOW) : m.end() + 10].replace("\n", " "),
            )
        )
    return claims


def _compatible_units(a: str, b: str) -> bool:
    """Whether two unit strings can describe the same quantity.

    Unknown on either side means "cannot rule it out" — the ledger only carries a unit
    when the code passed one, and a missing unit must not be read as a mismatch.
    """
    if not a or not b:
        return True
    norm = {"cm⁻³": "cm-3", "cm^-3": "cm-3", "/cm3": "cm-3", "R_E": "RE", "Re": "RE"}
    return norm.get(a, a).lower() == norm.get(b, b).lower()


def _named_entry(context: str, claim_units: str, entries: list[dict]) -> dict | None:
    """Ledger entry whose quantity the wording around a number names, if any.

    `compression_ratio_density` is written "density compression ratio" in prose, so the
    words are matched rather than the identifier: the underscores become a word set and
    every word has to be there. Units have to agree as well — the word "downstream" sits
    near both a field and a density, and on its own it accused the wrong one.
    """
    low = context.lower()
    best = None
    for entry in entries:
        name = entry.get("name") or ""
        if not _compatible_units(claim_units, entry.get("units") or ""):
            continue
        words = [w for w in re.split(r"[._\s]+", name.lower()) if len(w) > 2]
        if words and all(w in low for w in words):
            if best is None or len(name) > len(best.get("name") or ""):
                best = entry
    return best


def verify(claims: list[Claim], ledger: dict, rtol: float = 5e-3) -> Report:
    """Give every claim a provenance status against the ledger.

    - `matched` — a recorded value (mean, min, max or std) equals it within `rtol`.
    - `contradicted` — the wording names a recorded quantity but the number is not one of
      its statistics. The strongest signal available here: the value was computed, and
      what got published is something else.
    - `derived` — no match, but a ratio or a percentage, which is normally *computed from*
      recorded values rather than being one. Counted apart so the noise does not bury the
      rest.
    - `unsourced` — a physical quantity with a unit that nothing in the session produced.
    """
    entries = ledger.get("values") or []
    report = Report()

    for claim in claims:
        hits = [
            e
            for e in entries
            for key in ("mean", "min", "max", "std")
            if isinstance(e.get(key), (int, float))
            and abs(e[key] - claim.value) <= rtol * max(abs(claim.value), abs(e[key]), 1e-12)
        ]
        named = _named_entry(claim.context, claim.units, entries)
        if hits:
            claim.status = "matched"
            claim.name = hits[0].get("name")
            claim.code_path = hits[0].get("code_path")
            report.matched += 1
        elif named:
            claim.status = "contradicted"
            claim.name = named.get("name")
            claim.code_path = named.get("code_path")
            report.contradicted += 1
        elif claim.units in ("", "%"):
            claim.status = "derived"
            report.derived += 1
        else:
            claim.status = "unsourced"
            report.unsourced += 1

    flagged = [c for c in claims if c.status in ("contradicted", "unsourced")]
    flagged.sort(key=lambda c: (c.status != "contradicted", not c.units))
    report.details = [
        {"text": c.text, "status": c.status, "name": c.name, "code_path": c.code_path}
        for c in flagged[:12]
    ]
    return report


def check_reply(text: str, session_dir) -> dict | None:
    """Provenance event payload for a reply, or None when there is nothing to say.

    Returns None on an empty ledger: a session that computed nothing (a search, a
    catalogue listing) would otherwise have every number in its answer reported as
    unsourced, which is true and useless.
    """
    ledger = provenance.read_ledger(session_dir)
    if not ledger.get("values"):
        return None
    claims = extract_claims(text)
    if not claims:
        return None
    return verify(claims, ledger).as_event()
