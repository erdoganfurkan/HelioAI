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

# The decimal separator may be a comma: a model asked in French answers "9,79 nT".
# Accepting only "." did not merely miss those — it started a token *after* the comma,
# so the ledger held 9.79 while the reply seemed to claim 79, and every real measurement
# came back `unsourced`.
_TOKEN = re.compile(rf"(?<![\w.,])(-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)(?:\s*({_UNITS})(?![\w/^]))?")

# A reply that writes any number as `d,d` or `d,dddd` uses the comma as its decimal mark:
# from then on `2,150 km/s` is 2.15, not two thousand. Without that evidence, three digits
# after a comma stay a thousands separator ("1,800 points").
_COMMA_DECIMAL_HINT = re.compile(r"\d,(?:\d{1,2}|\d{4,})(?!\d)")

# A bare integer under this is almost always "3 panels", "2 spacecraft" or a day of the
# month; matching them floods the report with noise that hides the one number that matters.
# Written with a decimal point it is a measurement — "M_A ~ 4.0" is a claim, "4 panels" is
# not. The floor is 32 rather than 10 because a calendar day reaches 31: "the 17 March 2015
# shock" put a bare 17 inside the name window of `compression_ratio` and reported the date as
# contradicting a ratio of 2.59.
_MIN_BARE_INT = 32.0
_YEARS = range(1994, 2031)

_NAME_WINDOW = 40

# An arithmetic expression written out next to the number: "U1 = Vs - Vu ~ 176.98 km/s",
# "dX / Vs ~ 349 s". The operator must be spaced, which is what keeps ISO dates
# (2015-03-17) and ranges (40-10 min) out.
_SHOWN_WORK = re.compile(r"\s[−–—/×·*]\s")


@dataclass
class Claim:
    """A number stated in a reply, with the wording immediately around it."""

    value: float
    units: str
    text: str
    context: str
    pos: int = 0
    status: str = "unsourced"
    name: str | None = None
    code_path: str | None = None


@dataclass
class Report:
    """Tally of how the numbers in an answer relate to the provenance ledger.

    A number is `matched` when a ledger entry backs it, `contradicted` when an
    entry with the same name and unit holds a different value, `derived` when
    the answer states the calculation behind it, and `unsourced` otherwise.
    `details` carries one dict per number, for display.
    """

    matched: int = 0
    contradicted: int = 0
    derived: int = 0
    unsourced: int = 0
    details: list[dict] = field(default_factory=list)

    def as_event(self) -> dict:
        """Flatten the tally into the `provenance` SSE event payload.

        Example:
            >>> Report(matched=3, derived=1).as_event()
            {'matched': 3, 'contradicted': 0, 'derived': 1, 'unsourced': 0, ...}
        """
        return {
            "matched": self.matched,
            "contradicted": self.contradicted,
            "derived": self.derived,
            "unsourced": self.unsourced,
            "details": self.details,
        }


def _normalise_decimal(raw: str, comma_is_decimal: bool = False) -> str:
    """Return `raw` with a decimal comma turned into a point.

    Exactly three digits after the comma is a thousands separator — "1,800 points" —
    and reading that as 1.8 would be a worse error than the one being fixed. Two things
    override that: a leading zero (`0,657` is never six hundred and fifty-seven), and a
    reply that has already shown it writes decimals with a comma. `-0,657 nT` read as
    -657 nT was the audit's example; `Bz` never reaches that value, so the check went
    on believing a fabricated number.
    """
    if "," not in raw:
        return raw
    head, _, tail = raw.rpartition(",")
    if len(tail) == 3 and not comma_is_decimal and head.lstrip("-") != "0":
        return head + tail
    return head + "." + tail


def extract_claims(text: str) -> list[Claim]:
    """Numbers a reply states as facts, with their unit and surrounding wording.

    Deliberately conservative. Years, clock times, ISO dates, version numbers and small
    bare integers are dropped, because a report where the real problem sits at rank 30
    is a report nobody reads.

    Args:
        text: The reply, in whatever language the user asked in. Decimal commas
            are handled: a French answer writing `9,79 nT` once yielded the
            number 79, which made the whole check blind without failing a test.

    Returns:
        One claim per number kept, carrying its unit and the words around it.
    """
    if not text:
        return []
    norm = text.replace("−", "-").replace("≈", "~")
    comma_is_decimal = bool(_COMMA_DECIMAL_HINT.search(norm))
    claims: list[Claim] = []
    for m in _TOKEN.finditer(norm):
        raw, unit = m.group(1), (m.group(2) or "")
        before = norm[m.start() - 1 : m.start()]
        after = norm[m.end() : m.end() + 1]
        if before in ("-", ":", "/", ".") or after in (":", "/", "-"):
            continue
        raw = _normalise_decimal(raw, comma_is_decimal)
        try:
            value = float(raw)
        except ValueError:
            continue
        if not unit:
            written_as_int = "." not in raw and "e" not in raw.lower()
            if written_as_int and (abs(value) < _MIN_BARE_INT or int(value) in _YEARS):
                continue
        # Sliced from the original text, not from `norm`: both replacements are
        # character-for-character, so the offsets hold, and the real minus sign is what
        # tells a subtraction from a hyphen.
        start = max(0, m.start() - _NAME_WINDOW)
        context = text[start : m.end() + 10].replace("\n", " ")
        claims.append(
            Claim(
                value=value,
                units=unit,
                text=f"{raw} {unit}".strip(),
                context=context,
                pos=m.start() - start,
            )
        )
    return claims


_UNIT_ALIASES = {
    "cm⁻³": "cm-3",
    "cm^-3": "cm-3",
    "/cm3": "cm-3",
    "R_E": "RE",
    "Re": "RE",
    # A recipe exports "deg"; a reply writes "62.68°". Same unit — and with unit
    # agreement now deciding whether a hit counts, missing this alias read the demo's
    # one number as unsourced.
    "°": "deg",
}


def _units_conflict(claim_units: str, entry_units: str) -> bool:
    """Whether a claim and an entry name two different quantities outright.

    Only a stated disagreement counts. Most exports carry no unit at all — the model
    writes it into the name and leaves the argument empty — and treating those as
    incompatible with any claim that has one would silence the check on the majority
    of real sessions. What must never happen is the opposite: a field claimed in nT
    being vouched for by a density recorded in cm-3 because the two numbers coincide.
    """
    if not claim_units or not entry_units:
        return False
    return not _same_unit(claim_units, entry_units)


def _same_unit(a: str, b: str) -> bool:
    """Whether two unit strings denote the same quantity. Unitless matches only unitless.

    Accusing a reply of contradicting the ledger is the strongest thing this module says,
    so it demands agreement rather than absence of disagreement. Treating an unknown unit
    as compatible made "upstream: 40 to 10 minutes before the shock" contradict a field in
    nT four times over, because the bare window bounds sit next to the word "upstream" —
    every contradiction of the first real run was that same false positive. Unitless still
    matches unitless, which is what keeps a compression ratio checkable.
    """
    return _UNIT_ALIASES.get(a, a).lower() == _UNIT_ALIASES.get(b, b).lower()


def _named_entry(
    context: str, claim_units: str, entries: list[dict], pos: int | None = None
) -> dict | None:
    """Ledger entry whose quantity the wording around a number names, if any.

    `compression_ratio_density` is written "density compression ratio" in prose, so the
    words are matched rather than the identifier: the underscores become a word set and
    every word has to be there. Units have to agree as well — the word "downstream" sits
    near both a field and a density, and on its own it accused the wrong one.

    Two rules keep the accusation honest, both written after a run where every one of
    fifteen contradictions was false:

    - **Whole words, not substrings.** `wind_Np_upstream_window_n` matched its own word
      "wind" inside "window", so the entry vouched for itself in any sentence mentioning
      an upstream window.
    - **A unitless entry needs two words.** Agreement on a real unit is the second signal
      that lets one word suffice: "downstream" plus nT does single out `B_downstream`.
      When the export carried no unit — which is most of them, because the model writes
      the unit into the name and leaves the argument empty — unitless matches unitless
      for free and the wording is all that is left. `min_Bz_nT` then reduces to "min"
      alone (the length filter drops `bz` and `nt`, the two parts that identify it), and
      "min" occurs in "minimum-variance", "30-min medians" and "Sonnerup & Cahill 1967":
      a citation year was reported as contradicting the minimum Bz.

    When several entries qualify, the one written **closest to the number** wins, `pos`
    being the number's offset in `context`. The rule used to be the longest name, which is
    arbitrary about what the sentence is actually saying — and it decides the verdict now
    that only a scalar entry can support a contradiction: whichever of a vector and a
    scalar is picked out of the same window is the difference between an accusation and
    none. Name length only breaks ties, which is what `pos=None` falls back to. Entries
    that tie on both — the same quantity exported again by a later run — resolve to the
    most recent one, as `provenance.find_value` does.
    """
    low = context.lower()
    best = None
    best_key: tuple[int, int, int] | None = None
    for idx, entry in enumerate(entries):
        name = entry.get("name") or ""
        entry_units = entry.get("units") or ""
        if not _same_unit(claim_units, entry_units):
            continue
        words = [w for w in re.split(r"[._\s]+", name.lower()) if len(w) > 2]
        if not words or (not entry_units and len(words) < 2):
            continue
        starts = [[m.start() for m in re.finditer(rf"\b{re.escape(w)}\b", low)] for w in words]
        if not all(starts):
            continue
        distance = 0 if pos is None else min(abs(s - pos) for occ in starts for s in occ)
        key = (distance, -len(name), -idx)
        if best_key is None or key < best_key:
            best, best_key = entry, key
    return best


def _n_values(entry: dict) -> int | None:
    """How many numbers an entry summarises, or None if the ledger predates `shape`.

    None means "unknown", and every caller here reads it as "assume the old behaviour":
    a ledger written before shape was recorded must not be reinterpreted after the fact.
    """
    shape = entry.get("shape")
    if not isinstance(shape, list) or not all(isinstance(d, int) for d in shape):
        return None
    n = 1
    for d in shape:
        n *= d
    return n


def _is_scalar(entry: dict) -> bool:
    """Whether the entry holds a single value, so that a different number contradicts it."""
    n = _n_values(entry)
    return n is None or n <= 1


def _whole_array(entry: dict) -> list:
    """The entry's values when `sample` holds all of them, else empty.

    `export()` keeps the first eight flattened values, so this is the whole array exactly
    for the short vectors that matter here — a shock normal, a mean field, a 3-component
    velocity. Longer arrays fall back to their four statistics.
    """
    sample = entry.get("sample")
    n = _n_values(entry)
    if not isinstance(sample, list) or n is None:
        return []
    return sample if 0 < n <= len(sample) else []


def _states(entry: dict, value: float, rtol: float) -> bool:
    """Whether a ledger entry holds this number: one of its four statistics, or — for a
    short vector the ledger stores whole — one of its components.

    Magnitudes are compared: the ledger keeps the sign of what was computed, while a reply
    quotes the size of it — "a 464.5 s lag" for a recorded -464.5. Reporting that as
    unsourced was wrong, and this module answers where a number came from, not whether its
    sign is right.

    Components count because a reply that prints `n = [-0.661, -0.657, 0.362]` is quoting
    the export three times over, and min/max alone vouched for two of the three. The rule
    is deliberately narrow: only when `sample` holds every value of the array, never a
    truncated one, so a long time series is still judged on its statistics.
    """

    def close(v: float) -> bool:
        # A positive claim may quote the magnitude of a negative record; a negative
        # claim against a positive record is not a magnitude, it is the wrong sign.
        if value < 0 and v > 0:
            return False
        return abs(abs(v) - abs(value)) <= rtol * max(abs(value), abs(v), 1e-12)

    for key in ("mean", "min", "max", "std"):
        v = entry.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and close(v):
            return True
    return any(
        isinstance(v, (int, float)) and not isinstance(v, bool) and close(v)
        for v in _whole_array(entry)
    )


def _within_rounding(entry: dict, value: float) -> bool:
    """Whether the claim is the entry's mean rounded to the digits the claim shows, or —
    for an entry that summarises a series — within its spread. `2.5` states a recorded
    `2.519`; `60` states `59.95`; a mean quoted inside one standard deviation of a time
    series is not a different number. The sign rule is `_states`': a negative claim is
    not the rounding of a positive record.

    Shared by the prose check, for a number whose wording names a recorded scalar, and
    by the claims validator, so one answer is held to one rounding rule."""
    mean = entry.get("mean")
    if not isinstance(mean, (int, float)) or isinstance(mean, bool):
        return False
    if value < 0 and mean > 0:
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


# Relative tolerance for calling a stated number equal to a recorded one: wide enough
# for the rounding a reply does (2.59 for 2.5848), narrow enough that a different result
# is a different number. Shared with the claims judged by name (`runtime.validator`), so
# one answer is held to one rule.
RTOL = 5e-3


def verify(claims: list[Claim], ledger: dict, rtol: float = RTOL) -> Report:
    """Give every claim a provenance status against the ledger.

    - `matched` — a recorded value (mean, min, max or std) equals it within `rtol`, from
      an entry whose unit does not contradict the claim's. When the wording names a
      recorded scalar, only that entry can source the number: a density of 25 cm-3 must
      not vouch for "B downstream = 25 nT". That entry also sources its value rounded to
      the digits the reply shows (`_within_rounding`): "std 1.5 nT" for a recorded
      1.5153 is the number, not a contradiction of it.
    - `contradicted` — the wording names a recorded **scalar** and the number is not it.
      The strongest signal available here: the value was computed, and what got published
      is something else. An entry holding several values cannot support the accusation —
      `B_up = [-2.33, -0.40, 9.38] nT` legitimately states numbers that are neither the
      mean, the min, the max nor the std of that vector. Nor can a quantity the session
      exported more than once accuse a number one of its runs produced: a preliminary
      `compression_ratio` of 3.045 and the final 2.538 were both computed, and each run
      sources the value it gave. What the accusation says is that no run produced it.
    - `derived` — no match, but either a ratio or a percentage, or a number the reply
      spells out the arithmetic for ("U1 = Vs - Vu ~ 176.98 km/s"). Both are computed
      *from* recorded values rather than being one, and a reader can follow them. Counted
      apart so the noise does not bury the rest.
    - `unsourced` — a physical quantity with a unit that nothing in the session produced.

    Args:
        claims: Output of `extract_claims`.
        ledger: Output of `provenance.read_ledger`. A ledger written before
            `shape` was recorded is treated as it was before, so an old session
            keeps its previous verdicts.
        rtol: Relative tolerance for calling a claim equal to a recorded value.

    Returns:
        A `Report` counting each status and naming the claims behind it.
    """
    entries = ledger.get("values") or []
    report = Report()

    for claim in claims:
        named = _named_entry(claim.context, claim.units, entries, claim.pos)
        # The quantity the sentence names is judged first: when the wording points at a
        # recorded scalar, that quantity alone decides, and a coincidence with some other
        # number in the ledger cannot rescue a value the named quantity does not hold.
        if named and _is_scalar(named):
            hits = [
                e
                for e in reversed(entries)
                if e.get("name") == named.get("name")
                and (_states(e, claim.value, rtol) or _within_rounding(e, claim.value))
            ]
        else:
            hits = [
                e
                for e in entries
                if not _units_conflict(claim.units, e.get("units") or "")
                and _states(e, claim.value, rtol)
            ]
        if hits:
            claim.status = "matched"
            claim.name = hits[0].get("name")
            claim.code_path = hits[0].get("code_path")
            report.matched += 1
        elif named and _is_scalar(named):
            claim.status = "contradicted"
            claim.name = named.get("name")
            claim.code_path = named.get("code_path")
            report.contradicted += 1
        elif claim.units in ("", "%") or _SHOWN_WORK.search(claim.context):
            claim.status = "derived"
            report.derived += 1
        else:
            claim.status = "unsourced"
            report.unsourced += 1

    flagged = [c for c in claims if c.status in ("contradicted", "unsourced")]
    flagged.sort(key=lambda c: (c.status != "contradicted", not c.units))
    report.details = [
        {
            "text": c.text,
            "value": c.value,
            "status": c.status,
            "name": c.name,
            "code_path": c.code_path,
        }
        for c in flagged[:12]
    ]
    return report


def check_reply(text: str, session_dir) -> dict | None:
    """Provenance event payload for a reply, or None when there is nothing to say.

    Returns None on an empty ledger: a session that computed nothing (a search, a
    catalogue listing) would otherwise have every number in its answer reported as
    unsourced, which is true and useless.

    Args:
        text: The finished reply.
        session_dir: Session workspace whose ledger to check against.

    Returns:
        The event payload, or None when the ledger is empty.
    """
    ledger = provenance.read_ledger(session_dir)
    if not ledger.get("values"):
        return None
    claims = extract_claims(text)
    if not claims:
        return None
    return verify(claims, ledger).as_event()
