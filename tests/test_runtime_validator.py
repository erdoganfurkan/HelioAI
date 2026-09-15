"""The one verdict on a finished answer — claims judged by name, unit-aware."""

from __future__ import annotations

import json

import pytest

from helioai import provenance
from helioai.runtime.validator import Verdict, judge_claim, validate


def _entry(name, mean, units="", *, n=1, std=0.0, code_path="/w/code_0.py", **extra):
    e = {
        "name": name,
        "mean": mean,
        "min": mean,
        "max": mean,
        "std": std,
        "units": units,
        "shape": [] if n == 1 else [n],
        "code_path": code_path,
    }
    e.update(extra)
    return e


LEDGER = [
    _entry("theta_bn", 57.16, "deg"),
    _entry("compression_ratio", 3.045),
    _entry("compression_ratio", 2.519, code_path="/w/code_2.py"),
    _entry("Bmag_up", 10.077, "nT", n=5217, std=0.4),
    _entry("shock_normal_gsm", -0.3377, "", n=3, sample=[-0.806, -0.509, 0.302]),
]


def _claim(name, value, units="", source=None):
    return {"name": name, "value": value, "units": units, "source": source or name}


# ── the healthy case first ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("claim", "why"),
    [
        (_claim("theta_bn", 57.16, "deg"), "the exact value"),
        (_claim("theta_bn", 57.2, "deg"), "rounded to the digits the answer shows"),
        (_claim("theta_bn", 57, "deg"), "rounded to a whole degree"),
        (_claim("theta_bn", 0.9976, "rad", "theta_bn"), "the same angle in radians"),
        (_claim("r_B", 2.5, source="compression_ratio"), "named through its source export"),
        (_claim("r_B", 3.05, source="compression_ratio"), "an earlier run of the same export"),
        (_claim("B_up", 10.1, "nT", "Bmag_up"), "a series quoted by its rounded mean"),
        (_claim("B_up", 0.0101, "uT", "Bmag_up"), "the same field in microtesla"),
        (_claim("B_up", 10.4, "nT", "Bmag_up"), "within one standard deviation of a series"),
        (_claim("n_y", -0.509, source="shock_normal_gsm"), "one component of a short vector"),
    ],
)
def test_a_correct_claim_is_matched(claim, why):
    status, detail = judge_claim(claim, LEDGER)
    assert status == "matched", why
    assert detail["ledger"] is not None and detail["code_path"]


# ── then what is accused, and what never is ────────────────────────────────────────


def test_a_named_scalar_that_holds_another_value_is_contradicted():
    status, detail = judge_claim(_claim("theta_bn", 62.7, "deg"), LEDGER)
    assert status == "contradicted"
    assert detail["ledger"] == 57.16 and detail["ledger_units"] == "deg"


def test_incompatible_units_on_a_named_scalar_are_a_contradiction():
    status, _ = judge_claim(_claim("theta_bn", 57.2, "km/s"), LEDGER)
    assert status == "contradicted"


def test_units_the_ledger_spells_unparseably_leave_the_claim_unjudged():
    ledger = [_entry("n_p", 48.5, "#/cc")]
    status, detail = judge_claim(_claim("n_p", 48.5, "cm-3"), ledger)
    assert status == "unsourced" and "reconciled" in detail["note"]


def test_a_series_cannot_accuse_a_number_it_does_not_hold():
    """An entry summarising thousands of values legitimately has numbers stated about it
    that are none of its statistics; only a recorded scalar can contradict."""
    status, _ = judge_claim(_claim("B_up", 30.0, "nT", "Bmag_up"), LEDGER)
    assert status == "unsourced"


@pytest.mark.parametrize("source", ["asserted", "literature"])
def test_a_claim_the_model_did_not_attribute_to_the_session_is_never_contradicted(source):
    status, detail = judge_claim(_claim("theta_bn", 99.0, "deg", source), LEDGER)
    assert status == "unsourced" and detail["source"] == source


def test_an_export_the_session_never_wrote_is_unsourced():
    status, _ = judge_claim(_claim("V_sw", 512, "km/s"), LEDGER)
    assert status == "unsourced"


def test_a_value_that_is_not_a_number_is_unsourced_not_a_crash():
    status, _ = judge_claim(_claim("theta_bn", "about sixty", "deg"), LEDGER)
    assert status == "unsourced"


# ── validate: the four checks and the prose net, in one call ──────────────────────


def _write_ledger(session_dir, entries):
    ledger = provenance._ledger_path(session_dir)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps({"values": entries}), encoding="utf-8")
    assert provenance.read_ledger(session_dir)["values"]


def test_validate_judges_claims_and_keeps_the_prose_net(tmp_path, monkeypatch):
    monkeypatch.setattr("helioai.tools.rag.unknown_ids", lambda ids: [])
    monkeypatch.setattr("helioai.tools.rag.extract_ids", lambda text: [])
    _write_ledger(tmp_path, LEDGER)
    text = "θ_Bn = 57.2° and r_B = 2.9; the sheath field reached 24.5 nT."
    claims = [
        _claim("theta_bn", 57.2, "deg"),
        _claim("r_B", 2.9, source="compression_ratio"),
        _claim("Dst", -223, "nT", "literature"),
    ]

    annotated, verdict = validate(
        text, claims, history=[], artifacts=[], session_dir=tmp_path, figure_reviews=["OK"]
    )

    assert annotated == text
    assert [c["name"] for c in verdict.matched] == ["theta_bn"]
    assert [c["name"] for c in verdict.contradicted] == ["r_B"]
    assert [c["name"] for c in verdict.unsourced] == ["Dst"]
    assert verdict.figure_reviews == ["OK"] and verdict.unknown_ids == []
    assert verdict.prose is not None and verdict.prose["unsourced"] >= 1, "24.5 nT is unclaimed"
    event = verdict.as_event()
    assert (event["matched"], event["contradicted"], event["unsourced"]) == (1, 1, 1)
    assert [c["status"] for c in event["claims"]] == ["contradicted", "unsourced", "matched"]


def test_validate_without_claims_is_the_old_checks_and_nothing_more(tmp_path, monkeypatch):
    monkeypatch.setattr("helioai.tools.rag.unknown_ids", lambda ids: ["cda/BOGUS/x"])
    monkeypatch.setattr("helioai.tools.rag.extract_ids", lambda text: ["cda/BOGUS/x"])
    annotated, verdict = validate(
        "use cda/BOGUS/x", [], history=[], artifacts=[], session_dir=tmp_path
    )
    assert "cda/BOGUS/x" in annotated and len(annotated) > len("use cda/BOGUS/x")
    assert verdict.unknown_ids == ["cda/BOGUS/x"]
    assert not verdict.has_claims and verdict.prose is None


def test_verdict_event_carries_the_contract_keys():
    from helioai.core import events

    v = Verdict(matched=[{"name": "a", "value": 1}], unknown_ids=["x"])
    ev = events.make("verdict", **v.as_event())
    assert ev["data"]["matched"] == 1 and ev["data"]["unknown_ids"] == ["x"]


async def test_the_lead_follows_a_claimed_reply_with_one_verdict_and_journals_it(
    monkeypatch, tmp_path
):
    """End to end through `stream_chat`: the answer closes with `final_answer`, its claims
    are judged against the session's own ledger, and the `verdict` lands in the journal
    after the `reply`. A prose answer produces no verdict at all."""
    from helioai import workspace
    from helioai.config import settings
    from helioai.core import agent_loop
    from helioai.core.llm.base import Message, ToolCall
    from helioai.core.session import SessionStore
    from tests.support.scripted import ScriptedLLM

    monkeypatch.setattr(settings.agent, "experiments", frozenset({"final_answer"}))
    monkeypatch.setattr("helioai.tools.rag.unknown_ids", lambda ids: [])
    monkeypatch.setattr("helioai.tools.rag.extract_ids", lambda text: [])
    test_store = SessionStore(tmp_path / "sessions.db")
    monkeypatch.setattr(agent_loop, "store", test_store)
    test_store.save("web", "s-verdict", [])
    test_store.set_workspace_dir("web", "s-verdict", "verdict-run")
    _write_ledger(workspace.session_dir_for("web", "s-verdict", "verdict-run"), LEDGER)

    def closing(answer, claims):
        return Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="fa", name="final_answer", arguments={"answer": answer, "claims": claims}
                )
            ],
        )

    llm = ScriptedLLM(
        [
            closing(
                "θ_Bn = 57.2° and r_B = 2.9.",
                [_claim("theta_bn", 57.2, "deg"), _claim("r_B", 2.9, source="compression_ratio")],
            ),
            Message(role="assistant", content="Nothing numeric to say."),
        ]
    )

    live = [ev async for ev in agent_loop.stream_chat(llm, "web", "s-verdict", "q")]
    kinds = [e["event"] for e in live]
    assert kinds.index("reply") < kinds.index("verdict") < kinds.index("done")
    verdict = next(e for e in live if e["event"] == "verdict")["data"]
    assert (verdict["matched"], verdict["contradicted"], verdict["unsourced"]) == (1, 1, 0)
    assert verdict["claims"][0]["name"] == "r_B" and verdict["claims"][0]["ledger"] == 2.519
    journaled = [e for e in test_store.events("web", "s-verdict") if e["event"] == "verdict"]
    assert journaled == [next(e for e in live if e["event"] == "verdict")]

    prose = [ev async for ev in agent_loop.stream_chat(llm, "web", "s-verdict", "again")]
    assert "verdict" not in [e["event"] for e in prose]


# ── what the first live run taught (2026-09-14, a876b27) ──────────────────────────


def test_a_unitless_claim_pointing_at_a_dimensioned_export_is_not_accused():
    """The model named the shock normal's components with `source: theta_bn` — the run
    that printed them — and no units. The ledger's `theta_bn` is the angle, 54.85 deg.
    'n_x stated -0.509, the session computed 54.85 deg' is not a contradiction anyone
    can act on: without units the claim may be another quantity, and it is."""
    ledger = [_entry("theta_bn", 54.85, "deg")]
    status, detail = judge_claim(_claim("shock normal n_x (GSM)", -0.509, "", "theta_bn"), ledger)
    assert status == "unsourced"
    assert "units" in detail["note"] and "54.85 deg" in detail["note"]


def test_a_unitless_claim_that_states_the_recorded_value_still_matches():
    ledger = [_entry("theta_bn", 54.85, "deg")]
    status, _ = judge_claim(_claim("theta_bn", 54.9, "", "theta_bn"), ledger)
    assert status == "matched"


def test_the_claims_tolerance_is_the_prose_checkers():
    """2.59 for a recorded 2.5848 (0.2 % off, a rounding slip) was traced by the
    `provenance` line and contradicted by the `verdict` line of the same answer. One
    answer, one tolerance."""
    from helioai.core.provenance_check import RTOL, verify

    assert verify.__defaults__ == (RTOL,) == (5e-3,)
    ledger = [_entry("compression_ratio", 2.5848101937326744)]
    status, _ = judge_claim(_claim("compression_ratio", 2.59), ledger)
    assert status == "matched"
    status, _ = judge_claim(_claim("compression_ratio", 2.61), ledger)
    assert status == "contradicted", "0.5 % is still the line"


# ── what the SEA live run taught (2026-09-15, 099f5c2) ─────────────────────────────


@pytest.mark.parametrize(
    "label", ["dimensionless (normalized epoch)", "events per epoch point", "events", "ratio"]
)
def test_a_label_on_a_dimensionless_export_is_not_an_irreconcilable_unit(label):
    """`peak_tau = 0.606, units: "dimensionless (normalized epoch)"` against an export
    recorded with "" came back unsourced: the label does not parse as a unit. A word on a
    dimensionless quantity is a label, not a dimension; the value is what to compare."""
    ledger = [_entry("peak_tau", 0.6060606)]
    status, _ = judge_claim(_claim("peak_tau", 0.606, label, "peak_tau"), ledger)
    assert status == "matched"
    status, _ = judge_claim(_claim("peak_tau", 0.9, label, "peak_tau"), ledger)
    assert status == "contradicted", "the label does not shield a wrong number either"


def test_a_blank_ledger_unit_means_unknown_so_the_value_alone_is_judged():
    """Most `run_python` exports carry no unit; a claim that names one is more specific
    than the ledger, not wrong. Only two units that both parse can be incompatible."""
    ledger = [_entry("Bmag_up", 9.7)]
    status, _ = judge_claim(_claim("B_up", 9.7, "nT", "Bmag_up"), ledger)
    assert status == "matched"
    status, _ = judge_claim(_claim("B_up", 12.0, "nT", "Bmag_up"), ledger)
    assert status == "contradicted"
    ledger = [_entry("theta_bn", 57.16, "deg")]
    status, _ = judge_claim(_claim("theta_bn", 57.2, "degre", "theta_bn"), ledger)
    assert status == "unsourced", "a misspelling against a real unit is left unjudged"


# ── what the MMS1 web run taught (2026-09-15, 355d5a6) ─────────────────────────────


def test_a_number_a_claim_covers_is_not_rejudged_by_the_prose_regex(tmp_path, monkeypatch):
    """The live reply read "X = 21.36 R_E, Y = −16.11 R_E, Z = 5.12 R_E …". The claims
    matched each by name; the regex, reading the wording, contradicted 21.36 against the
    Z export. One number, one verdict: the claim's. A number with no claim at all (the
    37° here) is still the regex's to judge."""
    monkeypatch.setattr("helioai.tools.rag.unknown_ids", lambda ids: [])
    monkeypatch.setattr("helioai.tools.rag.extract_ids", lambda text: [])
    ledger = [
        _entry("MMS1_X_GSM_Re", 21.363887, "Re"),
        _entry("MMS1_Y_GSM_Re", -16.106245, "Re"),
        _entry("MMS1_Z_GSM_Re", 5.117744, "Re"),
    ]
    _write_ledger(tmp_path, ledger)
    text = (
        "MMS1 was located in GSM at X = 21.36 R_E, Y = −16.11 R_E, Z = 5.12 R_E "
        "(~37 ° off the Sun–Earth line); the Shue standoff was 7.96 R_E."
    )
    claims = [
        _claim("MMS1_X_GSM", 21.363887, "Re", "MMS1_X_GSM_Re"),
        _claim("MMS1_Y_GSM", -16.106245, "Re", "MMS1_Y_GSM_Re"),
        _claim("MMS1_Z_GSM", 5.117744, "Re", "MMS1_Z_GSM_Re"),
        _claim("magnetopause_standoff", 7.96, "Re", "asserted"),
    ]
    _, verdict = validate(text, claims, history=[], artifacts=[], session_dir=tmp_path)
    assert len(verdict.matched) == 3 and verdict.contradicted == []
    assert [c["name"] for c in verdict.unsourced] == ["magnetopause_standoff"]
    prose = verdict.prose
    assert prose is not None and prose["contradicted"] == 0, prose
    texts = [d["text"] for d in prose["details"]]
    assert not any(n in t for t in texts for n in ("21.36", "16.11", "5.12", "7.96")), texts
    assert any("37" in t for t in texts), "a number without a claim is still the regex's"


def test_the_prose_counts_shrink_by_exactly_the_numbers_the_claims_took(tmp_path, monkeypatch):
    """The regex contradicts 21.36 against MMS1_Z_GSM_Re when the wording puts "Z" next
    to it; the claim for 21.36 names MMS1_X_GSM_Re. The report must lose that
    contradiction, not just hide its detail line."""
    monkeypatch.setattr("helioai.tools.rag.unknown_ids", lambda ids: [])
    monkeypatch.setattr("helioai.tools.rag.extract_ids", lambda text: [])
    monkeypatch.setattr(
        "helioai.runtime.validator._prose_report",
        lambda text, sd: {
            "matched": 2,
            "contradicted": 1,
            "derived": 0,
            "unsourced": 1,
            "details": [
                {
                    "text": "21.36 R_E",
                    "value": 21.36,
                    "status": "contradicted",
                    "name": "MMS1_Z_GSM_Re",
                },
                {"text": "7.96 R_E", "value": 7.96, "status": "unsourced", "name": None},
            ],
        },
    )
    _write_ledger(tmp_path, [_entry("MMS1_X_GSM_Re", 21.363887, "Re")])
    claims = [_claim("MMS1_X_GSM", 21.36, "Re", "MMS1_X_GSM_Re")]
    _, verdict = validate("…", claims, history=[], artifacts=[], session_dir=tmp_path)
    assert [c["name"] for c in verdict.matched] == ["MMS1_X_GSM"]
    assert verdict.prose == {
        "matched": 2,
        "contradicted": 0,
        "derived": 0,
        "unsourced": 1,
        "details": [{"text": "7.96 R_E", "value": 7.96, "status": "unsourced", "name": None}],
    }
