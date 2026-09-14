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
    from helioai.core import agent_loop
    from helioai.core.llm.base import Message, ToolCall
    from helioai.core.session import SessionStore
    from tests.support.scripted import ScriptedLLM

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
