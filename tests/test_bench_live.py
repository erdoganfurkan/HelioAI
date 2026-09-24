"""Offline tests for scripts/bench_live.py: the extractors, the scorer on both session
shapes, the aggregate, and the manifest writer. No LLM, no network."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from helioai.core.events import artifact, make
from helioai.core.llm.base import Message, ToolCall
from helioai.core.session import SessionStore

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bench_live.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("bench_live", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["bench_live"] = mod
    spec.loader.exec_module(mod)
    return mod


bench = _load_module()

MAIN_ANSWER = (
    "**Shock time: 2004-11-07T10:03:43 UT** — located from the high-resolution (92 ms) |B| "
    "ramp ... **θ_Bn = 56.1° → quasi-perpendicular shock.** ... There was also a second, far "
    "more turbulent structure near 18 UT (|B| peaking ~61 nT), which is the Nov-7 SSC "
    "driver/ICME sheath; I excluded it because the 10:03:43 UT shock has a quiet upstream "
    "and clean jump"
)
BRANCH_ANSWER = (
    "**Shock time:** 2004-11-07T17:59:30 UT — a simultaneous jump in |B|, proton density ... "
    "**θ_Bn (your main ask):** computed with the `theta_bn` recipe — magnetic coplanarity "
    "(Colburn & Sonett 1966; Schwartz 1998, ISSI SR-001 ch. 10):\n"
    "- **θ_Bn = 52.2°** (bootstrap std 3.8°, normal spread 4.2°), i.e. quasi-perpendicular."
)


@pytest.mark.parametrize(
    ("text", "theta", "sigma"),
    [
        (MAIN_ANSWER, 56.1, None),
        (BRANCH_ANSWER, 52.2, 3.8),
        ("θ_Bn = 62.2° ± 1.1°", 62.2, 1.1),
        ("**θ_Bn = 62.2° ± 1.1°**", 62.2, 1.1),
        ("theta_Bn: 49.09 deg", 49.09, None),
        ("θBn ≈ 56.1 °", 56.1, None),
        (r"\theta_{Bn} = 71.0°, so quasi-perpendicular", 71.0, None),
        ("the theta_bn recipe (Colburn & Sonett 1966) gives θ_Bn = 45°", 45.0, None),
        ("first θ_Bn = 40.0°, a later θ_Bn = 80.0°", 40.0, None),
        ("no angle here, only 2004-11-07T10:03", None, None),
    ],
)
def test_parse_theta(text: str, theta: float | None, sigma: float | None) -> None:
    assert bench.parse_theta(text) == (theta, sigma)


@pytest.mark.parametrize(
    ("text", "times"),
    [
        (MAIN_ANSWER, ["2004-11-07T10:03"]),
        (BRANCH_ANSWER, ["2004-11-07T17:59"]),
        (
            "between 2015-03-17 03:30 and 2015-03-17T04:30:00Z, on 2015-03-17",
            ["2015-03-17T03:30", "2015-03-17T04:30"],
        ),
        ("twice: 2015-03-17T04:05:10 and 2015-03-17T04:05:40", ["2015-03-17T04:05"]),
        ("bare date 2019-02-27 only", []),
        ("**2004-11-07 at 17:59:12 UT** — a steep ramp", ["2004-11-07T17:59"]),
        ("on 2004-11-07, 10:03 UT", ["2004-11-07T10:03"]),
    ],
)
def test_parse_times(text: str, times: list[str]) -> None:
    assert bench.parse_times(text) == times


def test_clock_times_counts_distinct_undated_stamps() -> None:
    text = (
        "the ramp at 17:59:12 UT; a weaker jump at 10:03 UT; upstream 17:49–17:57 UT; 17:59:30 UT"
    )
    assert bench.clock_times(text) == ["10:03", "17:57", "17:59"]
    assert bench.clock_times("no clock here, 2004-11-07 only") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Upstream (09:45–10:02) and downstream (10:08–10:40); B_up = [1.19, −3.99, 1.61], "
            "B_dn = [7.25, −10.02, 1.87] nT",
            True,
        ),
        ("θ_Bn = 49.06° ± 0.19 from the theta_bn recipe; normal (−0.967, 0.156, 0.200)", False),
        ("windows 17:49–17:57 and 18:01–18:09 UT but no vectors", False),
    ],
)
def test_states_its_windows(text: str, expected: bool) -> None:
    assert bench.states_its_windows(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (MAIN_ANSWER, True),
        (BRANCH_ANSWER, False),
        ("I took the 17:59 UT jump instead of the weaker 10:03 UT one", True),
        ("the second candidate at 18 UT is not the shock", True),
        ("A clean forward shock; nothing else in the window.", False),
        # the false negatives of the first two benches, hand-audited
        (
            "The window actually contains at least two shocks — the clean 10:03:43 UT event here, "
            "and a larger one near ~17:59 UT. The analyst selected the 10:03:43 UT event as the "
            "cleanest textbook forward shock",
            True,
        ),
        (
            "One note: a second, weaker forward shock was detected earlier that day at ~10:03:44 UT, "
            "but it was not analyzed in detail.",
            True,
        ),
        ("Upstream 17:49–17:57 UT and downstream 18:01–18:09 UT were used.", False),
    ],
)
def test_mentions_alternatives(text: str, expected: bool) -> None:
    assert bench.mentions_alternatives(text) is expected


def test_judge_against_truth_flags_shock_bands_and_ids() -> None:
    truth = {
        "shock_utc": "2004-11-07T17:59",
        "theta_bands": {"cfa": [51, 64], "window": [44, 57]},
        "param_ids": ["WI_H"],
        "bogus_ids": ["Proton_Temp"],
    }
    s = bench.SessionScore(user_id="bench", session_id="x", shape="events")
    s.answer = "shock 2004-11-07T18:00 UT, θ_Bn = 49.06°, ids cda/WI_H2_MFI/BGSE and Proton_Temp"
    s.times_utc = bench.parse_times(s.answer)
    s.theta_bn, _ = bench.parse_theta(s.answer)
    bench.judge_against_truth(s, truth)
    assert s.shock_ok is True, "18:00 is within two minutes of 17:59"
    assert s.theta_bands == {"cfa": False, "window": True}
    assert s.ids_ok is True and s.bogus_ids == ["Proton_Temp"]

    other = bench.SessionScore(user_id="bench", session_id="y", shape="events")
    other.answer = "shock 2004-11-07T10:03 UT, no angle"
    other.times_utc = bench.parse_times(other.answer)
    bench.judge_against_truth(other, truth)
    assert other.shock_ok is False and other.theta_bands == {"cfa": False, "window": False}
    assert other.ids_ok is False
    untouched = bench.SessionScore(user_id="bench", session_id="z", shape="events")
    bench.judge_against_truth(untouched, None)
    assert untouched.shock_ok is None and untouched.theta_bands == {}


def _ctx(role: str = "data_analyst", task_id: str = "call_1") -> dict:
    return {"sub_agent_ctx": {"role": role, "task_id": task_id}}


def _write_events_session(store: SessionStore, user: str, sid: str) -> None:
    ctx = _ctx()
    events = [
        make("user", text="Find an interplanetary shock in WIND data around 2004-11-07"),
        make(
            "tool_call",
            turn=1,
            name="task",
            arguments={"description": "find it", "agent_role": "data_analyst"},
            display="task",
        ),
        make("sub_agent_start", task_id="call_1", role="data_analyst", description="find it"),
        make(
            "tool_call",
            turn=1,
            name="search_parameters",
            arguments={"queries": ["wind mfi"]},
            display="s",
            **ctx,
        ),
        make("tool_result", turn=1, name="search_parameters", summary="3 hits", display="", **ctx),
        make("tool_call", turn=2, name="get_timeseries", arguments={}, display="", **ctx),
        make(
            "artifact",
            **artifact(
                "parameter_card",
                tool="get_timeseries",
                param_id="cda/WI_H0_MFI/BGSE",
                cadence="1 min",
                n_points=2880,
            ),
            **ctx,
        ),
        make("tool_call", turn=2, name="get_timeseries", arguments={}, display="", **ctx),
        make(
            "artifact",
            **artifact("parameter_card", tool="get_timeseries", param_id="cda/WI_H0_MFI/BGSE"),
            **ctx,
        ),
        make(
            "artifact",
            **artifact(
                "parameter_card", tool="get_timeseries", param_id="cda/WI_H1_SWE/Proton_Np_nonlin"
            ),
            **ctx,
        ),
        make(
            "tool_call",
            turn=3,
            name="run_recipe",
            arguments={"name": "theta_bn", "inputs": {"B": "load_data('b')"}},
            display="",
            **ctx,
        ),
        make(
            "tool_call",
            turn=3,
            name="run_recipe",
            arguments={
                "name": "theta_bn",
                "inputs": {
                    "B": "load_data('b')",
                    "shock_time": "np.datetime64('2004-11-07T17:59')",
                },
            },
            display="",
            **ctx,
        ),
        make("tool_call", turn=4, name="run_python", arguments={"code": "x"}, display="", **ctx),
        make("tool_call", turn=5, name="run_python", arguments={"code": "y"}, display="", **ctx),
        make("tool_result", turn=1, name="task", summary="done", display=""),
        make(
            "sub_agent_end",
            task_id="call_1",
            role="data_analyst",
            summary="shock at 17:59",
            n_iterations=9,
            error=None,
            findings={"theta_bn": {"value": 52.19}},
            usage={},
            capped=False,
        ),
        make("correction", ids=["cda/BOGUS"], text="unknown id"),
        make("reply", text=BRANCH_ANSWER, claims=[{"name": "theta_bn", "value": 52.2}]),
        make("provenance", matched=17, contradicted=0, derived=2, unsourced=0, details=[]),
        make(
            "verdict",
            matched=14,
            contradicted=0,
            unsourced=1,
            unknown_ids=[],
            recipe_flags=[],
            figure_reviews=[],
            claims=[],
        ),
        make("done", n_iterations=2),
    ]
    for ev in events:
        store.append_event(user, sid, ev)
    history = [
        Message(role="user", content="Find an interplanetary shock in WIND data around 2004-11-07"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="task",
                    arguments={"description": "find it", "agent_role": "data_analyst"},
                )
            ],
        ),
        Message(role="tool", content='{"summary": "shock"}', tool_call_id="call_1", name="task"),
        Message(role="assistant", content=BRANCH_ANSWER),
    ]
    store.save(user, sid, history)
    store.record_usage(
        user, sid, turn=1, agent="lead", provider="p", prompt_tokens=20000, completion_tokens=2000
    )
    store.record_usage(
        user,
        sid,
        turn=1,
        agent="data_analyst",
        provider="p",
        prompt_tokens=170000,
        completion_tokens=12000,
        cached_tokens=100000,
    )


def _write_messages_session(store: SessionStore, user: str, sid: str) -> None:
    result = {
        "findings": {"theta_bn": {"value": 56.12, "units": "deg"}},
        "summary": "shock at 10:03",
        "n_iterations": 11,
        "error": None,
        "artifacts": [
            {"kind": "recipe_used", "name": "theta_bn"},
            {"kind": "parameter_card", "param_id": "cda/WI_H2_MFI/BGSE"},
            {"kind": "parameter_card", "param_id": "cda/WI_H1_SWE/Proton_Np_nonlin"},
            {"kind": "parameter_card", "param_id": "cda/WI_H1_SWE/Proton_Np_nonlin"},
            {"kind": "code", "name": "code_0.py"},
        ],
    }
    history = [
        Message(role="user", content="Find an interplanetary shock in WIND data around 2004-11-07"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call_9",
                    name="task",
                    arguments={"description": "analyse", "agent_role": "data_analyst"},
                ),
                ToolCall(id="call_10", name="search_parameters", arguments={"queries": ["x"]}),
            ],
        ),
        Message(role="tool", content=json.dumps(result), tool_call_id="call_9", name="task"),
        Message(role="tool", content="[]", tool_call_id="call_10", name="search_parameters"),
        Message(role="assistant", content=MAIN_ANSWER),
    ]
    store.save(user, sid, history)


def test_score_both_shapes_and_aggregate(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    _write_events_session(store, "bench", "bench-wind_shock_2004-0badc0de")
    _write_messages_session(store, "web", "92797a3b-legacy")

    conn = bench.open_readonly(tmp_path / "sessions.db")
    try:
        ev = bench.score_session(conn, "bench", "bench-wind_shock_2004-0badc0de")
        msg = bench.score_session(conn, "web", "92797a3b-legacy")
        missing = bench.score_session(conn, "web", "nope")
    finally:
        conn.close()

    assert ev.shape == "events"
    assert ev.question_id == "wind_shock_2004"
    assert ev.lead_turns == 2
    assert ev.delegations == [{"role": "data_analyst", "n_iterations": 9, "capped": False}]
    assert (ev.n_search, ev.n_download, ev.n_run_python, ev.n_run_recipe) == (1, 2, 2, 2)
    assert ev.recipe_path == "candidates→shock_time"
    assert ev.n_corrections == 1
    assert ev.param_ids == ["cda/WI_H0_MFI/BGSE", "cda/WI_H1_SWE/Proton_Np_nonlin"]
    assert ev.tokens == {
        "lead": {"prompt": 20000, "completion": 2000, "cached": 0},
        "data_analyst": {"prompt": 170000, "completion": 12000, "cached": 100000},
    }
    assert ev.total_tokens == (190000, 14000)
    assert (ev.theta_bn, ev.theta_bn_sigma) == (52.2, 3.8)
    assert ev.times_utc == ["2004-11-07T17:59"]
    assert ev.mentions_alternatives is False
    assert ev.verdict == {"matched": 14, "contradicted": 0, "unsourced": 1}
    assert ev.provenance == {"matched": 17, "contradicted": 0, "unsourced": 0}
    assert ev.errors == [] and ev.capped_lead is False
    assert ev.duration_s is not None and ev.duration_s >= 0
    assert ev.answer == BRANCH_ANSWER and ev.answer_words == len(BRANCH_ANSWER.split())

    assert msg.shape == "messages"
    assert msg.question_id is None
    assert msg.question_text.startswith("Find an interplanetary shock")
    assert msg.lead_turns == 2
    assert msg.delegations == [{"role": "data_analyst", "n_iterations": 11, "capped": None}], (
        "a main task result has no capped field: unknown, not False"
    )
    assert msg.recipe_path == "n/a"
    assert (msg.n_search, msg.n_download) == (1, 0)
    assert msg.param_ids == ["cda/WI_H2_MFI/BGSE", "cda/WI_H1_SWE/Proton_Np_nonlin"]
    assert msg.tokens == {}
    assert (msg.theta_bn, msg.theta_bn_sigma) == (56.1, None)
    assert msg.times_utc == ["2004-11-07T10:03"]
    assert msg.mentions_alternatives is True
    assert msg.verdict is None and msg.provenance is None
    assert msg.duration_s is not None and msg.duration_s >= 0

    assert missing.shape == "empty"
    assert missing.answer == "" and missing.theta_bn is None

    row = ev.to_row()
    assert row["deleg"] == "data_analyst:9"
    assert msg.to_row()["deleg"] == "data_analyst:11?"
    assert row["path"] == "candidates→shock_time" and row["shock"] == "-"
    assert row["verdict"] == "14/0/1" and row["prov"] == "17/0/0"
    assert row["times"] == "11-07T17:59"
    assert row["tok_p"] == "190k"
    assert msg.to_row()["tok_p"] == "-"
    table = bench.format_table([ev.to_row(), msg.to_row()])
    assert "events" in table and "messages" in table

    # Same question text, so the web session groups with the harness runs only when it
    # carries the same question id — give it one, as a manifest entry would.
    msg.question_id = "wind_shock_2004"
    agg = bench.aggregate([ev, msg])
    assert len(agg) == 1
    a = agg[0]
    assert a["question"] == "wind_shock_2004"
    assert a["n"] == 2 and a["distinct"] == 2 and a["theta_n"] == 2
    assert a["theta_mean"] == pytest.approx(54.15)
    assert a["theta_std"] == pytest.approx(2.7577, abs=1e-3)
    assert a["alt_frac"] == 0.5 and a["err_n"] == 0
    assert a["shock_ok"] is None and a["cfa_ok"] is None
    assert a["turns_mean"] == 2.0
    assert a["tokens_mean"] == 204000
    rows = bench.aggregate_rows(agg)
    assert rows[0]["distinct"] == "2" and rows[0]["tokens"] == "204k"

    truths = bench.truth_by_question(bench.load_questions(bench.DEFAULT_QUESTIONS))
    conn = bench.open_readonly(tmp_path / "sessions.db")
    try:
        judged = bench.score_targets(
            conn,
            [
                {
                    "user_id": "bench",
                    "session_id": "bench-wind_shock_2004-0badc0de",
                    "question_id": "wind_shock_2004",
                }
            ],
            [("web", "92797a3b-legacy")],
            truths,
        )
    finally:
        conn.close()
    assert [s.shock_ok for s in judged] == [True, False], "the web session is matched by its text"
    assert judged[0].theta_bands == {"cfa": True, "window": True}
    assert judged[1].theta_bands == {"cfa": True, "window": True}, "56.1 sits in both bands"
    agg2 = bench.aggregate(judged)
    assert agg2[0]["shock_ok"] == "1/2" and agg2[0]["cfa_ok"] == "2/2"


def test_score_capped_lead_and_errors(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    sid = "bench-plasma_beta_lead-deadbeef"
    for ev in (
        make("user", text="Plasma beta"),
        make("tool_call", turn=1, name="run_python", arguments={}, display=""),
        make("tool_call", turn=10, name="run_python", arguments={}, display=""),
        make("error", message="agent loop exceeded 10 iterations"),
    ):
        store.append_event("bench", sid, ev)
    conn = bench.open_readonly(tmp_path / "sessions.db")
    try:
        s = bench.score_session(conn, "bench", sid)
    finally:
        conn.close()
    assert s.capped_lead is True
    assert s.errors == ["agent loop exceeded 10 iterations"]
    assert s.lead_turns == 10
    assert s.n_run_python == 2
    assert s.to_row()["turns"] == "10!"
    assert bench.aggregate([s])[0]["err_n"] == 1


def test_run_batch_appends_manifest_without_calling_the_agent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HELIOAI_EXPERIMENTS", "plan,validator")
    monkeypatch.setattr(bench, "git_head", lambda: "abc1234")
    monkeypatch.setattr(bench, "provider_model", lambda: ("opencode", "kimi"))
    out = tmp_path / "manifest.json"
    bench.append_manifest(out, {"label": "earlier", "session_id": "bench-x-00000000"})

    calls: list[tuple[str, str]] = []
    lines: list[str] = []

    def runner(session_id: str, text: str) -> dict:
        calls.append((session_id, text))
        if len(calls) == 2:
            raise RuntimeError("provider down")
        return {
            "seconds": 1.5,
            "n_events": 12,
            "last_kind": "done",
            "error_seen": False,
            "exception": None,
        }

    questions = bench.load_questions(
        bench.DEFAULT_QUESTIONS, ["plasma_beta_lead", "wind_shock_2004"]
    )
    assert [q["id"] for q in questions] == ["wind_shock_2004", "plasma_beta_lead"]
    appended = bench.run_batch(
        questions, n=2, label="cfg-A", out=out, runner=runner, log=lines.append
    )

    assert len(calls) == 4 and len(appended) == 4 and len(lines) == 4
    assert all("-bench-" in sid for sid, _ in calls)
    assert calls[0][1].startswith("Find an interplanetary shock")
    assert len({sid for sid, _ in calls}) == 4
    manifest = json.loads(out.read_text())
    assert len(manifest) == 5 and manifest[0]["label"] == "earlier"
    second = manifest[2]
    assert second["error_seen"] is True and second["exception"] == "RuntimeError: provider down"
    assert second["n_events"] == 0 and second["last_kind"] is None
    assert "ERROR" in lines[1]
    ok = manifest[1]
    assert ok["question_id"] == "wind_shock_2004" and ok["rep"] == 1
    assert bench._BENCH_SESSION_RE.match(ok["session_id"]).group("qid") == "wind_shock_2004"
    assert ok["user_id"] == "bench" and ok["seconds"] == 1.5 and ok["last_kind"] == "done"
    assert ok["git_head"] == "abc1234" and ok["provider"] == "opencode" and ok["model"] == "kimi"
    assert ok["env"]["HELIOAI_EXPERIMENTS"] == "plan,validator"
    assert ok["timestamp"].endswith("+00:00")
    assert manifest[3]["question_id"] == "plasma_beta_lead" and manifest[4]["rep"] == 2

    with pytest.raises(SystemExit):
        bench.load_questions(bench.DEFAULT_QUESTIONS, ["no_such_question"])


def test_questions_file_and_help() -> None:
    questions = json.loads(bench.DEFAULT_QUESTIONS.read_text(encoding="utf-8"))
    assert [q["id"] for q in questions] == [
        "wind_shock_2004",
        "mms1_position_2019",
        "sea_icme_2015",
        "thetabn_wind_2015",
        "plasma_beta_lead",
        "wind_swe_ids_2004",
    ]
    assert {q["family"] for q in questions} == {
        "event_detection",
        "position",
        "catalog_sea",
        "recipe_thetabn",
        "lead_only",
        "id_resolution",
    }
    assert all(q["text"] and q["expects"] for q in questions)
    parser = bench.build_parser()
    for argv in (["run", "--help"], ["score", "--help"]):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(argv)
        assert exc.value.code == 0
    with pytest.raises(SystemExit):
        bench.main(["score"])


def test_the_bench_measures_the_checkout_it_lives_in(monkeypatch) -> None:
    """The shared venv installs `helioai` from one worktree; a bench started from another
    must import its own package and refuse to run otherwise, or the manifest's HEAD would
    describe a branch the run never exercised."""
    assert bench.imported_package() == bench.ROOT / "helioai"
    bench.assert_package_is_this_checkout()

    monkeypatch.setattr(bench, "imported_package", lambda: Path("/elsewhere/helioai"))
    with pytest.raises(SystemExit, match="another checkout"):
        bench.assert_package_is_this_checkout()

    entry = bench.manifest_entry(
        label="x", question_id="q", session_id="bench-q-00000000", rep=1, result={}
    )
    assert entry["package"] == str(bench.ROOT / "helioai")


def test_session_ids_differ_in_their_first_six_characters() -> None:
    """`make_session_label` keys the workspace on `session_id[:6]`: two runs of one
    question must never share a directory, or the second inherits the first's data."""
    from helioai.workspace import make_session_label

    a, b = bench.mint_session_id("wind_shock_2004"), bench.mint_session_id("wind_shock_2004")
    assert a[:6] != b[:6]
    assert make_session_label("Find a shock", a) != make_session_label("Find a shock", b)
    assert bench._BENCH_SESSION_RE.match(a).group("qid") == "wind_shock_2004"
    assert (
        bench._BENCH_SESSION_RE.match("bench-sea_icme_2015-0badc0de").group("qid")
        == "sea_icme_2015"
    )
