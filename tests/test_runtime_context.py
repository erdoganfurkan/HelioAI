"""`RunContext`: the run's user, session and directories as an object, bound in one place."""

from __future__ import annotations

import helioai.workspace as ws
from helioai.runtime.context import RunContext


def test_for_session_derives_the_directory_the_way_get_session_dir_does(tmp_path):
    labelled = RunContext.for_session("alice", "sess-1", label="plot-imf_sess-1")
    bare = RunContext.for_session("alice", "sess-2")
    assert labelled.session_dir == tmp_path / "users" / "alice" / "workspace" / "plot-imf_sess-1"
    assert bare.session_dir == tmp_path / "users" / "alice" / "workspace" / "sess-2"
    assert labelled.data_dir == labelled.session_dir / "data"
    assert labelled.catalogs_dir == tmp_path / "users" / "alice" / "catalogs"
    with labelled.bound():
        assert ws.get_session_dir() == labelled.session_dir


def test_bound_sets_the_three_contextvars_and_resets_them_in_reverse(tmp_path):
    ctx = RunContext.for_session("alice", "s1", label="lbl")
    assert RunContext.current() is None
    with ctx.bound():
        assert ws.current_user() == "alice"
        assert ws._current_session.get() == "s1"
        assert ws._current_label.get() == "lbl"
        assert RunContext.current().session_dir == ctx.session_dir
    assert (ws._current_user.get(), ws._current_session.get(), ws._current_label.get()) == (
        None,
        None,
        None,
    )


def test_a_childs_binding_nested_in_its_parents_hands_the_parents_back(tmp_path):
    """A sub-agent runs inside the lead's run: same user, session and directory, its own
    identity, and the lead's bindings intact when it finishes."""
    lead = RunContext.for_session("alice", "s1", label="lbl")
    child = lead.child(agent="data_analyst", task_id="t1", no_network=True)
    assert child.session_dir == lead.session_dir
    assert (child.agent, child.task_id, child.no_network) == ("data_analyst", "t1", True)
    with lead.bound():
        with child.bound():
            assert ws.get_session_dir() == lead.session_dir
        assert ws.current_user() == "alice" and ws._current_label.get() == "lbl"


def test_a_session_id_that_could_escape_the_workspace_is_sanitised(tmp_path):
    ctx = RunContext.for_session("alice", "../../etc")
    assert ctx.session_dir.is_relative_to(tmp_path / "users" / "alice" / "workspace")


async def test_the_runner_hands_the_writing_tools_their_directories(tmp_path):
    """End to end through the runner: the sandbox gets the session directory, the data
    tools the data directory, the catalogue tool the user's catalogue directory — from the
    context, whatever the contextvars said before the run."""
    from support.scripted import ScriptedLLM, ScriptedRegistry, assistant_calls, assistant_text

    from helioai.core.llm.base import Message, ToolDef
    from helioai.runtime.policies import Policy
    from helioai.runtime.runner import Runner

    names = ("run_python", "get_timeseries", "save_catalog", "list_missions")
    tools = tuple(ToolDef(name=n, description="", parameters={"type": "object"}) for n in names)
    reg = ScriptedRegistry({n: {"ok": True} for n in names})
    llm = ScriptedLLM([assistant_calls(*names), assistant_text("done")])
    ctx = RunContext.for_session("carol", "s9", label="shock_s9")
    runner = Runner(
        Policy(name="t", system_prompt="", tools=tools, max_turns=2), llm, registry=reg, ctx=ctx
    )
    async for _ in runner.run([Message(role="user", content="q")]):
        pass

    by_name = dict(zip(reg.invoked, reg.trusted, strict=True))
    assert by_name["run_python"] == {"_plot_dir": str(ctx.session_dir), "_run_idx": 0}
    assert by_name["get_timeseries"] == {"_data_dir": str(ctx.data_dir)}
    assert by_name["save_catalog"] == {"_catalogs_dir": str(ctx.catalogs_dir)}
    assert by_name["list_missions"] == {}
    assert RunContext.current() is None, "the run's bindings are gone once it ends"
