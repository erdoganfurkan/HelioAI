"""`run_recipe(name, inputs)`: the shipped recipe runs as shipped, on the session's data.

The healthy case first, in the real sandbox and on the shapes the data actually has —
a Wind SWE scalar arrives as `(N, 1)`, not `(N,)` — then what the tool refuses, and how
the rest of the runtime reads its result: the artifacts it yields, the exemption the
recipe check grants it, the line the interfaces show.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from helioai.config import _PKG_RECIPES
from helioai.core.llm.base import Message, ToolCall
from helioai.core.tool_exec import _extract_artifact, _flag_recipe_bypass, trusted_args
from helioai.tools import recipes as _rcp
from helioai.tools.recipes import recipe_script, run_recipe
from helioai.tools.results import ToolResult


@pytest.fixture
def recipes_dir(monkeypatch):
    from helioai.config import settings

    monkeypatch.setattr(settings.recipes, "recipes_dir", _PKG_RECIPES)
    return _PKG_RECIPES


def _save(data_dir: Path, name: str, t, values, units: str, columns: list[str]) -> None:
    data_dir.mkdir(exist_ok=True)
    np.savez_compressed(data_dir / f"{name}.npz", time=t, values=values)
    mfile = data_dir / "manifest.json"
    manifest = json.loads(mfile.read_text()) if mfile.exists() else {"datasets": {}}
    manifest["datasets"][name] = {
        "kind": "timeseries",
        "file": f"{name}.npz",
        "param_id": f"test/{name}",
        "units": units,
        "columns": columns,
        "source": "test",
        "created": "0",
    }
    mfile.write_text(json.dumps(manifest), encoding="utf-8")


# ── the healthy case, in the sandbox ────────────────────────────────────────────────


async def test_theta_bn_runs_verbatim_on_two_windows_of_a_downloaded_field(tmp_path, recipes_dir):
    """The run block of theta_bn reads B_up / B_dn when they are defined: bound as inputs,
    the recipe computes the angle of the data — 60° here — and exports it itself."""
    n = 40
    bx, bz = 10 * np.cos(np.radians(60)), 10 * np.sin(np.radians(60))
    t = np.datetime64("2015-03-17T03:50:00", "s") + np.arange(n) * np.timedelta64(30, "s")
    values = np.tile([bx, 0.0, bz], (n, 1))
    values[n // 2 :, 2] *= 2.5
    values[5] = np.nan
    _save(tmp_path / "data", "b", t, values, "nT", ["Bx", "By", "Bz"])

    result = await run_recipe(
        "theta_bn",
        inputs={
            "B_up": f"load_data('b').values[:{n // 2}]",
            "B_dn": f"load_data('b').values[{n // 2}:]",
        },
        _plot_dir=str(tmp_path),
        _run_idx=0,
    )

    assert result.get("error") is None, result.get("stderr", "")
    assert result["exports"]["theta_bn"]["mean"] == pytest.approx(60.0, abs=0.01)
    assert result["recipe"]["name"] == "theta_bn" and "Colburn" in result["recipe"]["reference"]
    assert result["inputs"] == {
        "B_up": f"load_data('b').values[:{n // 2}]",
        "B_dn": f"load_data('b').values[{n // 2}:]",
    }
    assert {"kind": "method_used", "name": "theta_bn"}.items() <= result["cards"][-1].items()

    script = Path(result["code_path"]).read_text(encoding="utf-8")
    recipe_src = (_PKG_RECIPES / "theta_bn.py").read_text(encoding="utf-8")
    assert recipe_src.rstrip("\n") in script, "the recipe is in the script verbatim"
    assert script.index("B_up = (") < script.index("def theta_bn("), "inputs come first"


async def test_a_function_recipe_is_applied_through_call_on_wind_swe_shaped_series(
    tmp_path, recipes_dir
):
    """rankine_hugoniot is a library: nothing runs on the inputs unless a function is
    called. `call` applies it, here on a proton speed and a density stored as Wind SWE
    stores them — `(N, 1)` — and on a `(N, 3)` field; its own `export()` calls fill the
    exports, and its self-check has already run."""
    n = 61
    shock = np.datetime64("2015-03-17T04:45:00", "s")
    t = shock + (np.arange(n) - n // 2) * np.timedelta64(60, "s")
    after = t > shock
    density = np.where(after, 45.12, 17.43)[:, None]
    speed = np.where(after, 514.1, 411.3)[:, None]
    field = np.tile([6.0, 0.0, 8.0], (n, 1))
    field[after] *= 2.527
    _save(tmp_path / "data", "np", t, density, "cm-3", ["Np"])
    _save(tmp_path / "data", "vp", t, speed, "km/s", ["Vp"])
    _save(tmp_path / "data", "b", t, field, "nT", ["Bx", "By", "Bz"])

    result = await run_recipe(
        "rankine_hugoniot",
        inputs={
            "n": "load_data('np')",
            "v": "load_data('vp')",
            "b": "load_data('b')",
            "shock": "np.datetime64('2015-03-17T04:45:00')",
        },
        call=(
            "rh_jump(*upstream_downstream(n.time, n.values, shock), "
            "*upstream_downstream(v.time, v.values, shock), "
            "*upstream_downstream(b.time, b.values, shock))"
        ),
        _plot_dir=str(tmp_path),
        _run_idx=0,
    )

    assert result.get("error") is None, result.get("stderr", "")
    assert result["exports"]["r"]["mean"] == pytest.approx(45.12 / 17.43, rel=1e-3)
    assert result["exports"]["V_shock"]["mean"] == pytest.approx(579, abs=5)
    assert "Compression check" in result["stdout"]
    assert result["stdout"].rstrip().endswith(")"), "the call's value is printed last"


async def test_a_recipe_that_guards_its_demo_behind_main_does_not_run_the_demo(
    tmp_path, recipes_dir
):
    """pressure_balance demonstrates itself under `if __name__ == "__main__"`.
    Through run_recipe the bound-input run block executes, but the demo stays off."""
    result = await run_recipe(
        "pressure_balance",
        inputs={"n_sw": 12.0, "V_sw": 600.0, "B_sw": 5.0},
        _plot_dir=str(tmp_path),
        _run_idx=0,
    )
    assert result.get("error") is None, result.get("stderr", "")
    assert 6 < result["exports"]["r_mp_RE"]["mean"] < 9, (
        "a compressed magnetopause, not the demo's 10"
    )
    assert result["stdout"].count("Magnetopause standoff") == 1, (
        "one run: the run block, not the demo"
    )


# ── what the tool refuses, before any process is spawned ───────────────────────────


async def test_an_unknown_recipe_is_an_error_that_points_at_list_recipes(recipes_dir):
    result = await run_recipe("no_such_recipe", _plot_dir="/nonexistent")
    assert "not found" in result["error"] and "list_recipes" in result["error"]


@pytest.mark.parametrize("name", ["../../etc/passwd", "../recipes/theta_bn", ""])
async def test_a_path_is_not_a_recipe_name(name, recipes_dir):
    assert "error" in await run_recipe(name, _plot_dir="/nonexistent")


async def test_an_input_that_is_not_a_python_name_is_refused(recipes_dir, monkeypatch):
    spawned = []
    monkeypatch.setattr("helioai.tools.sandbox.run_python", lambda *a, **k: spawned.append(a) or {})
    result = await run_recipe("theta_bn", inputs={"B up": "1"}, _plot_dir="/nonexistent")
    assert "not a valid Python name" in result["error"] and spawned == []
    result = await run_recipe("theta_bn", inputs={"B_up": "  "}, _plot_dir="/nonexistent")
    assert "has no value" in result["error"] and spawned == []


def test_the_script_binds_expressions_and_literals_then_the_recipe_then_the_call():
    script = recipe_script(
        "demo",
        "# name: demo\nresult = f(x, y)\n",
        {"x": "load_data('b').values", "y": [1, 2.5], "z": 3, "flag": True},
        call="g(result)",
    )
    lines = script.splitlines()
    assert '__name__ = "recipe"' in lines
    assert "x = (load_data('b').values)" in lines
    assert "y = ([1, 2.5])" in lines and "z = (3)" in lines and "flag = (True)" in lines
    assert (
        lines.index("z = (3)")
        < lines.index("result = f(x, y)")
        < lines.index("_recipe_result = (g(result))")
    )
    assert lines[-1] == "print(repr(_recipe_result))"


def test_without_a_call_the_script_ends_with_the_recipe():
    script = recipe_script("demo", "result = 1\n", {}, None)
    assert script.rstrip().endswith("result = 1")
    assert "_recipe_result" not in script


def test_a_string_that_cannot_be_an_expression_is_bound_as_the_literal_it_is():
    """Live `superposed_epoch` run, 2026-09-24: `inputs={"units": "nT", "param_label":
    "|B| OMNI 1-min"}` became `param_label = (|B| OMNI 1-min)` — a SyntaxError, a lost
    turn, and the model re-quoting every string. A string that does not parse, or is a
    bare name nothing in a fresh script can carry, is the literal it obviously is;
    numbers, lists and calls stay the expressions they always were."""
    script = recipe_script(
        "demo",
        "result = 1\n",
        {
            "units": "nT",
            "param_label": "|B| OMNI 1-min",
            "when": "2015-03-17T04:00:00",
            "events": "load_data('f_events')",
            "n_grid": "200",
            "seq": "[1, 2]",
            "t0": "np.datetime64('2015-03-17')",
        },
        None,
    )
    lines = script.splitlines()
    assert "units = ('nT')" in lines
    assert "param_label = ('|B| OMNI 1-min')" in lines
    assert "when = ('2015-03-17T04:00:00')" in lines
    assert "events = (load_data('f_events'))" in lines
    assert "n_grid = (200)" in lines and "seq = ([1, 2])" in lines
    assert "t0 = (np.datetime64('2015-03-17'))" in lines


# ── how the runtime reads a recipe run ───────────────────────────────────────────


def _run_recipe_payload(**extra) -> dict:
    return {
        "stdout": "{'theta_bn_deg': 60.0}",
        "stderr": None,
        "figure_paths": [],
        "n_figures": 0,
        "exports": {
            "theta_bn": {"mean": 60.0, "min": 60.0, "max": 60.0, "std": 0.0, "units": "deg"}
        },
        "cards": [
            {
                "kind": "method_used",
                "name": "theta_bn",
                "reference": "Colburn & Sonett 1966",
                "method": "…",
            }
        ],
        "code_path": "/w/code_0.py",
        "n_lines": 120,
        "recipe": {"name": "theta_bn", "reference": "Colburn & Sonett 1966", "description": "…"},
        "inputs": {"B_up": "load_data('b').values[:20]"},
        **extra,
    }


def test_a_recipe_run_yields_the_exports_the_recipe_chip_and_the_script():
    kinds = [a["kind"] for a in _extract_artifact("run_recipe", _run_recipe_payload())]
    assert kinds == ["exports", "recipe_used", "code"]
    chip = next(
        a
        for a in _extract_artifact("run_recipe", _run_recipe_payload())
        if a["kind"] == "recipe_used"
    )
    assert chip["name"] == "theta_bn" and chip["reference"] == "Colburn & Sonett 1966"


def test_a_failed_recipe_run_still_surfaces_its_script():
    arts = _extract_artifact("run_recipe", _run_recipe_payload(error="NameError: B_dn"))
    assert [a["kind"] for a in arts] == ["code"] and arts[0]["failed"]


def test_a_recipe_run_gets_the_sandbox_directories_as_trusted_arguments(tmp_path):
    from helioai.runtime.context import RunContext

    ctx = RunContext.for_session("u", "s", label="lbl")
    args = trusted_args("run_recipe", ctx)
    assert args["_plot_dir"] == str(ctx.session_dir) and "_run_idx" in args
    assert trusted_args("run_recipe", ctx, no_network=True)["_no_net"] is True


def _history_with(*calls: ToolCall) -> list[Message]:
    return [
        Message(role="user", content="q"),
        Message(role="assistant", content="", tool_calls=list(calls)),
        *(
            Message(role="tool", tool_call_id=tc.id, name=tc.name, content=json.dumps({"ok": True}))
            for tc in calls
        ),
    ]


def test_a_recipe_run_through_run_recipe_is_not_a_bypass():
    exports = [{"kind": "exports", "values": {"theta_bn": {"mean": 60.0}}}]
    text = "θ_Bn = 60°."
    _, flagged_without = _flag_recipe_bypass(text, _history_with(), exports)
    assert flagged_without == [{"recipe": "theta_bn", "reason": "not_loaded"}], "the control case"

    ran = _history_with(ToolCall(id="r1", name="run_recipe", arguments={"name": "theta_bn"}))
    out, flagged = _flag_recipe_bypass(text, ran, exports)
    assert flagged == [] and out == text


def test_a_recipe_read_then_run_through_run_recipe_is_neither_shallow_nor_uncalled():
    load = ToolCall(id="l1", name="load_recipe", arguments={"name": "theta_bn"})
    run = ToolCall(id="r1", name="run_recipe", arguments={"name": "theta_bn"})
    history = _history_with(load, run)
    recipe_src = (_PKG_RECIPES / "theta_bn.py").read_text(encoding="utf-8")
    history[2].content = json.dumps(
        {"name": "theta_bn", "code": recipe_src, "metadata": {"outputs": "theta_bn_deg"}}
    )
    exports = [{"kind": "exports", "values": {"theta_bn": {"mean": 60.0}}}]
    _, flagged = _flag_recipe_bypass("θ_Bn = 60°.", history, exports)
    assert flagged == []


def test_the_export_lists_a_recipe_the_session_ran():
    from helioai.export import _collect_recipes

    history = _history_with(ToolCall(id="r1", name="run_recipe", arguments={"name": "theta_bn"}))
    found = _collect_recipes(history)
    assert [r["name"] for r in found] == ["theta_bn"] and "Colburn" in found[0]["reference"]


def test_the_interfaces_describe_a_recipe_run_by_its_name_and_inputs():
    from helioai.core.event_display import describe_tool_call, describe_tool_result

    line = describe_tool_call(
        "run_recipe", {"name": "theta_bn", "inputs": {"B_up": "…", "B_dn": "…"}}
    )
    assert line == "theta_bn(B_up, B_dn)"
    result = ToolResult.from_raw("run_recipe", _run_recipe_payload())
    assert "1 exported" in describe_tool_result("run_recipe", result.for_llm())


def test_load_recipe_still_reads_and_run_recipe_is_registered_for_the_analyst_roles():
    from helioai.core.sub_agents import AGENT_ROLES

    assert "run_recipe" in AGENT_ROLES["data_analyst"].allowed_tools
    assert "run_recipe" in AGENT_ROLES["plasma_physicist"].allowed_tools
    assert "run_recipe" not in AGENT_ROLES["parameter_hunter"].allowed_tools
    assert "run_recipe" not in AGENT_ROLES["librarian"].allowed_tools
    assert _rcp._recipe_path("theta_bn") is not None
