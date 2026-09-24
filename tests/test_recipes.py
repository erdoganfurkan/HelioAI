"""Tests for helioai.tools.recipes (list_recipes, load_recipe)."""

from __future__ import annotations

from pathlib import Path

import pytest

from helioai.config import settings
from helioai.tools import recipes as _rcp

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def recipes_dir(tmp_path: Path, monkeypatch):
    """Isolated recipes directory with two seed files."""
    d = tmp_path / "recipes"
    d.mkdir()
    monkeypatch.setattr(settings.recipes, "recipes_dir", d)

    (d / "theta_bn.py").write_text(
        "# name: theta_bn\n# description: Shock normal angle.\n# inputs: B_up, B_dn\n# outputs: theta_bn_deg\n# reference: Coplanarity theorem; Schwartz 1998.\npass\n",
        encoding="utf-8",
    )
    (d / "walen_test.py").write_text(
        "# name: walen_test\n# description: Walén test for RDs.\n# inputs: V, B, n\n# outputs: slope\npass\n",
        encoding="utf-8",
    )
    return d


# ── list_recipes ──────────────────────────────────────────────────────────────


async def test_list_recipes_returns_entries(recipes_dir):
    result = await _rcp.list_recipes()
    assert "recipes" in result
    assert len(result["recipes"]) == 2
    names = [r["name"] for r in result["recipes"]]
    assert "theta_bn" in names
    assert "walen_test" in names


async def test_list_recipes_sorted(recipes_dir):
    result = await _rcp.list_recipes()
    names = [r["name"] for r in result["recipes"]]
    assert names == sorted(names)


async def test_list_recipes_includes_description(recipes_dir):
    result = await _rcp.list_recipes()
    theta = next(r for r in result["recipes"] if r["name"] == "theta_bn")
    assert "Shock normal angle" in theta["description"]


async def test_list_recipes_empty_dir(tmp_path, monkeypatch):
    d = tmp_path / "empty_recipes"
    d.mkdir()
    monkeypatch.setattr(settings.recipes, "recipes_dir", d)
    result = await _rcp.list_recipes()
    assert result == {"recipes": []}


async def test_list_recipes_missing_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.recipes, "recipes_dir", tmp_path / "nonexistent")
    result = await _rcp.list_recipes()
    assert result == {"recipes": []}


# ── load_recipe ───────────────────────────────────────────────────────────────


async def test_load_recipe_returns_code(recipes_dir):
    result = await _rcp.load_recipe("theta_bn")
    assert "code" in result
    assert "name" in result
    assert len(result["code"]) > 0


async def test_load_recipe_returns_metadata_with_reference(recipes_dir):
    result = await _rcp.load_recipe("theta_bn")
    assert "metadata" in result
    assert result["metadata"].get("reference")


async def test_load_recipe_not_found(recipes_dir):
    result = await _rcp.load_recipe("nonexistent")
    assert "error" in result
    assert "not found" in result["error"]


async def test_load_recipe_traversal_slash(recipes_dir):
    result = await _rcp.load_recipe("../../etc/passwd")
    assert "error" in result


async def test_load_recipe_traversal_dotdot(recipes_dir):
    result = await _rcp.load_recipe("../recipes/theta_bn")
    assert "error" in result


async def test_load_recipe_empty_name(recipes_dir):
    result = await _rcp.load_recipe("")
    assert "error" in result


# ── Real recipes on disk ──────────────────────────────────────────────────────


def _real_recipe(filename: str) -> Path:
    """Locate a shipped recipe.

    Resolved through `settings` rather than hardcoded, so moving the recipe
    directory cannot silently break these tests again — they now fail loudly if
    the configured location is wrong, which is the point.
    """
    from helioai.config import _PKG_RECIPES

    return _PKG_RECIPES / filename


async def test_real_recipes_all_present():
    """All seven expected recipes must be loadable from the real data/recipes dir."""
    expected = {
        "theta_bn",
        "mvab",
        "walen_test",
        "rankine_hugoniot",
        "pressure_balance",
        "pitch_angle_dist",
        "superposed_epoch",
    }
    result = await _rcp.list_recipes()
    assert "recipes" in result, result
    names = {r["name"] for r in result["recipes"]}
    missing = expected - names
    assert not missing, f"Missing recipes: {missing}"


async def test_real_recipes_have_valid_headers():
    """Each real recipe must have a parseable name and description."""
    result = await _rcp.list_recipes()
    for entry in result.get("recipes", []):
        assert "name" in entry and entry["name"], f"Missing name: {entry}"
        assert "description" in entry and entry["description"], f"Missing description: {entry}"


async def test_real_recipes_have_reference():
    """Each real recipe must carry a scientific reference in its header (provenance)."""
    result = await _rcp.list_recipes()
    for entry in result.get("recipes", []):
        loaded = await _rcp.load_recipe(entry["name"])
        assert loaded["metadata"].get("reference"), f"Missing reference: {entry['name']}"


def test_sep_onset_cusum_recipe_detects_synthetic_onset():
    import types

    import matplotlib
    import numpy as np

    matplotlib.use("Agg")

    src = _real_recipe("sep_onset_poisson_cusum.py")
    code = src.read_text(encoding="utf-8")

    rng = np.random.default_rng(42)
    n = 600
    t = np.datetime64("2023-05-01T00:00:00") + np.arange(n) * np.timedelta64(60, "s")
    x = rng.normal(10.0, 1.0, n)
    x[300:] += np.linspace(0, 60, n - 300)

    exported: dict = {}
    ns = {
        "flux": types.SimpleNamespace(time=t, values=x),
        "export": lambda name, data, units="": exported.setdefault(name, np.asarray(data)),
        "bg_hours": 2.0,
    }
    exec(code, ns)

    assert ns["onset_time"] is not None
    onset = np.datetime64(ns["onset_time"])
    assert np.datetime64("2023-05-01T04:58:00") <= onset <= np.datetime64("2023-05-01T05:30:00")
    assert "cusum" in exported and exported["cusum"].shape == (n,)


def test_solar_mach_recipe_graceful_without_dep():
    from unittest.mock import patch

    import matplotlib

    matplotlib.use("Agg")
    src = _real_recipe("solar_mach.py")
    code = src.read_text(encoding="utf-8")
    with patch.dict("sys.modules", {"solarmach": None}):
        ns: dict = {}
        exec(code, ns)
    assert ns["SolarMACH"] is None


def test_recipe_usage_examples_match_their_signatures():
    """A usage line in a recipe header is code an agent copies verbatim.

    rankine_hugoniot documented `r, V_shock = rh_jump(...)` while returning
    (V_shock, r), so a copied line swapped a 579 km/s shock speed with a compression
    ratio of 2.59 — both plausible numbers, silently in the wrong variables.

    Detects a permutation, not a rename. An arity check passes the inversion, and
    comparing names outright would reject `pa, counts, edges = compute_pad(...)` for a
    function returning `pa_deg, ...`, which is an ordinary rename while unpacking.
    Reusing the SAME names in a DIFFERENT order is the thing that is never deliberate.
    """
    import ast
    from pathlib import Path

    from helioai.config import settings

    checked = 0
    for path in sorted(Path(settings.recipes.recipes_dir).glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        doc = ast.get_docstring(tree) or ""
        returns = {
            node.name: node.body[-1].value
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and isinstance(node.body[-1], ast.Return)
        }
        for func, ret in returns.items():
            elements = ret.elts if isinstance(ret, ast.Tuple) else [ret]
            if not all(isinstance(e, ast.Name) for e in elements):
                continue
            returned = [e.id for e in elements]
            for line in doc.splitlines():
                if f"= {func}(" not in line or "=" not in line:
                    continue
                unpacked = [t.strip() for t in line.split("=", 1)[0].split(",")]
                assert len(unpacked) == len(returned), (
                    f"{path.name}: docstring unpacks {len(unpacked)} value(s) from "
                    f"{func}() which returns {len(returned)}"
                )
                if set(unpacked) == set(returned):
                    assert unpacked == returned, (
                        f"{path.name}: docstring unpacks {unpacked} from {func}() "
                        f"but it returns {returned} — same names, wrong order"
                    )
                checked += 1

    assert checked, "no usage example was actually compared — the test would be vacuous"


def test_rankine_hugoniot_windows_are_derived_from_the_shock_time():
    """The recipe owns the averaging windows, because hand-picking them is the failure.

    A real run chose shock+30 min to shock+120 min, averaged the decaying sheath, and
    reported r = 1.89 against a published 2.59 — with the shock time itself correct.
    """
    from pathlib import Path

    import numpy as np

    from helioai.config import settings

    ns: dict = {}
    exec(
        (Path(settings.recipes.recipes_dir) / "rankine_hugoniot.py").read_text(encoding="utf-8"), ns
    )

    u0, u1, d0, d1 = ns["shock_windows"](np.datetime64("2015-03-17T04:00:59"))
    assert (str(u0), str(u1)) == ("2015-03-17T03:35:59", "2015-03-17T03:55:59")
    assert (str(d0), str(d1)) == ("2015-03-17T04:05:59", "2015-03-17T04:25:59")
    assert u1 < d0, "the guard band must leave the ramp out of both windows"

    # The published event reproduces, and the run that got it wrong is caught.
    ref = ns["_rh_core"](
        n_u=17.43, n_d=45.12, V_u=411.3, V_d=514.1, B_u=10.00, B_d=25.27, T_u=8.34, T_d=45.0
    )
    assert ref["r"] == pytest.approx(2.59, abs=0.02)
    assert ref["V_shock"] == pytest.approx(579, abs=5)
    assert ref["M_A"] == pytest.approx(3.20, abs=0.15)
    assert ref["r_mismatch"] < 0.25

    bad = ns["_rh_core"](
        n_u=16.99, n_d=32.11, V_u=410.3, V_d=511.9, B_u=8.86, B_d=20.81, T_u=3.42, T_d=15.6
    )
    assert bad["r_mismatch"] > 0.25, "a 1.89 compression must not pass the check"


def test_window_mean_refuses_a_nearly_empty_window():
    """A mean of one or two samples looks like a measurement and is not one."""
    from pathlib import Path

    import numpy as np

    from helioai.config import settings

    ns: dict = {}
    exec(
        (Path(settings.recipes.recipes_dir) / "rankine_hugoniot.py").read_text(encoding="utf-8"), ns
    )

    t = np.array(["2015-03-17T04:00:00", "2015-03-17T04:10:00"], dtype="datetime64[s]")
    v = np.array([10.0, 20.0])
    assert np.isnan(ns["window_mean"](t, v, t[0], t[1]))


def test_window_mean_rejects_the_1e31_fill_convention():
    """`isfinite` passes a 1e31 fill: one in a window returns a compression of 1e30.

    Found on a second event whose raw plasma moments still carried the sentinel — the
    recipe is meant to be copied into standalone scripts, where nothing blanks it.
    """
    from pathlib import Path

    import numpy as np

    from helioai.config import settings

    ns: dict = {}
    exec(
        (Path(settings.recipes.recipes_dir) / "rankine_hugoniot.py").read_text(encoding="utf-8"), ns
    )

    t = np.array(
        [
            "2015-03-17T04:00:00",
            "2015-03-17T04:01:00",
            "2015-03-17T04:02:00",
            "2015-03-17T04:03:00",
        ],
        dtype="datetime64[s]",
    )
    v = np.array([10.0, 12.0, 1e31, 11.0])
    got = ns["window_mean"](t, v, t[0], t[-1])
    assert got == pytest.approx(11.0), got


def test_shock_timing_refuses_to_invent_a_normal_from_the_timing():
    """Two spacecraft cannot determine a shock normal, and the failure is silent.

    A run wrote `n = dr / (V_shock * dt)` — the separation vector rescaled — so the
    along-normal separation came back as |dr| and the transverse one as exactly 0 km for
    any input. That zero was published as a geometrical result while the real transverse
    separation was ~470 000 km.
    """
    import numpy as np

    from helioai.config import settings

    ns: dict = {}
    exec(
        (Path(settings.recipes.recipes_dir) / "shock_timing_2sc.py").read_text(encoding="utf-8"),
        ns,
    )
    timing = ns["timing_2sc"]
    t1, t2 = np.datetime64("2015-03-17T04:00:04"), np.datetime64("2015-03-17T04:04:25")
    wind, ace = np.array([1610910.0, 346281.0, 80077.0]), np.array([1406537.0, -68250.0, -148602.0])

    for bad in ([0.0, 0.0, 0.0], [1.0, 2.0], [np.nan, 0.0, 0.0], []):
        assert "error" in timing(t1, t2, wind, ace, bad), bad

    out = timing(t1, t2, wind, ace, [0.8974, 0.3153, -0.3087], V_shock_rh=585.4)
    assert out["transverse_separation_km"] > 4e5, "the old code returned 0 here"
    assert out["shock_speed_km_s"] > 0, "the comparable speed is a magnitude, never signed"
    assert out["verdict"] == "INCONSISTENT"
    assert any("transverse" in w for w in out["warnings"])

    # The degenerate case must read as degenerate, not as a measurement.
    assert timing(t1, t2, wind, ace, ace - wind)["transverse_separation_km"] < 1.0


def test_rankine_hugoniot_normal_projection_differs_from_norm_on_oblique_flow():
    """When a normal is provided, upstream_downstream projects V·n̂ instead of |V|."""
    from pathlib import Path

    import numpy as np

    from helioai.config import settings

    ns: dict = {}
    exec(
        (Path(settings.recipes.recipes_dir) / "rankine_hugoniot.py").read_text(encoding="utf-8"), ns
    )

    t = np.array(
        [
            "2015-03-17T03:40:00",
            "2015-03-17T03:45:00",
            "2015-03-17T03:50:00",
            "2015-03-17T04:10:00",
            "2015-03-17T04:15:00",
            "2015-03-17T04:20:00",
        ],
        dtype="datetime64[s]",
    )
    # 45 deg flow: |V| = 500 km/s, Vx = Vy = 353.55 km/s
    v = np.array(
        [
            [353.5534, 353.5534, 0.0],
            [353.5534, 353.5534, 0.0],
            [353.5534, 353.5534, 0.0],
            [450.0, 450.0, 0.0],
            [450.0, 450.0, 0.0],
            [450.0, 450.0, 0.0],
        ]
    )
    shock_time = np.datetime64("2015-03-17T04:00:59")
    normal_x = np.array([1.0, 0.0, 0.0])

    # Default normal=None averages Euclidean norm |V|
    vu_norm, vd_norm = ns["upstream_downstream"](t, v, shock_time)
    assert vu_norm == pytest.approx(500.0, abs=0.1)

    # Passing normal projects V·n̂
    vu_proj, vd_proj = ns["upstream_downstream"](t, v, shock_time, normal=normal_x)
    assert vu_proj == pytest.approx(353.55, abs=0.1)
    assert vu_proj < vu_norm


def test_rankine_hugoniot_refuses_to_project_a_scalar_speed():
    """A scalar speed stored as (n, 1) must be refused by name, not by matmul.

    The benchmark fixture for the 17 March 2015 shock stores Wind SWE Proton_V_moment
    as (1361, 1) — `v.ndim > 1` is true, so the projection branch used to be entered and
    `v @ n_hat` raised "matmul: Input operand 1 has a mismatch in its core dimension 0".
    The recipe's own docstring recommends the `normal=` pipeline, so an agent following
    it hit that on the only dataset the task offers.
    """
    from pathlib import Path

    import numpy as np

    from helioai.config import settings

    ns: dict = {}
    exec(
        (Path(settings.recipes.recipes_dir) / "rankine_hugoniot.py").read_text(encoding="utf-8"), ns
    )

    t = np.arange(
        np.datetime64("2015-03-17T03:40:00"),
        np.datetime64("2015-03-17T03:56:00"),
        np.timedelta64(1, "m"),
    )
    speed = np.full((len(t), 1), 410.0)
    window = (t[0], t[-1])

    # Without a normal the scalar speed still averages, which is what the references use.
    assert ns["window_mean"](t, speed, *window) == pytest.approx(410.0)

    with pytest.raises(ValueError, match="scalar speed"):
        ns["window_mean"](t, speed, *window, normal=np.array([-0.9, 0.3, 0.1]))


# ── theta_bn — the recipe of the demo scenario (E9-D1) ────────────────────────


def _theta_bn_function():
    """The recipe's function, exec'd without its trailing run block.

    The run block assigns placeholder vectors and calls `export`; the tests here are
    about the function, and exec'ing the whole file would also need an `export` stub.
    """
    import numpy as np

    code = _real_recipe("theta_bn.py").read_text(encoding="utf-8").split("# ── Run")[0]
    ns: dict = {"np": np}
    exec(code, ns)
    return ns["theta_bn"]


def test_theta_bn_on_a_planar_shock_returns_the_analytic_angle():
    """Fixed before looking at any run, independent of the code: a shock whose normal
    is x̂ conserves B_x and compresses the tangential B_z. With |B_up| = 10 nT at 60°
    from x̂, coplanarity must give back exactly that normal and that angle.
    """
    import numpy as np

    theta_bn = _theta_bn_function()
    bx, bz = 10 * np.cos(np.radians(60)), 10 * np.sin(np.radians(60))
    res = theta_bn(np.array([bx, 0.0, bz]), np.array([bx, 0.0, 2.5 * bz]))

    assert res["theta_bn_deg"] == pytest.approx(60.0, abs=0.01)
    assert res["geometry"] == "quasi-perpendicular"
    assert np.abs(res["shock_normal"]) == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)


def test_theta_bn_on_the_wind_2015_03_17_means_lands_in_the_published_band():
    """Regression against the two databases the notebook cites, not against the agent.

    Mean Wind/MFI 3 s GSM vectors over the notebook's windows (03:48–03:56 and
    04:04–04:12 UT, measured in the live runs of 00_quickstart). Harvard-CfA gives
    58.8 ± 2.7° by the same magnetic-coplanarity method, IPShocks 63.1 ± 16.8°; the
    notebook fixes the expected band at 55–66°. The angle must land in it and the
    quoted value — 62.68° in every live run — must not drift.
    """
    theta_bn = _theta_bn_function()
    res = theta_bn([-2.329, -0.113, 9.617], [0.818, 2.529, 24.718])

    assert 55.0 <= res["theta_bn_deg"] <= 66.0
    assert res["theta_bn_deg"] == pytest.approx(62.68, abs=0.05)


def test_theta_bn_does_not_classify_an_invalid_input():
    """Audit reproduction: a NaN anywhere in the input flowed through the cross
    products to `theta_bn_deg: nan` — and `nan < 45` being False, the reply carried
    `geometry: "quasi-perpendicular"`. A physical classification of an angle that
    does not exist. An invalid input is reported as such, and nothing else.
    """
    import numpy as np

    theta_bn = _theta_bn_function()
    good = np.array([0.818, 2.529, 24.718])

    for bad in (
        np.array([np.nan, -0.113, 9.617]),
        np.full((5, 3), np.nan),
        np.zeros(3),
    ):
        res = theta_bn(bad, good)
        assert "error" in res, res
        assert "geometry" not in res
        assert "theta_bn_deg" not in res


def test_theta_bn_averages_over_the_finite_rows_of_a_gapped_interval():
    """Wind/MFI has 2.5 % fill in the demo interval, already NaN by the time the
    analyst sees it. The recipe documents an (N, 3) input, and `mean(axis=0)` over
    such an array is NaN: the documented input on the demo's own data gave the
    invalid result above. Rows with a gap are left out of the mean; enough remain.
    """
    import numpy as np

    theta_bn = _theta_bn_function()
    bx, bz = 10 * np.cos(np.radians(60)), 10 * np.sin(np.radians(60))
    up = np.tile([bx, 0.0, bz], (8, 1))
    up[3] = np.nan
    dn = np.tile([bx, 0.0, 2.5 * bz], (8, 1))
    dn[0, 2] = np.nan

    res = theta_bn(up, dn)
    assert res["theta_bn_deg"] == pytest.approx(60.0, abs=0.01)
    assert res["B_up_mean_nT"] == pytest.approx([bx, 0.0, bz], abs=1e-9)


@pytest.mark.asyncio
async def test_load_recipe_says_how_to_run_it_as_shipped():
    """A model that has just read a recipe is one paste away from running a copy of it;
    the payload names the call and, exactly, the inputs the recipe reads."""
    from helioai.tools.recipes import load_recipe

    theta = await load_recipe("theta_bn")
    assert theta["run_with"].startswith("run_recipe('theta_bn', inputs={")
    for name in ("'B':", "'shock_time':", "'B_up':", "'B_dn':"):
        assert name in theta["run_with"]
    rh = await load_recipe("rankine_hugoniot")
    assert "call='rh_jump(...)'" in rh["run_with"] and "library of functions" in rh["run_with"]
