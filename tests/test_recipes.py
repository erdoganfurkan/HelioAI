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
        "export": lambda name, data: exported.setdefault(name, np.asarray(data)),
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
