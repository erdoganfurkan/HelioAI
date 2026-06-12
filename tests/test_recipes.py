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
        "# name: theta_bn\n# description: Shock normal angle.\n# inputs: B_up, B_dn\n# outputs: theta_bn_deg\npass\n",
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
