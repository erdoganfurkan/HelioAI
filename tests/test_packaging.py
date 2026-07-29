"""Tests that the installed package is actually usable.

These guard a bug that shipped: `[tool.hatch.build.targets.wheel] packages =
["helioai"]` never included `data/recipes/`, and every storage path was derived
from `Path(__file__).parent.parent`, which resolves to `site-packages/` once
installed. A `pip install helioai` therefore had zero recipes and tried to write
user data into site-packages. Nothing in the suite noticed, because the test
suite only ever ran from a git clone.
"""

from __future__ import annotations

from pathlib import Path

import helioai
from helioai.config import _PKG_RECIPES, settings

PACKAGE_DIR = Path(helioai.__file__).resolve().parent


def test_recipes_live_inside_the_package():
    """Recipes must sit under helioai/ so the wheel carries them."""
    assert _PKG_RECIPES.is_relative_to(PACKAGE_DIR), (
        f"{_PKG_RECIPES} is outside {PACKAGE_DIR} and would not ship in the wheel"
    )


def test_every_shipped_recipe_is_present():
    shipped = sorted(p.stem for p in _PKG_RECIPES.glob("*.py"))
    assert shipped == [
        "mvab",
        "pitch_angle_dist",
        "pressure_balance",
        "rankine_hugoniot",
        "sep_onset_poisson_cusum",
        "solar_mach",
        "superposed_epoch",
        "theta_bn",
        "walen_test",
    ]


def test_recipes_dir_defaults_to_the_packaged_copy():
    assert settings.recipes.recipes_dir == _PKG_RECIPES


def test_user_data_is_never_written_inside_the_package():
    """User data must land in the repo (dev) or the XDG dir (installed), never site-packages."""
    for label, path in (
        ("data_dir", settings.data_dir),
        ("chroma_dir", settings.rag.chroma_dir),
        ("workspace_dir", settings.workspace.workspace_dir),
        ("catalogs_dir", settings.catalogs.catalogs_dir),
        ("profile_path", settings.profile.profile_path),
    ):
        assert not Path(path).resolve().is_relative_to(PACKAGE_DIR), (
            f"{label}={path} would write inside the installed package"
        )


def test_session_db_is_not_inside_the_package():
    from helioai.core.session import DEFAULT_DB

    assert not Path(DEFAULT_DB).resolve().is_relative_to(PACKAGE_DIR)
