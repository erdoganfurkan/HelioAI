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

import pytest

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
        "fill_values",
        "mvab",
        "pitch_angle_dist",
        "pressure_balance",
        "rankine_hugoniot",
        "sep_onset_poisson_cusum",
        "shock_timing_2sc",
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


def test_version_is_single_sourced():
    """pyproject reads the version from __init__; they cannot drift apart.

    They were duplicated, which is the kind of thing nobody notices until a
    release ships with the wrong number on it.
    """
    import tomllib

    pyproject = tomllib.loads((PACKAGE_DIR.parent / "pyproject.toml").read_text())
    assert "version" in pyproject["project"].get("dynamic", []), (
        "project.version should be dynamic, sourced from helioai/__init__.py"
    )
    assert pyproject["tool"]["hatch"]["version"]["path"] == "helioai/__init__.py"
    assert helioai.__version__


def test_changelog_documents_the_current_version():
    changelog = (PACKAGE_DIR.parent / "CHANGELOG.md").read_text()
    assert f"[{helioai.__version__}]" in changelog, (
        f"CHANGELOG.md has no entry for {helioai.__version__}"
    )


def test_citation_matches_the_current_version():
    citation = (PACKAGE_DIR.parent / "CITATION.cff").read_text()
    assert f"version: {helioai.__version__}" in citation


# ── example notebooks ──────────────────────────────────────────────────────────

EXAMPLES = sorted((PACKAGE_DIR.parent / "examples").glob("*.ipynb"))


def test_example_notebooks_exist():
    assert EXAMPLES, "the examples/ notebooks are part of the documented onboarding path"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_notebook_is_valid(path):
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.read(path, as_version=4))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_notebook_has_no_committed_outputs(path):
    """Outputs are a snapshot of one run and bloat the repository.

    PyHC standard 14 asks that notebooks not be committed as binary files; keeping
    them output-free is what makes shipping them here defensible.
    """
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(path, as_version=4)
    with_outputs = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and c.get("outputs")]
    assert not with_outputs, f"{path.name} has outputs in cells {with_outputs}"


def _cell_bodies(path):
    """Yield (index, parseable-source) for every non-magic code cell."""
    nbformat = pytest.importorskip("nbformat")
    for i, cell in enumerate(nbformat.read(path, as_version=4).cells):
        if cell.cell_type != "code" or cell.source.lstrip().startswith("%%"):
            continue
        # Line magics are valid in a notebook but not in plain Python.
        yield (
            i,
            "\n".join(
                "pass" if ln.lstrip().startswith(("%", "!")) else ln
                for ln in cell.source.split("\n")
            ),
        )


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_notebook_calls_async_tools_with_await(path):
    """The bug that actually shipped: an async tool called without `await`.

    Every registered tool is a coroutine function. Calling one without awaiting is
    perfectly valid Python — it just returns a coroutine, so the notebook fails at
    runtime with a confusing formatting error rather than at parse time. A syntax
    check cannot see this; only an AST walk can.
    """
    import ast
    import inspect

    from helioai.tools import plasmapy_tools

    async_names = {
        name
        for name, obj in vars(plasmapy_tools).items()
        if inspect.iscoroutinefunction(obj) and not name.startswith("_")
    }
    assert async_names, "expected plasmapy_tools to expose coroutine functions"

    offenders = []
    for i, body in _cell_bodies(path):
        tree = ast.parse("async def _cell():\n" + "\n".join("    " + ln for ln in body.split("\n")))
        awaited = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Await)}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in async_names
                and id(node) not in awaited
            ):
                offenders.append(f"cell {i}: {node.func.id}() is not awaited")

    assert not offenders, f"{path.name} — " + "; ".join(offenders)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_notebook_code_parses(path):
    """A cell that stopped being valid Python at all."""
    import ast

    nbformat = pytest.importorskip("nbformat")
    for i, cell in enumerate(nbformat.read(path, as_version=4).cells):
        if cell.cell_type != "code" or cell.source.lstrip().startswith("%%"):
            continue
        # Line magics are valid in a notebook but not in plain Python.
        body = "\n".join(
            "pass" if ln.lstrip().startswith(("%", "!")) else ln for ln in cell.source.split("\n")
        )
        # Wrapped in async def so top-level `await` parses, as it does in Jupyter.
        wrapped = "async def _cell():\n" + "\n".join("    " + ln for ln in body.split("\n"))
        try:
            ast.parse(wrapped)
        except SyntaxError as exc:
            raise AssertionError(f"{path.name} cell {i} is not valid Python: {exc}") from exc
