"""Run a shipped recipe the way the sandbox does, offline and without a model.

Recipes are scripts `exec`'d inside `run_python` with an injected namespace (`np`,
`plt`, `export`, `load_data`, the physics helpers). This fixture rebuilds that namespace
from the same helper source the notebook export ships (`export._HELPER_DEFS` mirrors the
sandbox preamble and is held to it by `tests/test_export.py`), captures every `export()`
and every figure, and sets `__name__ = "recipe"` so guarded demos stay out of
`run_recipe`-style executions.

The helper here intentionally imports nothing from `helioai.data.recipes`: recipes are
not modules, and they must keep working with no HelioAI installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from helioai.export import _HELPER_DEFS  # noqa: E402

RECIPES_DIR = Path(__file__).resolve().parents[2] / "helioai" / "data" / "recipes"
RECIPE_NAMES = sorted(p.stem for p in RECIPES_DIR.glob("*.py"))


@dataclass
class RecipeRun:
    """What a recipe left behind: its namespace, its exports and its figures."""

    namespace: dict[str, Any]
    exports: dict[str, dict[str, Any]] = field(default_factory=dict)
    figures: list[Any] = field(default_factory=list)

    def value(self, name: str) -> np.ndarray:
        return np.asarray(self.exports[name]["value"])


def _sandbox_like_namespace(inputs: dict[str, Any], exports: dict, figures: list) -> dict:
    ns: dict[str, Any] = {"__name__": "recipe", "np": np, "plt": plt}
    exec(_HELPER_DEFS, ns)

    def export(name, data, units=""):
        exports[name] = {"value": np.asarray(data), "units": units}

    def load_data(name):
        try:
            return inputs["datasets"][name]
        except KeyError as e:
            raise KeyError(f"no dataset {name!r} injected into this recipe run") from e

    ns.update(
        {
            "export": export,
            "load_data": load_data,
            "document_method": lambda *a, **k: None,
            "param_card": lambda *a, **k: None,
            "save_path": lambda name: str(Path(inputs.get("_tmp", ".")) / name),
        }
    )
    ns.update({k: v for k, v in inputs.items() if k not in ("datasets", "_tmp")})
    return ns


def run_recipe(name: str, tmp_path: Path | None = None, **inputs: Any) -> RecipeRun:
    """Execute `<name>.py` with `inputs` bound as globals, capturing exports and figures.

    Args:
        name: Recipe stem, as `load_recipe` takes it.
        tmp_path: Where `save_path()` points; a temporary directory in tests.
        **inputs: Globals the recipe reads (`B_up`, `flux`, `events`…). `datasets`
            is special: `{name: obj}` answered by `load_data(name)`.

    Returns:
        The finished run. Any `assert` the recipe carries has already passed.
    """
    src = (RECIPES_DIR / f"{name}.py").read_text(encoding="utf-8")
    exports: dict = {}
    figures: list = []
    inputs = {**inputs, "_tmp": str(tmp_path or ".")}
    ns = _sandbox_like_namespace(inputs, exports, figures)
    with _figures_captured(figures):
        exec(compile(src, f"{name}.py", "exec"), ns)
    return RecipeRun(namespace=ns, exports=exports, figures=figures)


class _figures_captured:
    """Make `plt.show()` record the figure and close it, the way the sandbox saves it.

    Re-entrant, so the fixture can hold it for a whole test while `run_recipe` takes
    it again: the recipe functions a test calls afterwards (`compute_pad` plots) must
    not fall back to Agg's real `show`.
    """

    _depth = 0
    _original = None

    def __init__(self, figures: list):
        self.figures = figures

    def __enter__(self):
        if _figures_captured._depth == 0:
            _figures_captured._original = plt.show
        _figures_captured._depth += 1
        figures = self.figures

        def capture_show(*a, **k):
            figures.append(plt.gcf())
            plt.close("all")

        plt.show = capture_show
        return self

    def __exit__(self, *exc):
        _figures_captured._depth -= 1
        plt.close("all")
        if _figures_captured._depth == 0:
            plt.show = _figures_captured._original
            # pitch_angle_dist switches the global style; do not let it leak into the suite
            plt.style.use("default")
        return False


@pytest.fixture
def recipe(tmp_path):
    """`recipe(name, **inputs)` → RecipeRun, with `save_path()` under tmp_path and every
    figure the test provokes afterwards captured rather than shown."""
    figures: list = []
    with _figures_captured(figures):

        def _run(name: str, **inputs: Any) -> RecipeRun:
            return run_recipe(name, tmp_path=tmp_path, **inputs)

        yield _run
