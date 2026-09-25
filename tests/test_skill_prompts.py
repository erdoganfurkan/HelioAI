"""Tests on the shipped skill prompts.

These read the SKILL.md files straight from disk rather than through
`skills_loader`: the subject is the prompt text HelioAI ships, not the loader.
Prompt wording is functional here — session 39 lost two run_python calls to a
helper signature that had been trimmed out of a prompt — so the rules that were
added in response to a real failure are pinned.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helioai.core.skills_loader import SKILLS_DIR


def skill(name: str) -> str:
    return (Path(SKILLS_DIR) / name / "SKILL.md").read_text(encoding="utf-8")


def test_all_expected_skills_are_shipped():
    assert sorted(p.name for p in Path(SKILLS_DIR).iterdir() if p.is_dir()) == [
        "data_analyst",
        "helioai_helper",
        "librarian",
        "parameter_hunter",
        "plasma_physicist",
        "plotting",
    ]


def test_parameter_hunter_requires_verbatim_ids():
    """Guards a real hallucination.

    Asked for Cluster C3 CIS-HIA ion density, the sub-agent searched correctly —
    the right product was the top hit — then reported
    `csa/C3_PP_CIS/C3_HIA_ONBOARD_MOMENTS/density`, which does not exist in the
    index. It had tidied the real, ugly id
    `csa/C3_CP_CIS-HIA_ONBOARD_MOMENTS/density__C3_CP_CIS-HIA_ONBOARD_MOMENTS`
    into a plausible shape. A confidently wrong id is the worst possible answer:
    the user tries it, it fails, and everything else the agent said becomes
    suspect.
    """
    body = skill("parameter_hunter")
    assert "verbatim" in body.lower(), "the copy-exactly rule must stay in the prompt"
    assert "never invent" in body.lower()
    assert "RULE ZERO" in body, "the rule must be top-level, not buried in a fallback branch"


def test_parameter_hunter_shows_a_long_ugly_id_example():
    """Only ever showing short tidy AMDA ids teaches the wrong shape.

    The invented id looked exactly like the skill's only example
    (`amda/ace_imf_all`): provider/short_name. CSA ids do not look like that, so
    at least one realistic long id must be in front of the model.
    """
    body = skill("parameter_hunter")
    assert "density__C3_CP_CIS-HIA_ONBOARD_MOMENTS" in body


@pytest.mark.parametrize(
    "name",
    [
        "data_analyst",
        "helioai_helper",
        "librarian",
        "parameter_hunter",
        "plasma_physicist",
        "plotting",
    ],
)
def test_skill_has_frontmatter_and_a_body(name):
    body = skill(name)
    assert body.startswith("---"), f"{name} is missing YAML frontmatter"
    assert "name:" in body and "description:" in body
    assert len(body) > 400, f"{name} is suspiciously short"
    assert len(body) > 400, f"{name} is suspiciously short"


def test_plasma_physicist_does_not_use_async_tools_or_np_interp():
    body = skill("plasma_physicist")
    assert "helioai.tools.plasmapy_tools" not in body, (
        "the skill must not import async registry tools into synchronous sandbox code"
    )
    assert "np.interp(" not in body, (
        "np.interp bridges across telemetry gaps; use interp_to instead"
    )
    assert "interp_to" in body
    assert "magnitude" in body


def test_plasma_physicist_takes_a_fraction_over_finite_samples_only():
    """`np.nanmean(x > threshold)` is nan-aware in appearance only.

    `NaN > 1.0` is False, so the comparison files every gap under "does not exceed"
    and leaves a boolean array with no NaN for nanmean to skip. Measured on a series
    half missing: 0.25 reported against a true fraction of 0.5. The template carried
    that idiom, and a template is copied — this pins the shape that reads the same
    and is correct.

    Asserted on the export line alone, not on the whole file: the prompt names the
    trap in prose right above the fix, the way the other skills name theirs, so the
    bad form is *supposed* to appear in the text. A blanket substring check failed
    on that comment — it could not tell a warning from a usage.
    """
    body = skill("plasma_physicist")
    assert 'export("beta_gt_1_fraction", float(np.nanmean(' not in body
    assert 'export("beta_gt_1_fraction", float(np.mean(finite_beta > 1.0)))' in body
    assert "finite_beta = beta_ts[np.isfinite(beta_ts)]" in body


def test_data_analyst_templates_use_the_real_attribute_name():
    """`load_data()` returns `.units`; the skill used to document and use `.unit`.

    The model was not inventing it — it was obeying the skill, and every session lost a
    turn to `AttributeError: no attribute 'unit'`. Asserted on the executable lines only:
    the prose deliberately names the trap, so scanning the whole file cannot tell a
    warning from a usage.
    """
    body = skill("data_analyst")
    code_lines = [ln for ln in body.splitlines() if "var." in ln and "there is no" not in ln]
    assert code_lines, "no template lines found — did the skill move?"
    offenders = [ln for ln in code_lines if ".unit)" in ln or ".unit}" in ln]
    assert not offenders, f"templates call the non-existent .unit: {offenders}"


_COPY = re.compile(r"\b(paste|pasted|copy|include it)\b", re.IGNORECASE)
_SCRIPT = re.compile(
    r"usage\W{0,3}inside run_python|\brun this script|pass it to run_python", re.IGNORECASE
)
_ALLOWED = ("never", "not ", "n't", "outside helioai")


def _model_facing_texts() -> dict[str, str]:
    import helioai.tools.setup  # noqa: F401  — populates the registry
    from helioai.core.agent_loop import SYSTEM_PROMPT
    from helioai.core.sub_agents import AGENT_ROLES
    from helioai.tools.registry import registry

    texts = {
        f"skill {p.name}": (p / "SKILL.md").read_text(encoding="utf-8")
        for p in Path(SKILLS_DIR).iterdir()
        if p.is_dir()
    }
    texts |= {f"addon {name}": role.system_addon for name, role in AGENT_ROLES.items()}
    texts["lead prompt"] = SYSTEM_PROMPT
    texts |= {f"tool {t.name}": t.description for t in registry.list_tool_defs()}
    texts |= _recipe_texts()
    return texts


def _recipe_texts() -> dict[str, str]:
    """What `list_recipes` and `load_recipe` hand the model: the header and the usage."""
    import ast

    from helioai.config import _PKG_RECIPES
    from helioai.tools.recipes import _parse_header

    texts = {}
    for path in sorted(_PKG_RECIPES.glob("*.py")):
        code = path.read_text(encoding="utf-8")
        header = _parse_header(code)
        usage = ast.get_docstring(ast.parse(code)) or ""
        texts[f"recipe {path.stem}"] = "\n\n".join([*header.values(), usage])
    return texts


def _copy_instructions(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n\s*\n", text)
    return [
        " ".join(s.split())
        for s in sentences
        if not any(a in s.lower() for a in _ALLOWED)
        and ((_COPY.search(s) and "recipe" in s.lower()) or _SCRIPT.search(s))
    ]


def test_no_text_the_model_reads_says_to_copy_a_recipe():
    """One way to use a recipe: `run_recipe`, which runs its source verbatim. Six texts
    said "load_recipe, then paste it into run_python" — the path every recipe check was
    written to police, and the one two of twelve sessions of the 2026-09-25 A/B took with
    `rankine_hugoniot`. Copying stays legitimate for a script that will run outside
    HelioAI, and a sentence that says not to copy is not an instruction to. The recipes'
    own headers and usage notes are read too — `load_recipe` returns them — and seven of
    them said "Usage inside run_python: … then run this script", which a copy-word scan
    could not see."""
    offenders = {
        where: found
        for where, text in _model_facing_texts().items()
        if (found := _copy_instructions(text))
    }
    assert not offenders, offenders
    assert _copy_instructions("load_recipe(name), adapt it and paste it into run_python.")
    assert _copy_instructions("Usage inside run_python:\n    events = load_data('x')")
    assert _copy_instructions("B = load_data('b')\n    # Then run this script.")
    assert _copy_instructions("Returns the Python source — pass it to run_python to execute it.")
    assert not _copy_instructions("Never copy a recipe into run_python.")
