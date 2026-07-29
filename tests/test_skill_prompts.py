"""Tests on the shipped skill prompts.

These read the SKILL.md files straight from disk rather than through
`skills_loader`: the subject is the prompt text HelioAI ships, not the loader.
Prompt wording is functional here — session 39 lost two run_python calls to a
helper signature that had been trimmed out of a prompt — so the rules that were
added in response to a real failure are pinned.
"""

from __future__ import annotations

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
