"""Unit tests for helioai.core.skills_loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from helioai.core import skills_loader
from helioai.core.skills_loader import SkillError, load_index, load_skill


def _write_skill(root: Path, name: str, frontmatter: str, body: str = "Body text.") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    path = skill_dir / "SKILL.md"
    path.write_text(f"---\n{frontmatter}\n---\n{body}\n", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def isolated_skills_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(skills_loader, "SKILLS_DIR", tmp_path)
    skills_loader._discover.cache_clear()
    yield tmp_path
    skills_loader._discover.cache_clear()


def test_load_index_empty_when_no_skills() -> None:
    assert load_index() == "(no skills available)"


def test_discover_and_load_valid_skill(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skills_loader, "SKILLS_DIR", tmp_path)
    skills_loader._discover.cache_clear()
    _write_skill(
        tmp_path,
        "demo",
        frontmatter=(
            "name: demo\n"
            "description: A demo skill.\n"
            "when_to_use: When demoing.\n"
            "allowed_tools: [search_parameters]\n"
        ),
        body="# Procedure\nDo this then that.\n",
    )
    index = load_index()
    assert "`demo`" in index
    assert "When demoing" in index
    body = load_skill("demo")
    assert body.startswith("# Procedure")
    assert "name:" not in body


def test_load_unknown_skill_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skills_loader, "SKILLS_DIR", tmp_path)
    skills_loader._discover.cache_clear()
    _write_skill(tmp_path, "alpha", frontmatter="name: alpha\ndescription: x\nwhen_to_use: x\n")
    with pytest.raises(SkillError, match="unknown skill 'beta'") as exc:
        load_skill("beta")
    assert "alpha" in str(exc.value)


def test_missing_required_field_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skills_loader, "SKILLS_DIR", tmp_path)
    skills_loader._discover.cache_clear()
    _write_skill(tmp_path, "broken", frontmatter="name: broken\ndescription: missing when_to_use\n")
    with pytest.raises(SkillError, match="missing required field"):
        load_index()


def test_name_must_match_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skills_loader, "SKILLS_DIR", tmp_path)
    skills_loader._discover.cache_clear()
    _write_skill(tmp_path, "dirname", frontmatter="name: other\ndescription: x\nwhen_to_use: x\n")
    with pytest.raises(SkillError, match="other"):
        load_index()


def test_invalid_yaml_frontmatter_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skills_loader, "SKILLS_DIR", tmp_path)
    skills_loader._discover.cache_clear()
    skill_dir = tmp_path / "broken_yaml"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: broken_yaml\ndescription: [\nwhen_to_use: x\n---\nbody\n",
        encoding="utf-8",
    )
    with pytest.raises(SkillError, match="YAML invalid"):
        load_index()


def test_unclosed_frontmatter_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skills_loader, "SKILLS_DIR", tmp_path)
    skills_loader._discover.cache_clear()
    skill_dir = tmp_path / "unclosed"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: unclosed\ndescription: x\nwhen_to_use: x\nbody without close\n",
        encoding="utf-8",
    )
    with pytest.raises(SkillError, match="not closed"):
        load_index()


def test_allowed_tools_must_be_list(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skills_loader, "SKILLS_DIR", tmp_path)
    skills_loader._discover.cache_clear()
    _write_skill(
        tmp_path,
        "badtools",
        frontmatter="name: badtools\ndescription: x\nwhen_to_use: x\nallowed_tools: not_a_list\n",
    )
    with pytest.raises(SkillError, match="allowed_tools must be a list"):
        load_index()


def test_index_replaces_pipe_in_when_to_use(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skills_loader, "SKILLS_DIR", tmp_path)
    skills_loader._discover.cache_clear()
    _write_skill(
        tmp_path,
        "piped",
        frontmatter="name: piped\ndescription: x\nwhen_to_use: scenarios A | B | C\n",
    )
    index = load_index()
    assert "A | B | C" not in index
    assert "A / B / C" in index


def test_multiple_skills_in_index(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skills_loader, "SKILLS_DIR", tmp_path)
    skills_loader._discover.cache_clear()
    for name in ("alpha", "beta", "gamma"):
        _write_skill(
            tmp_path,
            name,
            frontmatter=f"name: {name}\ndescription: {name} skill\nwhen_to_use: {name}\n",
        )
    index = load_index()
    assert "`alpha`" in index
    assert "`beta`" in index
    assert "`gamma`" in index
