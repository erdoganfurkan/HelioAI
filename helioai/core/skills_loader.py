"""Discover and serve markdown-defined skills to the agent.

Each skill lives in skills/<name>/SKILL.md with YAML frontmatter:
    ---
    name: parameter_hunter
    description: Find speasy parameter ids from a natural language query.
    when_to_use: User asks about a parameter without giving its exact id.
    allowed_tools: [search_parameters]
    ---
    # body of the procedure
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

SKILLS_DIR = Path(__file__).resolve().parent / "skills"
REQUIRED_FIELDS = ("name", "description", "when_to_use")


class SkillError(RuntimeError):
    pass


@dataclass(frozen=True)
class SkillMeta:
    name: str
    description: str
    when_to_use: str
    allowed_tools: tuple[str, ...]
    path: Path


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        raise SkillError("missing YAML frontmatter fence")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise SkillError("frontmatter is not closed with '---'")
    try:
        data = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as e:
        raise SkillError(f"frontmatter YAML invalid: {e}") from e
    if not isinstance(data, dict):
        raise SkillError("frontmatter must be a YAML mapping")
    return data, parts[2].lstrip("\n")


def _validate(meta: dict[str, Any], dir_name: str, path: Path) -> SkillMeta:
    for f in REQUIRED_FIELDS:
        if not meta.get(f):
            raise SkillError(f"skill {path}: missing required field '{f}'")
    if meta["name"] != dir_name:
        raise SkillError(f"skill {path}: name {meta['name']!r} != dir {dir_name!r}")
    allowed = meta.get("allowed_tools") or []
    if not isinstance(allowed, list):
        raise SkillError(f"skill {path}: allowed_tools must be a list of strings")
    return SkillMeta(
        name=meta["name"],
        description=meta["description"].strip(),
        when_to_use=meta["when_to_use"].strip(),
        allowed_tools=tuple(allowed),
        path=path,
    )


@lru_cache(maxsize=1)
def _discover() -> dict[str, SkillMeta]:
    out: dict[str, SkillMeta] = {}
    if not SKILLS_DIR.exists():
        return out
    for sub in sorted(SKILLS_DIR.iterdir()):
        if not sub.is_dir():
            continue
        path = sub / "SKILL.md"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        meta_dict, _ = _split_frontmatter(text)
        meta = _validate(meta_dict, sub.name, path)
        out[meta.name] = meta
    return out


def load_index() -> str:
    skills = _discover()
    if not skills:
        return "(no skills available)"
    lines = ["| Skill | When to use |", "|---|---|"]
    for meta in skills.values():
        when = meta.when_to_use.replace("|", "/").replace("\n", " ")
        lines.append(f"| `{meta.name}` | {when} |")
    return "\n".join(lines)


def load_skill(name: str) -> str:
    skills = _discover()
    if name not in skills:
        known = ", ".join(sorted(skills)) or "(none)"
        raise SkillError(f"unknown skill {name!r}. Known: {known}")
    text = skills[name].path.read_text(encoding="utf-8")
    _, body = _split_frontmatter(text)
    return body


def list_skill_names() -> list[str]:
    return list(_discover().keys())
