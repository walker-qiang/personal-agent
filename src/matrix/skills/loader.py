"""Skill loader: parse Markdown skill definitions with YAML frontmatter."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SkillDefinition:
    """Parsed skill definition from a Markdown file with YAML frontmatter."""

    name: str
    title: str
    description: str = ""
    workflow: list[dict[str, Any]] = field(default_factory=list)
    output_format: str = ""

    @classmethod
    def from_markdown(cls, path: Path) -> "SkillDefinition":
        """Parse a skill definition from a Markdown file with YAML frontmatter."""
        content = path.read_text(encoding="utf-8")
        name = path.stem

        frontmatter, body = _split_frontmatter(content)
        title = frontmatter.get("title", name)

        workflow = _parse_workflow(body)
        output_format = _extract_section(body, r"##\s+输出格式\s*\n(.+)", "", re.DOTALL)

        return cls(
            name=name,
            title=title or name,
            description=frontmatter.get("description", "").strip(),
            workflow=workflow,
            output_format=output_format.strip(),
        )

    def matches(self, query: str) -> bool:
        """Check if the query matches this skill's title or description."""
        q = query.lower()
        text = (self.title + " " + self.description).lower()
        # Split into words (handles both English and Chinese)
        words = [w for w in re.split(r"[\s,，。！？、；：""''（）\(\)]+", text) if len(w) >= 2]
        if any(w in q for w in words):
            return True
        # Also check if query is a substring of skill text
        if len(q) >= 2 and q in text:
            return True
        # For Chinese: 2-char n-grams from skill text
        for w in words:
            if len(w) > 2:
                for i in range(len(w) - 1):
                    if w[i:i+2] in q:
                        return True
        return False


def load_skills(skills_dir: Path) -> list[SkillDefinition]:
    """Load all skills from a directory of Markdown files."""
    skills = []
    if not skills_dir.exists():
        return skills
    for md_file in sorted(skills_dir.glob("*.md")):
        try:
            skills.append(SkillDefinition.from_markdown(md_file))
        except Exception:
            continue
    return skills


def render_skill(skill: SkillDefinition) -> str:
    """Serialize a skill back to Markdown with YAML frontmatter."""
    return "\n".join([
        "---",
        f"name: {skill.name}",
        f"title: {skill.title}",
        f"description: {skill.description}",
        "---",
        "",
        f"# {skill.title}",
        "",
        "## 工作流",
        render_workflow(skill.workflow) or "",
        "",
        "## 输出格式",
        skill.output_format,
        "",
    ])


def render_workflow(workflow: list[dict[str, Any]]) -> str:
    """Serialize parsed workflow steps back to markdown text."""
    lines = []
    for step in workflow:
        num = step.get("step", len(lines) + 1)
        tool = step.get("tool", "")
        purpose = step.get("purpose", "")
        if tool:
            args = step.get("arguments", {})
            if args:
                args_str = ", ".join(f"{k}={v}" for k, v in args.items())
                lines.append(f"{num}. {tool}({args_str})")
            else:
                lines.append(f"{num}. {tool}()")
        elif purpose:
            lines.append(f"{num}. {purpose}")
    return "\n".join(lines)


# ---- Internal helpers ----

def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from markdown body."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return {}, text
    return fm, parts[2]


def _extract_section(text: str, pattern: str, default: str, flags: int = re.DOTALL) -> str:
    match = re.search(pattern, text, flags)
    if match:
        return match.group(1).strip()
    return default


def _parse_workflow(text: str) -> list[dict[str, Any]]:
    """Parse the workflow section into structured steps."""
    workflow_text = _extract_section(text, r"##\s+工作流\s*\n(.+?)(?=\n##|\Z)", "")
    if not workflow_text:
        return []

    steps = []
    step_pattern = r"(\d+)\.\s*(.+?)(?=\n\d+\.|\Z)"
    for match in re.finditer(step_pattern, workflow_text, re.DOTALL):
        step_num = int(match.group(1))
        step_text = match.group(2).strip()

        tool_match = re.match(r"(\w[\w.]*)\s*\((.*)\)", step_text)
        if tool_match:
            tool_name = tool_match.group(1)
            args_str = tool_match.group(2).strip()
            try:
                args = _parse_args(args_str)
            except Exception:
                args = {}
            steps.append({
                "step": step_num,
                "tool": tool_name,
                "arguments": args,
                "purpose": step_text,
            })
        else:
            steps.append({
                "step": step_num,
                "purpose": step_text,
            })
    return steps


def _parse_args(args_str: str) -> dict[str, Any]:
    """Parse simple key=value arguments string."""
    if not args_str:
        return {}
    args = {}
    for pair in args_str.split(","):
        pair = pair.strip()
        if "=" in pair:
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            args[key] = value
    return args