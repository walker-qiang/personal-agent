"""Skill loader: parse Markdown skill definitions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SkillDefinition:
    """Parsed skill definition from a Markdown file."""

    name: str
    title: str
    description: str = ""
    trigger_keywords: list[str] = field(default_factory=list)
    workflow: list[dict[str, Any]] = field(default_factory=list)
    output_format: str = ""

    @classmethod
    def from_markdown(cls, path: Path) -> "SkillDefinition":
        """Parse a skill definition from a Markdown file."""
        content = path.read_text(encoding="utf-8")
        name = path.stem

        title = _extract_section(content, r"#\s+(.+)", "", re.MULTILINE)
        description = _extract_section(content, r"##\s+简介\s*\n(.+?)(?=\n##|\Z)", "")
        trigger_text = _extract_section(content, r"##\s+触发条件\s*\n(.+?)(?=\n##|\Z)", "")
        trigger_keywords = [
            kw.strip().strip("`")
            for kw in re.findall(r"[-*]\s*`?([^`\n]+)`?", trigger_text)
            if kw.strip()
        ]
        workflow = _parse_workflow(content)
        output_format = _extract_section(content, r"##\s+输出格式\s*\n(.+?)(?=\n##|\Z)", "")

        return cls(
            name=name,
            title=title or name,
            description=description.strip(),
            trigger_keywords=trigger_keywords,
            workflow=workflow,
            output_format=output_format.strip(),
        )

    def matches(self, query: str) -> bool:
        """Check if the query matches this skill's trigger keywords."""
        query_lower = query.lower()
        return any(kw.lower() in query_lower for kw in self.trigger_keywords)


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
    # Match numbered steps like "1. tool_name(args)" or "1. 描述"
    step_pattern = r"(\d+)\.\s*(.+?)(?=\n\d+\.|\Z)"
    for match in re.finditer(step_pattern, workflow_text, re.DOTALL):
        step_num = int(match.group(1))
        step_text = match.group(2).strip()

        # Try to parse tool call: "tool_name(parameters)"
        tool_match = re.match(r"(\w[\w.]*)\s*\((.*)\)", step_text)
        if tool_match:
            tool_name = tool_match.group(1)
            args_str = tool_match.group(2).strip()
            try:
                # Simple key=value parsing
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