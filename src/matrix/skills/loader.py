"""Skill loader: parse Markdown skill definitions from directory-based skills."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SKILL_FILE = "SKILL.md"
KNOWLEDGE_DIR = "references"
SCRIPTS_DIR = "scripts"


@dataclass
class SkillDefinition:
    """Parsed skill definition from a skill directory with SKILL.md."""

    name: str
    title: str
    description: str = ""
    workflow: list[dict[str, Any]] = field(default_factory=list)
    output_format: str = ""
    knowledge_files: list[str] = field(default_factory=list)
    script_files: list[str] = field(default_factory=list)

    @classmethod
    def from_dir(cls, skill_dir: Path) -> "SkillDefinition":
        """Parse a skill definition from a directory containing SKILL.md."""
        md_path = skill_dir / SKILL_FILE
        content = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
        name = skill_dir.name

        frontmatter, body = _split_frontmatter(content)
        title = frontmatter.get("title", name)

        workflow = _parse_workflow(body)
        output_format = _extract_section(body, r"##\s+输出格式\s*\n(.+)", "", re.DOTALL)

        # Collect knowledge files
        knowledge_files = _list_files(skill_dir / KNOWLEDGE_DIR, skill_dir)

        # Collect script files
        script_files = _list_files(skill_dir / SCRIPTS_DIR, skill_dir)

        return cls(
            name=name,
            title=title or name,
            description=frontmatter.get("description", "").strip(),
            workflow=workflow,
            output_format=output_format.strip(),
            knowledge_files=knowledge_files,
            script_files=script_files,
        )

    # Chinese negation words that indicate the user does NOT want this skill
    _NEGATION_WORDS = {"没有", "不是", "不用", "不需要", "别", "不要", "没", "无", "不", "否", "非"}

    def _has_negation(self, query: str, matched_word: str) -> bool:
        """Check if the matched word is negated in the query.

        Uses clause-based detection: splits the query by Chinese punctuation
        and only checks the prefix within the same clause as the match.
        This avoids false positives when negation applies to a different clause
        (e.g. "没有异动，但帮我做组合复盘" — "没有" negates "异动", not "组合复盘").
        """
        q = query.lower()
        w = matched_word.lower()
        idx = q.find(w)
        if idx < 0:
            return False
        # Find the clause containing the match by looking for the nearest separator before idx
        clause_start = 0
        for sep in ("，", "。", "！", "？", "；", "：", "、", "\n", ",", "."):
            sep_pos = q.rfind(sep, 0, idx)
            if sep_pos >= clause_start:
                clause_start = sep_pos + 1
        # Check the prefix within this clause for negation words
        clause_prefix = q[clause_start:idx]
        for neg in self._NEGATION_WORDS:
            if neg in clause_prefix:
                return True
        return False

    def matches(self, query: str) -> bool:
        """Check if the query matches this skill's title or description.

        Returns False when the matched keyword is negated (e.g. "今天没有异动"
        will NOT match anomaly-diagnosis even though "异动" is in the description).
        """
        q = query.lower()
        text = (self.title + " " + self.description).lower()
        words = [w for w in re.split(r"[\s,，。！？、；：""''（）\(\)]+", text) if len(w) >= 2]

        # Exact word match
        for w in words:
            if w in q:
                if not self._has_negation(query, w):
                    return True
        # Substring match for short queries
        if len(q) >= 2 and q in text:
            return True
        # 2-gram fuzzy match
        for w in words:
            if len(w) > 2:
                for i in range(len(w) - 1):
                    bigram = w[i:i + 2]
                    if bigram in q:
                        if not self._has_negation(query, bigram):
                            return True
        return False

    def read_knowledge(self, skill_dir: Path) -> list[dict[str, str]]:
        """Read all knowledge files content."""
        result = []
        kdir = skill_dir / KNOWLEDGE_DIR
        for rel in self.knowledge_files:
            f = kdir / rel
            if f.exists():
                result.append({"name": rel, "content": f.read_text(encoding="utf-8")})
        return result

    def read_script(self, skill_dir: Path, script_name: str) -> str | None:
        """Read a script file content."""
        f = skill_dir / SCRIPTS_DIR / script_name
        if f.exists():
            return f.read_text(encoding="utf-8")
        return None

    def read_knowledge_file(self, skill_dir: Path, filename: str) -> str | None:
        """Read a single knowledge file content."""
        f = skill_dir / KNOWLEDGE_DIR / filename
        if f.exists():
            return f.read_text(encoding="utf-8")
        return None

    @staticmethod
    def read_knowledge_static(skill_dir: Path) -> list[dict[str, str]]:
        """Read all knowledge files from a skill directory (static)."""
        result = []
        kdir = skill_dir / KNOWLEDGE_DIR
        if not kdir.exists():
            return result
        for f in sorted(kdir.iterdir()):
            if f.is_file():
                result.append({"name": f.name, "content": f.read_text(encoding="utf-8")})
        return result

    @staticmethod
    def read_knowledge_file_static(skill_dir: Path, filename: str) -> str | None:
        """Read a single knowledge file (static)."""
        f = skill_dir / KNOWLEDGE_DIR / filename
        if f.exists():
            return f.read_text(encoding="utf-8")
        return None

    @staticmethod
    def read_script_static(skill_dir: Path, filename: str) -> str | None:
        """Read a script file (static)."""
        f = skill_dir / SCRIPTS_DIR / filename
        if f.exists():
            return f.read_text(encoding="utf-8")
        return None


def load_skills(skills_dir: Path) -> list[SkillDefinition]:
    """Load all skills from a directory of skill directories."""
    skills = []
    if not skills_dir.exists():
        return skills
    for entry in sorted(skills_dir.iterdir()):
        if not entry.is_dir():
            continue
        md_path = entry / SKILL_FILE
        if not md_path.exists():
            continue
        try:
            skills.append(SkillDefinition.from_dir(entry))
        except (yaml.YAMLError, OSError, ValueError, KeyError) as err:
            import logging
            logging.getLogger("matrix").warning("Failed to load skill dir %s: %s", entry.name, err)
            continue
    return skills


def render_skill(skill: SkillDefinition) -> str:
    """Serialize a skill's SKILL.md back to Markdown with YAML frontmatter."""
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


def create_skill_dir(skills_dir: Path, skill: SkillDefinition) -> Path:
    """Create a new skill directory with SKILL.md."""
    skill_dir = skills_dir / skill.name
    skill_dir.mkdir(parents=True, exist_ok=False)
    (skill_dir / SKILL_FILE).write_text(render_skill(skill), encoding="utf-8")
    return skill_dir


def update_skill_dir(skills_dir: Path, skill_name: str, skill: SkillDefinition) -> Path:
    """Update SKILL.md in an existing skill directory."""
    skill_dir = skills_dir / skill_name
    if not skill_dir.is_dir():
        raise FileNotFoundError(f"skill dir not found: {skill_dir}")
    (skill_dir / SKILL_FILE).write_text(render_skill(skill), encoding="utf-8")
    return skill_dir


def delete_skill_dir(skills_dir: Path, skill_name: str) -> None:
    """Delete a skill directory entirely."""
    skill_dir = skills_dir / skill_name
    if skill_dir.is_dir():
        shutil.rmtree(skill_dir)


def write_knowledge(skills_dir: Path, skill_name: str, filename: str, content: str) -> Path:
    """Write a knowledge file into the skill's knowledge/ directory."""
    kdir = skills_dir / skill_name / KNOWLEDGE_DIR
    kdir.mkdir(parents=True, exist_ok=True)
    f = kdir / filename
    f.write_text(content, encoding="utf-8")
    return f


def write_script(skills_dir: Path, skill_name: str, filename: str, content: str) -> Path:
    """Write a script file into the skill's scripts/ directory."""
    sdir = skills_dir / skill_name / SCRIPTS_DIR
    sdir.mkdir(parents=True, exist_ok=True)
    f = sdir / filename
    f.write_text(content, encoding="utf-8")
    return f


def delete_knowledge(skills_dir: Path, skill_name: str, filename: str) -> None:
    """Delete a knowledge file."""
    f = skills_dir / skill_name / KNOWLEDGE_DIR / filename
    if f.exists():
        f.unlink()


def delete_script(skills_dir: Path, skill_name: str, filename: str) -> None:
    """Delete a script file."""
    f = skills_dir / skill_name / SCRIPTS_DIR / filename
    if f.exists():
        f.unlink()


# ---- Internal helpers ----

def _list_files(directory: Path, base: Path) -> list[str]:
    """List relative file paths in a directory, recursively."""
    if not directory.exists():
        return []
    files = []
    for f in sorted(directory.rglob("*")):
        if f.is_file():
            files.append(str(f.relative_to(directory)))
    return files


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
            except (ValueError, json.JSONDecodeError) as err:
                import logging
                logging.getLogger("matrix").warning(
                    "Failed to parse args for tool %s: %s", tool_name, err
                )
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