"""Dynamic system prompt assembly: multi-layer AGENTS.md recursion + skills lazy-load.

Loads AGENTS.md / CLAUDE.md files from cwd upward to root, merges them
in outer-to-inner order (general → specific), and wraps in XML tags.

Skills are lazy-loaded: only name + description go into the prompt,
the LLM can read full SKILL.md content on demand.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("matrix.orchestration.context")

# TTL cache for project context files (avoid re-reading AGENTS.md every turn)
_cache: dict[str, tuple[float, list[tuple[Path, str]]]] = {}
_CACHE_TTL = 60  # seconds

# File names to look for, in priority order
_CONTEXT_FILE_NAMES = ("AGENTS.md", "CLAUDE.md")


def load_project_context_files(cwd: Path | None = None) -> list[tuple[Path, str]]:
    """Find AGENTS.md / CLAUDE.md files from cwd upward to root.

    Returns [(path, content), ...] in outer-to-inner order
    (ancestor directories first, cwd last).
    """
    if cwd is None:
        cwd = Path.cwd()

    cache_key = str(cwd)
    now = time.time()
    cached = _cache.get(cache_key)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

    results: list[tuple[Path, str]] = []
    seen_paths: set[str] = set()

    for path in [cwd, *cwd.parents]:
        for name in _CONTEXT_FILE_NAMES:
            f = path / name
            try:
                resolved = str(f.resolve())
            except (OSError, RuntimeError):
                resolved = str(f)
            if resolved in seen_paths:
                continue
            if f.exists() and f.is_file():
                try:
                    content = f.read_text(encoding="utf-8").strip()
                    if content:
                        results.append((f, content))
                        seen_paths.add(resolved)
                        break  # One file per directory
                except (OSError, UnicodeDecodeError):
                    pass

    _cache[cache_key] = (now, results)
    return results


def build_project_context_section(cwd: Path | None = None) -> str:
    """Build a <project_context> XML section from multi-layer AGENTS.md files.

    Returns empty string if no context files found.
    """
    files = load_project_context_files(cwd)
    if not files:
        return ""

    parts = ["<project_context>"]
    parts.append("Project-specific instructions and guidelines:")
    for path, content in files:
        parts.append(f'<project_instructions path="{path}">')
        parts.append(content)
        parts.append("</project_instructions>")
    parts.append("</project_context>")
    return "\n".join(parts)


def build_skills_section(
    agent_def: Any,
    agent_registry: Any,
) -> str:
    """Build an <available_skills> section with lazy-load清单.

    Only includes skill name + description. The LLM should use the
    read tool to load full SKILL.md content when needed.
    """
    try:
        skills = agent_registry.load_skills_for_agent(agent_def.id)
    except Exception:
        return ""

    if not skills:
        return ""

    parts = ["<available_skills>"]
    parts.append(
        "Use the read tool to load a skill's SKILL.md file when the task "
        "matches its description."
    )
    for skill in skills:
        parts.append(f'<skill name="{skill.name}">{skill.description}</skill>')
    parts.append("</available_skills>")
    return "\n".join(parts)


def enrich_system_prompt(
    base_prompt: str,
    cwd: Path | None = None,
    agent_def: Any = None,
    agent_registry: Any = None,
) -> str:
    """Enrich a base system prompt with project context and skills.

    Appends <project_context> and <available_skills> sections.
    """
    sections: list[str] = [base_prompt]

    ctx = build_project_context_section(cwd)
    if ctx:
        sections.append(ctx)

    if agent_def is not None and agent_registry is not None:
        skills = build_skills_section(agent_def, agent_registry)
        if skills:
            sections.append(skills)

    return "\n\n".join(sections)
