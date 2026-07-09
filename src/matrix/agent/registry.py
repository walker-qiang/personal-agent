"""Agent registry: manages agent definitions, tool bindings, and skill bindings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..skills import SkillDefinition, load_skills
from ..tools import ToolRegistry
from .base import AgentDefinition


class AgentRegistry:
    """Registry for Commander + Domain Agents.

    Responsibilities:
    - Register and discover agents
    - Build per-agent ToolRegistry based on tool patterns
    - Load per-agent skills (general + domain)
    - Provide agent info for Commander's planning

    Skills directory supports two structures:
    1. Domain-based: skills/{common,investment,general}/skill-name/
    2. Flat: skills/skill-name/ (all skills in one directory)
    """

    def __init__(self, skills_base_dir: str | Path = "skills") -> None:
        self._agents: dict[str, AgentDefinition] = {}
        self._skills_base_dir = Path(skills_base_dir)
        # Cache loaded skills: {dir_path: [SkillDefinition]}
        self._skills_cache: dict[str, list[SkillDefinition]] = {}
        # Detect structure: True if flat (skills directly in base dir), False if domain-based
        self._flat_structure = self._detect_structure()

    def _detect_structure(self) -> bool:
        """Detect if skills_base_dir uses flat or domain-based structure."""
        if not self._skills_base_dir.exists():
            return False
        # If base dir contains subdirs that look like domain dirs, it's domain-based
        domain_dirs = {"common", "investment", "general"}
        contents = {p.name for p in self._skills_base_dir.iterdir() if p.is_dir()}
        if contents & domain_dirs:
            return False
        # If it contains SKILL.md subdirs, it's flat
        for p in self._skills_base_dir.iterdir():
            if p.is_dir() and (p / "SKILL.md").exists():
                return True
        return False

    # ---- Registration ----

    def register(self, agent: AgentDefinition) -> None:
        """Register an agent definition."""
        if agent.id in self._agents:
            raise ValueError(f"agent already registered: {agent.id}")
        self._agents[agent.id] = agent

    def register_all(self, agents: list[AgentDefinition]) -> None:
        """Register multiple agent definitions."""
        for agent in agents:
            self.register(agent)

    # ---- Query ----

    def get(self, agent_id: str) -> AgentDefinition | None:
        """Get an agent by id."""
        return self._agents.get(agent_id)

    @property
    def commander(self) -> AgentDefinition | None:
        """Get the commander agent."""
        return self.get("commander")

    def list_domain_agents(self) -> list[AgentDefinition]:
        """List all non-commander (domain) agents."""
        return [a for a in self._agents.values() if a.domain != "commander"]

    def list_all(self) -> list[AgentDefinition]:
        """List all registered agents."""
        return list(self._agents.values())

    def agents_for_commander(self) -> list[dict[str, str]]:
        """Return agent descriptions for the commander's planning prompt."""
        return [
            {"id": a.id, "name": a.name, "description": a.description, "domain": a.domain}
            for a in self.list_domain_agents()
        ]

    # ---- Tool Binding ----

    def build_tool_registry(self, agent_id: str, full_registry: ToolRegistry) -> ToolRegistry:
        """Build a filtered ToolRegistry for a specific agent.

        Only includes tools that match the agent's tool patterns.
        """
        agent = self._agents.get(agent_id)
        if agent is None:
            raise ValueError(f"agent not found: {agent_id}")
        if not agent.tools:
            return ToolRegistry()

        filtered = ToolRegistry()
        for tool_name in full_registry.tool_names():
            if agent.matches_tool(tool_name):
                tool_def = full_registry.get(tool_name)
                if tool_def is not None:
                    filtered.register(tool_def)
        return filtered

    # ---- Skill Binding ----

    def load_skills_for_agent(self, agent_id: str) -> list[SkillDefinition]:
        """Load skills for a specific agent (general + domain).

        Supports both flat and domain-based directory structures.
        """
        agent = self._agents.get(agent_id)
        if agent is None:
            raise ValueError(f"agent not found: {agent_id}")

        all_skills: list[SkillDefinition] = []

        if self._flat_structure:
            # Flat structure: all skills in one directory, filter by name
            if agent.all_skills:
                all_skills = self._load_skills_dir(self._skills_base_dir)
                all_skills = [s for s in all_skills if s.name in agent.all_skills]
        else:
            # Domain-based structure: skills/common/ + skills/{domain}/
            if agent.general_skills:
                common_dir = self._skills_base_dir / "common"
                common_skills = self._load_skills_dir(common_dir)
                all_skills.extend(
                    s for s in common_skills if s.name in agent.general_skills
                )

            if agent.domain_skills and agent.domain != "commander":
                domain_dir = self._skills_base_dir / agent.domain
                domain_skills = self._load_skills_dir(domain_dir)
                all_skills.extend(
                    s for s in domain_skills if s.name in agent.domain_skills
                )

        return all_skills

    def list_all_skills(self) -> list[SkillDefinition]:
        """List all skills across all directories."""
        if self._flat_structure:
            return self._load_skills_dir(self._skills_base_dir)
        # Domain-based: load from all domain subdirs
        all_skills: list[SkillDefinition] = []
        for domain_dir in self._skills_base_dir.iterdir():
            if domain_dir.is_dir() and not domain_dir.name.startswith("."):
                all_skills.extend(self._load_skills_dir(domain_dir))
        return all_skills

    def _load_skills_dir(self, skills_dir: Path) -> list[SkillDefinition]:
        """Load skills from a directory, with caching."""
        if not skills_dir.exists():
            return []
        cache_key = str(skills_dir.resolve())
        if cache_key not in self._skills_cache:
            self._skills_cache[cache_key] = load_skills(skills_dir)
        return self._skills_cache[cache_key]

    def reload_skills(self) -> None:
        """Clear the skills cache and re-detect structure."""
        self._skills_cache.clear()
        self._flat_structure = self._detect_structure()

    # ---- Build configurable for LangGraph ----

    def build_configurable(self, agent_id: str, full_tools: ToolRegistry) -> dict[str, Any]:
        """Build the configurable dict for a LangGraph agent node.

        Includes: filtered tools, agent skills, agent definition, etc.
        """
        agent = self._agents.get(agent_id)
        if agent is None:
            raise ValueError(f"agent not found: {agent_id}")

        agent_tools = self.build_tool_registry(agent_id, full_tools)
        agent_skills = self.load_skills_for_agent(agent_id)

        return {
            "agent": agent,
            "tools": agent_tools,
            "skills": agent_skills,
            "skills_dir": self._skills_base_dir,
        }