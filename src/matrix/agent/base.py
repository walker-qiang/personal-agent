"""Agent system: AgentDefinition for multi-agent orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..skills import SkillDefinition


@dataclass
class AgentDefinition:
    """Definition of a domain agent or commander.

    Key differences from RoleDefinition:
    - domain: "commander" | "investment" | "general" | custom
    - tools: patterns like "finance.*" or "web_search" (runtime filtering)
    - general_skills: skill names from common/ directory, available to all agents
    - domain_skills: skill names from domain-specific directory, bound to this agent
    """

    id: str
    name: str
    description: str  # short description for commander to decide delegation
    domain: str  # "commander" | "investment" | "general" | etc.
    persona: str  # system prompt prefix
    expertise: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)  # tool name patterns
    general_skills: list[str] = field(default_factory=list)  # from skills/common/
    domain_skills: list[str] = field(default_factory=list)  # from skills/{domain}/
    output_constraints: list[str] = field(default_factory=list)
    safety_rules: list[str] = field(default_factory=list)
    # LLM override: if set, domain agent uses its own LLM config instead of commander's
    llm_provider: str = ""  # deepseek | anthropic | agnes
    llm_model: str = ""  # specific model override

    @property
    def all_skills(self) -> list[str]:
        """All skill names available to this agent (general + domain)."""
        return self.general_skills + self.domain_skills

    def to_system_prompt(self) -> str:
        """Generate a system prompt from the agent definition."""
        lines = [f"你是 {self.name}。{self.persona}\n"]
        if self.expertise:
            lines.append("专业领域：")
            for item in self.expertise:
                lines.append(f"  - {item}")
            lines.append("")
        if self.safety_rules:
            lines.append("安全规则：")
            for item in self.safety_rules:
                lines.append(f"  - {item}")
            lines.append("")
        if self.output_constraints:
            lines.append("输出约束：")
            for item in self.output_constraints:
                lines.append(f"  - {item}")
            lines.append("")
        return "\n".join(lines)

    def matches_tool(self, tool_name: str) -> bool:
        """Check if this agent has access to a given tool."""
        if not self.tools:
            return False
        for pattern in self.tools:
            if pattern.endswith(".*"):
                if tool_name.startswith(pattern[:-2]):
                    return True
            elif pattern == tool_name:
                return True
        return False

    def matches_skill(self, skill_name: str) -> bool:
        """Check if this agent can use a given skill (general or domain)."""
        return skill_name in self.all_skills