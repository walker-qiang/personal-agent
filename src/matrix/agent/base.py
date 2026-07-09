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
    - skills: skill names, can be general or domain-specific
    """

    id: str
    name: str
    description: str  # short description for commander to decide delegation
    domain: str  # "commander" | "investment" | "general" | etc.
    persona: str  # system prompt prefix
    expertise: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)  # tool name patterns
    skills: list[str] = field(default_factory=list)  # skill names
    output_constraints: list[str] = field(default_factory=list)
    safety_rules: list[str] = field(default_factory=list)

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
        """Check if this agent can use a given skill."""
        return skill_name in self.skills