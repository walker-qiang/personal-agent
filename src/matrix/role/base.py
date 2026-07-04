"""Role system: structured role definitions for agent personas."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RoleDefinition:
    """Structured definition of an agent role (岗位)."""

    id: str
    name: str
    persona: str
    expertise: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    output_constraints: list[str] = field(default_factory=list)
    safety_rules: list[str] = field(default_factory=list)

    def to_system_prompt(self) -> str:
        """Generate a system prompt from the role definition."""
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