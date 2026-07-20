"""EvalCase and ExpectedBehavior data definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Outcome = Literal["answer", "clarify", "abstain", "tool_error"]
Difficulty = Literal["easy", "medium", "hard"]
Risk = Literal["low", "medium", "high", "critical"]


@dataclass
class ExpectedBehavior:
    """Defines what a correct agent response should look like.

    All fields are optional lists — leave empty to skip that check.
    """

    outcome: Outcome = "answer"
    must_include: list[str] = field(default_factory=list)
    may_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    expected_agent: str | None = None
    min_evidence: int = 0


@dataclass
class EvalCase:
    """A single evaluation case: user input + expected behavior."""

    case_id: str
    user_input: str
    expected: ExpectedBehavior = field(default_factory=ExpectedBehavior)
    user_id: str = "default"
    tags: list[str] = field(default_factory=list)
    difficulty: Difficulty = "easy"
    risk: Risk = "low"

    @classmethod
    def from_dict(cls, data: dict) -> EvalCase:
        """Parse an EvalCase from a JSON-serializable dict."""
        exp = data.get("expected", {})
        return cls(
            case_id=data["case_id"],
            user_input=data["user_input"],
            expected=ExpectedBehavior(
                outcome=exp.get("outcome", "answer"),
                must_include=exp.get("must_include", []),
                may_include=exp.get("may_include", []),
                must_not_include=exp.get("must_not_include", []),
                required_tools=exp.get("required_tools", []),
                forbidden_tools=exp.get("forbidden_tools", []),
                expected_agent=exp.get("expected_agent"),
                min_evidence=exp.get("min_evidence", 0),
            ),
            user_id=data.get("user_id", "default"),
            tags=data.get("tags", []),
            difficulty=data.get("difficulty", "easy"),
            risk=data.get("risk", "low"),
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-serializable dict."""
        return {
            "case_id": self.case_id,
            "user_input": self.user_input,
            "expected": {
                "outcome": self.expected.outcome,
                "must_include": self.expected.must_include,
                "may_include": self.expected.may_include,
                "must_not_include": self.expected.must_not_include,
                "required_tools": self.expected.required_tools,
                "forbidden_tools": self.expected.forbidden_tools,
                "expected_agent": self.expected.expected_agent,
                "min_evidence": self.expected.min_evidence,
            },
            "user_id": self.user_id,
            "tags": self.tags,
            "difficulty": self.difficulty,
            "risk": self.risk,
        }