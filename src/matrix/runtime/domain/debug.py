"""Ephemeral debug values for one Runtime execution.

Debug trace is intentionally separate from :mod:`events`: Runtime events are
durable recovery/audit records, while debug trace is an opt-in diagnostic view
that may be discarded with the current RunHandle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DebugTraceEvent:
    """One in-memory diagnostic record scoped to an operation."""

    sequence: int
    kind: str
    timestamp: float
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "kind": self.kind,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }
