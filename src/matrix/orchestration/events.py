"""Structured event types for Agent orchestration.

Replaces string-based event types with typed dataclasses.
Events flow through queue.Queue to SSE consumers.

4-layer nesting (inspired by Pi-Agent):
  Agent → Turn → Message → Tool Execution
Each layer has start / update / end events.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class AgentEvent:
    """Base event type."""
    type: str
    timestamp: float = field(default_factory=time.time)
    _legacy_payload: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False, compare=False,
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for SSE/JSON transport."""
        d = asdict(self)
        # Remove None values to keep payload compact
        return {k: v for k, v in d.items() if v is not None}

    def __getitem__(self, index: int) -> Any:
        """Expose the historical ``(type, payload)`` indexing shape.

        Internal queues now carry structured events, but a few integrations
        still inspect queued values as tuples.  Keeping this read-only bridge
        makes that migration incremental without changing the SSE contract.
        """
        if index == 0:
            return self.type
        if index == 1:
            if self._legacy_payload:
                return dict(self._legacy_payload)
            payload = self.to_dict()
            payload.pop("type", None)
            payload.pop("timestamp", None)
            return payload
        raise IndexError(index)


# ---- Layer 1: Agent lifecycle ----

@dataclass
class AgentStartEvent(AgentEvent):
    type: str = "agent_start"

@dataclass
class AgentEndEvent(AgentEvent):
    type: str = "agent_end"
    final_answer: str = ""


# ---- Layer 2: Turn lifecycle ----

@dataclass
class TurnStartEvent(AgentEvent):
    type: str = "turn_start"
    agent_id: str = ""
    iteration: int = 0

@dataclass
class TurnEndEvent(AgentEvent):
    type: str = "turn_end"
    agent_id: str = ""
    answer: str = ""


# ---- Layer 3: Message lifecycle ----

@dataclass
class ThinkingEvent(AgentEvent):
    """LLM is producing text output."""
    type: str = "thinking"
    content: str = ""


# ---- Layer 4: Tool execution lifecycle ----

@dataclass
class ToolCallEvent(AgentEvent):
    """A tool is about to be executed."""
    type: str = "tool_call"
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolResultEvent(AgentEvent):
    """A tool has finished executing."""
    type: str = "tool_result"
    name: str = ""
    result: Any = None
    error: str = ""
    elapsed_ms: float = 0


# ---- Session-level events (beyond Pi's 10, Matrix-specific) ----

@dataclass
class ProgressEvent(AgentEvent):
    """Human-readable progress update for multi-step tasks."""
    type: str = "progress"
    message: str = ""
    step: int = 0
    total: int = 0

@dataclass
class PlanCreatedEvent(AgentEvent):
    """Commander created a delegation plan."""
    type: str = "plan_created"
    plan_type: str = ""
    total_steps: int = 0
    steps: list[dict[str, Any]] = field(default_factory=list)

@dataclass
class StepStartEvent(AgentEvent):
    """A plan step is starting."""
    type: str = "step_start"
    step: int = 0
    total: int = 0
    agent: str = ""
    task: str = ""

@dataclass
class StepDoneEvent(AgentEvent):
    """A plan step completed."""
    type: str = "step_done"
    step: int = 0
    total: int = 0
    result_preview: str = ""

@dataclass
class StepErrorEvent(AgentEvent):
    """A plan step failed."""
    type: str = "step_error"
    step: int = 0
    error: str = ""

@dataclass
class ReplanEvent(AgentEvent):
    """Plan was revised."""
    type: str = "replan"
    reason: str = ""
    attempt: int = 0


@dataclass
class DebugTraceEvent(AgentEvent):
    """Ephemeral diagnostic data for the current Runtime execution."""
    type: str = "debug_trace"
    operation_id: str = ""
    event: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfirmRequiredEvent(AgentEvent):
    """A side-effecting action is waiting for user approval."""
    type: str = "confirm_required"
    actions: list[dict[str, Any]] = field(default_factory=list)
    session_id: str = ""


# ---- Union type ----

AgentSessionEvent = (
    AgentStartEvent | AgentEndEvent |
    TurnStartEvent | TurnEndEvent |
    ThinkingEvent |
    ToolCallEvent | ToolResultEvent |
    ProgressEvent |
    PlanCreatedEvent | StepStartEvent | StepDoneEvent | StepErrorEvent | ReplanEvent
    | DebugTraceEvent | ConfirmRequiredEvent
)


# ---- Factory functions for backward compatibility ----

def make_event(event_type: str, payload: dict[str, Any]) -> AgentSessionEvent:
    """Create a structured event from a string type + payload dict.

    This provides backward compatibility with existing _push_event calls
    that use string-based event types.
    """
    factories = {
        "agent_start": lambda p: AgentStartEvent(),
        "agent_end": lambda p: AgentEndEvent(final_answer=p.get("final_answer", "")),
        "turn_start": lambda p: TurnStartEvent(agent_id=p.get("agent_id", ""), iteration=p.get("iteration", 0)),
        "turn_end": lambda p: TurnEndEvent(agent_id=p.get("agent_id", ""), answer=p.get("answer", "")),
        "thinking": lambda p: ThinkingEvent(content=p.get("content", "")),
        "tool_call": lambda p: ToolCallEvent(name=p.get("name", ""), args=p.get("args", {})),
        "tool_result": lambda p: ToolResultEvent(
            name=p.get("name", ""), result=p.get("result"),
            error=p.get("error", ""), elapsed_ms=p.get("elapsed_ms", 0),
        ),
        "progress": lambda p: ProgressEvent(
            message=p.get("message", ""), step=p.get("step", 0), total=p.get("total", 0),
        ),
        "plan_created": lambda p: PlanCreatedEvent(
            plan_type=p.get("plan_type", ""), total_steps=p.get("total_steps", 0),
            steps=p.get("steps", []),
        ),
        "step_start": lambda p: StepStartEvent(
            step=p.get("step", 0), total=p.get("total", 0),
            agent=p.get("agent", ""), task=p.get("task", ""),
        ),
        "step_done": lambda p: StepDoneEvent(
            step=p.get("step", 0), total=p.get("total", 0),
            result_preview=p.get("result_preview", ""),
        ),
        "step_error": lambda p: StepErrorEvent(step=p.get("step", 0), error=p.get("error", "")),
        "replan": lambda p: ReplanEvent(reason=p.get("reason", ""), attempt=p.get("attempt", 0)),
        "debug_trace": lambda p: DebugTraceEvent(
            operation_id=p.get("operation_id", ""), event=p.get("event", {}),
        ),
        "confirm_required": lambda p: ConfirmRequiredEvent(
            actions=p.get("actions", []), session_id=p.get("session_id", ""),
        ),
    }
    factory = factories.get(event_type)
    if factory:
        event = factory(payload)
        # Keep the original payload for older integrations that inspect the
        # queue using the historical (event_type, payload) shape.  Typed
        # consumers use the dataclass fields and to_dict() instead.
        event._legacy_payload = dict(payload)
        return event
    # Unknown event type: wrap in base AgentEvent while retaining its payload
    # for the compatibility adapter and future event-specific consumers.
    event = AgentEvent(type=event_type)
    event._legacy_payload = dict(payload)
    return event
