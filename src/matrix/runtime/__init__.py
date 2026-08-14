"""Independent single-Agent runtime contracts.

The runtime package is intentionally not wired into the production chat path
yet.  WP1 freezes the lower-level contracts; execution is added in WP2.
"""

from .core.runtime import AgentRuntime, RunHandle
from .domain.events import RuntimeEvent, RuntimeEventType
from .domain.requests import ExecutionOptions, ResumeInput, RunRequest
from .domain.results import RunOutcome, RunResult

__all__ = [
    "AgentRuntime",
    "ExecutionOptions",
    "ResumeInput",
    "RunHandle",
    "RunOutcome",
    "RunRequest",
    "RunResult",
    "RuntimeEvent",
    "RuntimeEventType",
]
