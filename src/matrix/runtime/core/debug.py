"""In-memory, opt-in Runtime debug trace collector."""

from __future__ import annotations

import re
import threading
import time
from typing import Any

from ..domain.debug import DebugTraceEvent


_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|cookie|password|secret|token)", re.I
)
_MAX_TEXT = 4000


class EphemeralDebugTrace:
    """Thread-safe trace buffer that is never connected to SQLite.

    It is deliberately small and operation-scoped.  The buffer is exposed by
    ``RunHandle`` only while the handle is alive; callers may explicitly clear
    it after rendering diagnostics.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[DebugTraceEvent] = []

    def emit(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._events.append(DebugTraceEvent(
                sequence=len(self._events) + 1,
                kind=kind,
                timestamp=time.time(),
                payload=_sanitize(payload or {}),
            ))

    def snapshot(self) -> list[DebugTraceEvent]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


def _sanitize(value: Any, key: str = "") -> Any:
    if _SECRET_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, str):
        return value if len(value) <= _MAX_TEXT else value[:_MAX_TEXT] + "…"
    if isinstance(value, dict):
        return {str(k): _sanitize(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value
