"""Observability: structured tracing and event logging."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class TraceLogger:
    """JSONL trace logger — appends structured events to a file. Thread-safe."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def record(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
                fh.write("\n")