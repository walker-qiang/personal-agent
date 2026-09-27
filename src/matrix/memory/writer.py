"""Bounded background worker for memory extraction.

Replaces the bare ``threading.Thread(..., daemon=True)`` launch that used to
sit in ``ChatService._remember`` (defect B-2). A raw daemon thread is killed
mid-flight when the process exits, and its exception is swallowed by a single
``logger.warning`` — which is why a memory pipeline could silently produce
nothing for 18 sessions without leaving a trace.

This worker gives the write path three properties the raw thread lacked:

- **Backpressure**: a bounded queue. When extraction is slower than the
  conversation rate, tasks are dropped *and counted* instead of piling up
  until the process dies with them.
- **Observability**: every task records a structured outcome
  (``ok`` / ``error:<type>`` / ``dropped``) with elapsed time.
- **Drainable**: ``close()`` and an ``atexit`` hook flush in-flight work
  before the interpreter goes away.

Usage::

    worker = MemoryWriteWorker(handler)
    worker.start()
    worker.submit({"question": ..., "answer": ..., "user_id": ...})
    worker.close(timeout=5.0)
"""

from __future__ import annotations

import atexit
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Sentinel pushed by close() to unblock the worker.
_SHUTDOWN = object()


@dataclass
class WorkerStats:
    """Counters for one worker's lifetime."""

    submitted: int = 0
    completed: int = 0
    failed: int = 0
    dropped: int = 0
    last_error: str = ""
    last_error_at: float = 0.0
    last_duration_ms: float = 0.0
    errors: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "submitted": self.submitted,
            "completed": self.completed,
            "failed": self.failed,
            "dropped": self.dropped,
            "last_error": self.last_error,
            "last_duration_ms": round(self.last_duration_ms, 1),
            "errors": dict(self.errors),
        }


class MemoryWriteWorker:
    """Single-consumer queue that runs a handler off the request path."""

    def __init__(
        self,
        handler: Callable[[dict[str, Any]], None],
        max_queue: int = 64,
        name: str = "memory-writer",
    ) -> None:
        self._handler = handler
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._name = name
        self._stats = WorkerStats()
        self._lock = threading.Lock()
        self._closed = False
        self._registered_atexit = False

    # ---- lifecycle -----------------------------------------------------

    def start(self) -> None:
        """Start the consumer thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name=self._name, daemon=True,
        )
        self._thread.start()
        if not self._registered_atexit:
            atexit.register(self.drain_on_exit)
            self._registered_atexit = True
        logger.info("memory_writer: started queue_max=%d", self._queue.maxsize)

    def close(self, timeout: float = 5.0) -> int:
        """Stop accepting work and flush what is queued. Returns items run."""
        self._closed = True
        self._queue.put(_SHUTDOWN)
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        return self._stats.completed

    def drain_on_exit(self) -> None:
        """Best-effort flush registered with ``atexit`` (bounded to 3s)."""
        pending = self._queue.qsize()
        if pending <= 0:
            return
        logger.info("memory_writer: atexit drain pending=%d", pending)
        self.close(timeout=3.0)

    # ---- submission ----------------------------------------------------

    def submit(self, payload: dict[str, Any]) -> bool:
        """Enqueue a task. Returns False when the queue is full or closed."""
        if self._closed:
            with self._lock:
                self._stats.dropped += 1
            logger.warning("memory_writer: dropped task, worker closed")
            return False
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            with self._lock:
                self._stats.dropped += 1
            logger.warning(
                "memory_writer: dropped task, queue full (max=%d)",
                self._queue.maxsize,
            )
            return False
        with self._lock:
            self._stats.submitted += 1
        return True

    # ---- internals -----------------------------------------------------

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SHUTDOWN:
                # Let any remaining real work run before exiting.
                while True:
                    try:
                        leftover = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if leftover is _SHUTDOWN:
                        continue
                    self._execute(leftover)
                return
            self._execute(item)

    def _execute(self, payload: dict[str, Any]) -> None:
        started = time.time()
        try:
            self._handler(payload)
            with self._lock:
                self._stats.completed += 1
                self._stats.last_duration_ms = (time.time() - started) * 1000
        except Exception as exc:  # noqa: BLE001 - worker must never die
            elapsed = (time.time() - started) * 1000
            reason = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._stats.failed += 1
                self._stats.last_error = reason
                self._stats.last_error_at = time.time()
                self._stats.last_duration_ms = elapsed
                self._stats.errors[type(exc).__name__] = (
                    self._stats.errors.get(type(exc).__name__, 0) + 1
                )
            logger.warning(
                "memory_writer: task failed in %.0fms reason=%s", elapsed, reason,
            )

    # ---- introspection -------------------------------------------------

    @property
    def stats(self) -> WorkerStats:
        with self._lock:
            return WorkerStats(**self._stats.as_dict())

    def snapshot(self) -> dict[str, Any]:
        """Stats plus queue depth, for health endpoints and logs."""
        data = self.stats.as_dict()
        data["queued"] = self._queue.qsize()
        return data
