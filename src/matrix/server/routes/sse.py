"""SSE streaming response utilities."""

from __future__ import annotations

import json
from typing import Any

from starlette.responses import StreamingResponse


def sse_event(event: str, payload: dict[str, Any]) -> bytes:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def sse_response(events):
    """Create a StreamingResponse from an iterator of SSE events."""
    return StreamingResponse(
        events,
        media_type="text/event-stream; charset=utf-8",
        headers={
            "cache-control": "no-cache",
            "x-accel-buffering": "no",
        },
    )