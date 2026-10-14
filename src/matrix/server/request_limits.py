"""Bounded request parsing and machine-readable HTTP errors."""

from __future__ import annotations

import json
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse


class RequestBodyTooLarge(ValueError):
    """Raised when a request body exceeds the route's configured limit."""

    def __init__(self, limit_bytes: int):
        super().__init__(f"request body exceeds {limit_bytes} bytes")
        self.limit_bytes = limit_bytes


async def read_json_object(request: Request, *, max_bytes: int) -> dict[str, Any]:
    """Read and decode one JSON object without buffering an unbounded body."""
    content_length = request.headers.get("content-length", "").strip()
    if content_length:
        try:
            declared = int(content_length)
        except ValueError:
            declared = 0
        if declared > max_bytes:
            raise RequestBodyTooLarge(max_bytes)

    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > max_bytes:
            raise RequestBodyTooLarge(max_bytes)

    value = json.loads(bytes(body))
    if not isinstance(value, dict):
        raise ValueError("request body must be an object")
    return value


def problem_response(
    *,
    status_code: int,
    code: str,
    detail: str,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Return an RFC 9457-shaped error while preserving local error fields."""
    body: dict[str, Any] = {
        "type": f"urn:personal-system:problem:{code}",
        "title": code.replace("_", " ").title(),
        "status": status_code,
        "detail": detail,
        "error": detail,
        "code": code,
        "error_code": code,
        "retryable": retryable,
        "no_automatic_retry": not retryable,
    }
    if details:
        body["details"] = details
    return JSONResponse(
        body,
        status_code=status_code,
        media_type="application/problem+json",
    )
