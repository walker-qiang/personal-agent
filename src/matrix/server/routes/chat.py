"""Chat endpoints: streaming chat and session reset."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...chat import ChatService
from ...tools import FinanceToolError
from .sse import sse_event, sse_response

router = APIRouter()


@router.post("/chat")
async def chat(request: Request):
    chat_service: ChatService = request.app.state.chat
    trace = request.app.state.trace
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise FinanceToolError("request body must be an object")
        message = str(payload.get("message", "")).strip()
        raw_session_id = payload.get("session_id")
        session_id = str(raw_session_id).strip() if raw_session_id else None
        mode = str(payload.get("mode", "")).strip().lower()
    except (FinanceToolError, json.JSONDecodeError) as err:
        _trace_error(request, str(err))
        return JSONResponse(
            {"error": f"invalid chat request: {err}"}, status_code=400
        )

    def iter_events():
        if mode == "graph":
            stream = chat_service.stream_chat_graph(message, session_id)
        else:
            stream = chat_service.stream_chat(message, session_id)
        for event in stream:
            event_type = str(event.get("type", "message"))
            payload_data = {key: value for key, value in event.items() if key != "type"}
            yield sse_event(event_type, payload_data)

    return sse_response(iter_events())


@router.post("/reset")
async def reset(request: Request):
    chat_service: ChatService = request.app.state.chat
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise FinanceToolError("request body must be an object")
        session_id = str(payload.get("session_id", "")).strip()
        chat_service.reset(session_id)
        return JSONResponse({"ok": True})
    except (FinanceToolError, json.JSONDecodeError) as err:
        _trace_error(request, str(err))
        return JSONResponse(
            {"error": f"invalid reset request: {err}"}, status_code=400
        )


def _trace_error(request: Request, error: str) -> None:
    trace = request.app.state.trace
    from ...chat import timestamp

    trace.record(
        {
            "ok": False,
            "error": error,
            "path": request.url.path,
            "ts": timestamp(),
        }
    )