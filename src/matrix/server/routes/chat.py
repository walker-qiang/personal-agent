"""Chat endpoints: streaming chat and session reset."""

from __future__ import annotations

import json
import logging
import os
import time
from urllib.parse import unquote_plus

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from ...chat import ChatService
from ...context.budget import measure_budget
from ...tools import FinanceToolError
from ..request_limits import (
    RequestBodyTooLarge,
    problem_response,
    read_json_object,
)
from .sse import sse_event, sse_response

logger = logging.getLogger("matrix.chat")
router = APIRouter()


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


_CHAT_REQUEST_MAX_BYTES = _int_env("MATRIX_CHAT_REQUEST_MAX_BYTES", 128 * 1024)
_CHAT_INPUT_TOKEN_LIMIT = _int_env("MATRIX_CHAT_INPUT_TOKENS", 24000)


def _get_user_id(request: Request) -> str:
    """Extract user_id from request state (set by AuthMiddleware)."""
    return getattr(request.state, "user_id", "default")


def _check_input_guard(request: Request, message: str) -> tuple[bool, str]:
    """Return whether the message passes the input guard and the failure reason."""
    measurement = measure_budget([message])
    if measurement.input_tokens > _CHAT_INPUT_TOKEN_LIMIT:
        logger.warning(
            "input_guard_blocked: reason=input_context_too_large chars=%d bytes=%d "
            "estimated_tokens=%d token_limit=%d",
            len(message), measurement.input_bytes, measurement.input_tokens,
            _CHAT_INPUT_TOKEN_LIMIT,
        )
        return False, "input_context_too_large"

    guardrails = getattr(request.app.state, "guardrails", None)
    if guardrails is None or guardrails.input is None:
        return True, ""
    result = guardrails.input.check(message, user_id=_get_user_id(request))
    if result.flags:
        logger.warning(
            "input_guard_flags: flags=%s chars=%d bytes=%d estimated_tokens=%d",
            result.flags, len(message), measurement.input_bytes,
            measurement.input_tokens,
        )
    if not result.allowed:
        logger.warning(
            "input_guard_blocked: reason=%s flags=%s",
            result.reason, result.flags,
        )
        return False, result.reason
    return True, ""


def _input_guard_message(reason: str) -> str:
    if reason == "message_too_long":
        return "请求内容过长，请缩短研究上下文后重试"
    if reason == "input_context_too_large":
        return "请求上下文超过模型预算，请缩短研究上下文后重试"
    return "请求被安全策略拦截"


def _guard_details(request: Request, message: str, reason: str) -> dict[str, int | str]:
    measurement = measure_budget([message])
    guardrails = getattr(request.app.state, "guardrails", None)
    input_limit = getattr(getattr(guardrails, "input", None), "_max_length", 0)
    return {
        "reason": reason,
        "message_chars": len(message),
        "message_bytes": measurement.input_bytes,
        "estimated_input_tokens": measurement.input_tokens,
        "input_bytes_limit": input_limit,
        "input_token_limit": _CHAT_INPUT_TOKEN_LIMIT,
    }


def _guard_sse_response(
    request: Request,
    message: str,
    reason: str,
    session_id: str | None = None,
):
    details = _guard_details(request, message, reason)
    error_code = (
        "input_context_too_large"
        if reason in {"message_too_long", "input_context_too_large"}
        else "input_blocked"
    )

    def iter_events():
        yield sse_event("error", {
            "message": _input_guard_message(reason),
            "code": error_code,
            "error_code": error_code,
            "retryable": False,
            "details": details,
        })
        yield sse_event("done", {
            "session_id": session_id or "unknown",
            "duration_ms": 0,
        })

    return sse_response(iter_events())


@router.post("/chat")
async def chat(request: Request):
    chat_service: ChatService = request.app.state.chat
    trace = request.app.state.trace
    try:
        payload = await read_json_object(
            request, max_bytes=_CHAT_REQUEST_MAX_BYTES,
        )
        message = str(payload.get("message", "")).strip()
        raw_session_id = payload.get("session_id")
        session_id = str(raw_session_id).strip() if raw_session_id else None
        file_id = str(payload.get("file_id", "")).strip() or None
        mode = str(payload.get("mode", "")).strip()
        agent_mode = str(payload.get("agent_mode", "")).strip()
        preset = str(payload.get("preset", "")).strip()
        debug_trace = payload.get("debug_trace") is True
    except RequestBodyTooLarge as err:
        return problem_response(
            status_code=413,
            code="request_body_too_large",
            detail="chat request body exceeds the configured limit",
            details={"max_bytes": err.limit_bytes},
        )
    except (FinanceToolError, json.JSONDecodeError, ValueError) as err:
        _trace_error(request, str(err))
        return JSONResponse(
            {"error": f"invalid chat request: {err}"}, status_code=400
        )

    user_id = _get_user_id(request)

    # ---- INPUT GUARD ----
    allowed, guard_reason = _check_input_guard(request, message)
    if not allowed:
        response = _guard_sse_response(request, message, guard_reason, session_id)
        return response
    # ---- END INPUT GUARD ----

    def iter_events():
        for event in chat_service.stream_chat(
            message, session_id, user_id=user_id, file_id=file_id, mode=mode,
            agent_mode=agent_mode, preset=preset,
            debug_trace=debug_trace,
        ):
            event_type = str(event.get("type", "message"))
            payload_data = {key: value for key, value in event.items() if key != "type"}
            yield sse_event(event_type, payload_data)

    return sse_response(iter_events())


@router.get("/chat/stream")
async def chat_stream(
    request: Request,
    message: str = Query(..., description="User message"),
    session_id: str = Query(default="", description="Session ID"),
    file_id: str = Query(default="", description="Uploaded file ID"),
    agent_mode: str = Query(default="", description="read_only or writeback"),
    preset: str = Query(default="", description="Named Agent preset"),
    debug_trace: bool = Query(default=False, description="Emit ephemeral debug trace"),
):
    """SSE streaming via EventSource (GET). Compatible with all browsers."""
    chat_service: ChatService = request.app.state.chat
    message = unquote_plus(message).strip()
    session_id = session_id.strip() or None
    file_id = file_id.strip() or None
    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)

    user_id = _get_user_id(request)

    # ---- INPUT GUARD ----
    allowed, guard_reason = _check_input_guard(request, message)
    if not allowed:
        return _guard_sse_response(request, message, guard_reason, session_id)
    # ---- END INPUT GUARD ----

    def iter_events():
        for event in chat_service.stream_chat(
            message, session_id, user_id=user_id, file_id=file_id,
            agent_mode=agent_mode, preset=preset,
            debug_trace=debug_trace,
        ):
            event_type = str(event.get("type", "message"))
            payload_data = {key: value for key, value in event.items() if key != "type"}
            yield sse_event(event_type, payload_data)

    return sse_response(iter_events())


@router.post("/chat/confirm")
async def chat_confirm(request: Request):
    """Resume a paused chat after user confirms or skips high-risk actions."""
    chat_service: ChatService = request.app.state.chat
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise FinanceToolError("request body must be an object")
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            return JSONResponse(
                {"error": "session_id is required"}, status_code=400
            )
        raw_decisions = payload.get("decisions", {})
        decisions = (
            {
                str(key): str(value)
                for key, value in raw_decisions.items()
                if str(key) and str(value) in ("approve", "skip")
            }
            if isinstance(raw_decisions, dict) else {}
        )
        approval_set_id = str(payload.get("approval_set_id", "")).strip()
        if not approval_set_id or not decisions:
            return JSONResponse(
                {"error": "approval_set_id and decisions are required"},
                status_code=400,
            )
        expected_approval_set_version = payload.get("expected_approval_set_version")
        expected_operation_version = payload.get("expected_operation_version")
        idempotency_key = str(payload.get("idempotency_key", "")).strip()
    except (FinanceToolError, json.JSONDecodeError) as err:
        return JSONResponse(
            {"error": f"invalid confirm request: {err}"}, status_code=400
        )

    def iter_events():
        for event in chat_service.resume_chat(
            session_id,
            user_id=_get_user_id(request),
            approval_set_id=approval_set_id,
            decisions=decisions,
            expected_approval_set_version=expected_approval_set_version,
            expected_operation_version=expected_operation_version,
            idempotency_key=idempotency_key,
        ):
            event_type = str(event.get("type", "message"))
            payload_data = {key: value for key, value in event.items() if key != "type"}
            yield sse_event(event_type, payload_data)

    return sse_response(iter_events())


@router.get("/reset")
async def reset_get(
    request: Request,
    session_id: str = Query(default="", description="Session ID"),
):
    chat_service: ChatService = request.app.state.chat
    session_id = session_id.strip()
    if session_id and not chat_service.reset(session_id, user_id=_get_user_id(request)):
        return JSONResponse({"error": "session not found"}, status_code=404)
    return JSONResponse({"ok": True})


@router.post("/reset")
async def reset_post(request: Request):
    chat_service: ChatService = request.app.state.chat
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise FinanceToolError("request body must be an object")
        session_id = str(payload.get("session_id", "")).strip()
        if session_id and not chat_service.reset(session_id, user_id=_get_user_id(request)):
            return JSONResponse({"error": "session not found"}, status_code=404)
        return JSONResponse({"ok": True})
    except (FinanceToolError, json.JSONDecodeError) as err:
        _trace_error(request, str(err))
        return JSONResponse(
            {"error": f"invalid reset request: {err}"}, status_code=400
        )


def _trace_error(request: Request, error: str) -> None:
    trace = request.app.state.trace
    trace.record(
        {
            "ok": False,
            "error": error,
            "path": request.url.path,
            "owner_id": getattr(request.state, "user_id", ""),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
