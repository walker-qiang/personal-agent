"""Tools endpoints: list and direct call."""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...chat import result_count, timestamp
from ...tools import FinanceToolError, ToolRegistry

router = APIRouter()


@router.get("/tools")
async def list_tools(request: Request) -> dict:
    registry: ToolRegistry = request.app.state.tools
    return {"tools": registry.list_tools()}


@router.post("/tools/call")
async def tools_call(request: Request) -> JSONResponse:
    registry: ToolRegistry = request.app.state.tools
    trace = request.app.state.trace
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise FinanceToolError("request body must be an object")
        tool = str(payload.get("tool", "")).strip()
        arguments = payload.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise FinanceToolError("arguments must be an object")
        if not tool:
            raise FinanceToolError("tool is required")

        started = time.perf_counter()
        result = registry.call(tool, arguments)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        trace.record(
            {
                "ok": True,
                "tool": tool,
                "arguments": arguments,
                "elapsed_ms": elapsed_ms,
                "result_count": result_count(result),
                "ts": timestamp(),
            }
        )
        return JSONResponse({"tool": tool, "result": result})
    except FinanceToolError as err:
        _trace_error(request, str(err))
        return JSONResponse({"error": str(err)}, status_code=400)
    except json.JSONDecodeError as err:
        _trace_error(request, str(err))
        return JSONResponse({"error": f"invalid json: {err}"}, status_code=400)
    except Exception as err:
        _trace_error(request, str(err))
        return JSONResponse({"error": str(err)}, status_code=500)


def _trace_error(request: Request, error: str) -> None:
    trace = request.app.state.trace
    trace.record(
        {
            "ok": False,
            "error": error,
            "path": request.url.path,
            "ts": timestamp(),
        }
    )