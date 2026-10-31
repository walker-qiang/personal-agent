"""Terminal responses for retired research execution addresses; no model client."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()
REPLACEMENT = "/api/investment/research/core/requests"


@router.get("/api/research/stock.v3/capabilities")
@router.get("/api/research/investment.v3/capabilities")
async def capabilities() -> dict[str, object]:
    return {
        "supported": False,
        "retired": True,
        "replacement_owner": "personal-os-api",
        "replacement": REPLACEMENT,
        "reason": "Legacy model planning, repair and synthesis have been removed.",
    }


@router.post("/api/research/stock.v3")
@router.post("/api/research/investment.v3")
@router.post("/api/research/investment.v3/metric-plan")
async def retired_research() -> JSONResponse:
    return JSONResponse(
        status_code=410,
        content={
            "error": "legacy_execution_retired",
            "detail": "旧研究执行已退出。历史响应和用量保持保全，不重试、不创建模型调用。",
            "replacement_owner": "personal-os-api",
            "replacement": REPLACEMENT,
        },
    )
