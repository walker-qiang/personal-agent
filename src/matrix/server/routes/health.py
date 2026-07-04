"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ...config import AgentConfig

router = APIRouter()


@router.get("/healthz")
async def healthz(request: Request) -> dict:
    config: AgentConfig = request.app.state.config
    return {
        "ok": True,
        "mode": "read-only",
        "cache_path": str(config.cache_path),
        "cache_exists": config.cache_path.exists(),
        "provider": config.agent_provider,
        "model": config.agent_model,
        "llm_available": config.llm_available,
        "llm_error": config.llm_unavailable_reason,
    }