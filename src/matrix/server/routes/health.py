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
        "codex_bin": config.codex_bin,
        "codex_available": config.llm_available if config.agent_provider == "codex" else False,
        "llm_available": config.llm_available,
        "llm_error": config.llm_unavailable_reason,
        "runtime_mode": config.runtime_mode,
        "rag_available": getattr(request.app.state, "retriever", None) is not None,
        "rag_status": getattr(request.app.state, "rag_status", "disabled"),
        "rag_error": getattr(request.app.state, "rag_error", ""),
        "mcp_available": getattr(request.app.state, "mcp_client", None) is not None,
        "mcp_error": getattr(request.app.state, "mcp_error", ""),
    }
