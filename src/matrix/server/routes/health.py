"""Health check endpoint."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from ...config import AgentConfig

router = APIRouter()


@router.get("/healthz")
async def healthz(request: Request) -> dict:
    config: AgentConfig = request.app.state.config
    response = {
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
    if request.query_params.get("probe") == "true":
        chat = request.app.state.chat
        user_id = getattr(request.state, "user_id", "default")
        probe = await asyncio.to_thread(
            chat.probe_llm,
            request.query_params.get("session_id") or None,
            user_id,
        )
        response.update(
            {
                "probe_ok": probe.get("ok") is True,
                "probe_error": probe.get("error", ""),
                "probe_provider": probe.get("provider", ""),
                "probe_model": probe.get("model", ""),
                "probe_latency_ms": probe.get("latency_ms"),
            }
        )
    return response


@router.post("/rag/warmup")
async def warmup_rag(request: Request) -> dict:
    """Start RAG initialization without waiting for the model and index to load."""
    start_warmup = getattr(request.app.state, "start_rag_warmup", None)
    if start_warmup is None:
        return {"ok": False, "rag_status": "disabled"}
    status = start_warmup()
    return {"ok": status != "disabled", "rag_status": status}
