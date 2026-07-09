"""Provider selection API routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/api/provider")
async def get_providers(request: Request):
    """List available LLM, image, and video providers."""
    import json as _json

    chat = request.app.state.chat
    session_id = request.query_params.get("session_id", "").strip() or None

    return JSONResponse({
        "providers": chat.available_providers,
        "image_models": chat.available_image_models,
        "video_models": chat.available_video_models,
        "current": chat.get_provider(session_id),
    })


@router.post("/api/provider")
async def switch_provider(request: Request):
    """Switch the LLM provider/model for a specific session.

    Expects JSON body: {"session_id": "...", "provider": "deepseek", "model": "deepseek-v4-flash"}
    """
    import json as _json

    chat = request.app.state.chat
    try:
        body = await request.body()
        data = _json.loads(body) if body else {}
    except (ValueError, _json.JSONDecodeError):
        return JSONResponse({"ok": False, "error": "invalid json"}, status_code=400)

    provider = data.get("provider", "").strip().lower()
    model = data.get("model", "").strip()
    session_id = data.get("session_id", "").strip()
    if not session_id:
        return JSONResponse({"ok": False, "error": "session_id is required"}, status_code=400)
    if not provider:
        return JSONResponse({"ok": False, "error": "provider is required"}, status_code=400)

    result = chat.switch_provider(session_id, provider, model)
    if result["ok"]:
        return JSONResponse(result)
    return JSONResponse(result, status_code=400)