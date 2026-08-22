"""Memory management API endpoints.

Exposes user memories (preferences/policies), cross-session lessons,
and memory evolution triggers to the frontend.

Endpoints:
  GET    /memory/list            — list all memories with metadata
  POST   /memory                 — create or update a memory
  DELETE /memory/{key}           — delete a memory by key
  POST   /memory/evolve          — manually trigger memory evolution
  GET    /memory/lessons         — list all lessons
  DELETE /memory/lessons/{id}    — delete a lesson by id
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ...vault_client import VaultWriteError, sync_memory_profile

logger = logging.getLogger("matrix.server.memory")

router = APIRouter()


def _get_user_id(request: Request) -> str:
    """Extract user_id from request state (set by AuthMiddleware)."""
    return getattr(request.state, "user_id", "")


def _get_store(request: Request):
    """Get the SessionStore from app state."""
    return request.app.state.chat.store


def _get_evolution(request: Request):
    """Get the MemoryEvolution instance from ChatService."""
    return request.app.state.chat._evolution


def _get_lesson_store(request: Request):
    """Get the LessonStore instance from ChatService."""
    return request.app.state.chat._lesson_store


# ── User Memory CRUD ──────────────────────────────────────────────────────


@router.get("/memory/list")
async def list_memories(request: Request):
    """List all memories for the authenticated user.

    Returns:
        {"memories": [{key, value, memory_type, created_at, updated_at}, ...],
         "count": int, "max": 80}
    """
    store = _get_store(request)
    user_id = _get_user_id(request)
    memories = store.get_all_memories(user_id)
    count = store.count_memories(user_id)
    return {
        "memories": memories,
        "count": count,
        "max": 80,
    }


@router.post("/memory")
async def upsert_memory(request: Request):
    """Create or update a memory entry.

    Body: {"key": str, "value": str, "memory_type": "preference"|"policy"}
    """
    store = _get_store(request)
    user_id = _get_user_id(request)

    payload = await request.json()
    key = str(payload.get("key", "")).strip()
    value = str(payload.get("value", "")).strip()
    memory_type = str(payload.get("memory_type", "preference")).strip()

    if not key or not value:
        raise HTTPException(status_code=400, detail="key and value are required")
    if memory_type not in ("preference", "policy"):
        raise HTTPException(status_code=400, detail="memory_type must be 'preference' or 'policy'")

    profile = store.get_profile(user_id)
    profile[key] = value
    try:
        sync_memory_profile(user_id, profile)
    except VaultWriteError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    store.upsert_profile(user_id, key, value, memory_type=memory_type)

    logger.info("memory upsert: user=%s key=%s type=%s", user_id, key, memory_type)
    return {"ok": True, "key": key, "memory_type": memory_type}


@router.delete("/memory/{key:path}")
async def delete_memory(request: Request, key: str):
    """Delete a memory entry by key.

    For policy-type memories, requires ?confirm=true query param.
    """
    store = _get_store(request)
    user_id = _get_user_id(request)

    # Check if the memory is a policy — require confirmation
    memories = store.get_all_memories(user_id)
    mem = next((m for m in memories if m["key"] == key), None)
    if mem and mem.get("memory_type") == "policy":
        confirm = request.query_params.get("confirm", "").lower()
        if confirm != "true":
            raise HTTPException(
                status_code=409,
                detail="此记忆为 policy 类型（硬约束），删除可能导致 agent 行为变化。"
                       "请添加 ?confirm=true 确认删除。",
            )

    if mem is None:
        raise HTTPException(status_code=404, detail=f"memory key '{key}' not found")
    profile = store.get_profile(user_id)
    profile.pop(key, None)
    try:
        sync_memory_profile(user_id, profile)
    except VaultWriteError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    store.delete_profile_key(user_id, key)

    logger.info("memory delete: user=%s key=%s", user_id, key)
    return {"ok": True, "key": key}


# ── Memory Evolution ──────────────────────────────────────────────────────


@router.post("/memory/evolve")
async def trigger_evolution(request: Request):
    """Manually trigger memory evolution.

    Runs the four-stage pipeline:
    1. Importance scoring
    2. Conflict detection & resolution
    3. Consolidation (merge near-duplicates)
    4. Active forgetting (if over limit)

    Returns an EvolutionReport with before/after counts.
    """
    evolution = _get_evolution(request)
    user_id = _get_user_id(request)

    try:
        report = evolution.evolve(user_id)
        return {
            "ok": True,
            "report": {
                "total_before": report.total_before,
                "total_after": report.total_after,
                "conflicts_resolved": report.conflicts_resolved,
                "memories_consolidated": report.memories_consolidated,
                "memories_forgotten": report.memories_forgotten,
                "details": report.details,
            },
        }
    except Exception as exc:
        logger.error("memory evolve failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"evolution failed: {exc}")


# ── Lessons ───────────────────────────────────────────────────────────────


@router.get("/memory/lessons")
async def list_lessons(request: Request):
    """List all cross-session lessons for the user.

    Returns:
        {"lessons": [{...}, ...], "count": int, "max": 200}
    """
    lesson_store = _get_lesson_store(request)
    user_id = _get_user_id(request)
    lessons = lesson_store.get_all_lessons(user_id=user_id, limit=200)
    count = lesson_store.count(user_id=user_id)
    return {
        "lessons": [l.to_dict() for l in lessons],
        "count": count,
        "max": 200,
    }


@router.delete("/memory/lessons/{lesson_id}")
async def delete_lesson(request: Request, lesson_id: int):
    """Delete a lesson by ID."""
    lesson_store = _get_lesson_store(request)
    deleted = lesson_store.delete_lesson(lesson_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"lesson {lesson_id} not found")
    return {"ok": True, "lesson_id": lesson_id}
