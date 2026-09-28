"""Memory management API endpoints.

Exposes user memories (preferences/policies), cross-session lessons,
and memory evolution triggers for API clients.

Endpoints:
  GET    /memory/list            — list all memories with provenance metadata
  POST   /memory                 — create or update a memory
  DELETE /memory/{key}           — retire (or hard-delete) a memory by key
  POST   /memory/evolve          — manually trigger memory evolution
  POST   /memory/recall          — cross-session conversation recall
  GET    /memory/stats           — store + retriever + worker health
  GET    /memory/lessons         — list all lessons (failure + success)
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
async def list_memories(request: Request, include_retired: bool = False):
    """List all memories for the authenticated user.

    Returns full provenance so a user can audit and trace any single entry:
    source session, source quote, confidence, validity window, access count.

    Query:
        include_retired — also return entries with a ``valid_to`` tombstone.
    """
    store = _get_store(request)
    user_id = _get_user_id(request)
    memories = store.get_all_memories(user_id, include_retired=include_retired)
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
    scope = str(payload.get("scope", "user")).strip()
    scope_id = str(payload.get("scope_id", "")).strip()

    if not key or not value:
        raise HTTPException(status_code=400, detail="key and value are required")
    if memory_type not in ("preference", "policy"):
        raise HTTPException(status_code=400, detail="memory_type must be 'preference' or 'policy'")
    if scope not in ("user", "session"):
        raise HTTPException(status_code=400, detail="scope must be 'user' or 'session'")
    if scope == "session" and not scope_id:
        raise HTTPException(status_code=400, detail="scope_id is required when scope='session'")

    # Screen before persisting: this text will be replayed into future prompts.
    chat = request.app.state.chat
    guard = getattr(chat, "_memory_safety", None)
    if guard is not None:
        screening = guard.screen(key, value)
        if not screening.allowed:
            raise HTTPException(
                status_code=422, detail=f"记忆内容未通过安全校验：{screening.reason}",
            )
        value = screening.value

    store.upsert_profile(
        user_id, key, value, memory_type=memory_type,
        scope=scope, scope_id=scope_id,
    )

    if scope == "user":
        # Only durable user facts are mirrored to the vault.
        profile = store.get_profile_for_vault(user_id)
        profile[key] = {"value": value, "memory_type": memory_type}
        try:
            sync_memory_profile(user_id, profile)
        except VaultWriteError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    retriever = getattr(chat, "_memory_retriever", None)
    if retriever is not None and scope == "user":
        try:
            retriever.index_keys(user_id, [(key, value)])
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory upsert: index failed: %s", exc)

    logger.info(
        "memory upsert: user=%s key=%s type=%s scope=%s",
        user_id, key, memory_type, scope,
    )
    return {"ok": True, "key": key, "memory_type": memory_type, "scope": scope}


@router.delete("/memory/{key:path}")
async def delete_memory(request: Request, key: str):
    """Retire (default) or hard-delete a memory entry by key.

    For policy-type memories, requires ?confirm=true query param.
    Add ?hard=true to physically remove the row instead of setting a
    ``valid_to`` tombstone — hard deletion loses temporal history.
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
    profile = store.get_profile_for_vault(user_id)
    profile.pop(key, None)
    try:
        sync_memory_profile(user_id, profile)
    except VaultWriteError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    hard = request.query_params.get("hard", "").lower() == "true"
    store.delete_profile_key(user_id, key, hard=hard)

    # Keep the retrieval index in step with the store.
    chat = request.app.state.chat
    retriever = getattr(chat, "_memory_retriever", None)
    if retriever is not None:
        try:
            retriever.drop_keys(user_id, [key])
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory delete: index drop failed: %s", exc)

    logger.info("memory delete: user=%s key=%s hard=%s", user_id, key, hard)
    return {"ok": True, "key": key, "hard": hard}


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


# ── Episodic recall & health ──────────────────────────────────────────────


@router.post("/memory/recall")
async def recall_conversations(request: Request):
    """Search past conversations across sessions (episodic memory).

    Body: {"query": str, "limit": int = 10, "exclude_session_id": str = ""}

    Returns matching turns with session id, title and a snippet, newest first.
    """
    store = _get_store(request)
    user_id = _get_user_id(request)
    payload = await request.json()
    query = str(payload.get("query", "")).strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    limit = int(payload.get("limit", 10))
    limit = max(1, min(limit, 50))
    exclude = str(payload.get("exclude_session_id", "")).strip()

    hits = store.search_conversations(
        user_id, query, limit=limit, exclude_session_id=exclude,
    )
    return {"ok": True, "query": query, "count": len(hits), "results": hits}


@router.get("/memory/stats")
async def memory_stats(request: Request):
    """Memory subsystem health: counts, retriever mode, write worker state."""
    chat = request.app.state.chat
    store = _get_store(request)
    user_id = _get_user_id(request)
    retriever = getattr(chat, "_memory_retriever", None)
    worker = getattr(chat, "_memory_writer", None)
    lesson_store = _get_lesson_store(request)
    return {
        "ok": True,
        "user_id": user_id,
        "active_memories": store.count_memories(user_id),
        "retired_memories": store.count_memories(user_id, include_retired=True)
        - store.count_memories(user_id),
        "retriever": {
            "available": retriever is not None,
            "semantic": bool(retriever and retriever.semantic_enabled),
        },
        "writer": worker.snapshot() if worker is not None else None,
        "lessons": lesson_store.count_by_kind(user_id=user_id),
        "max_memories": 80,
    }


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
    deleted = lesson_store.delete_lesson(
        lesson_id, user_id=_get_user_id(request),
    )
    if not deleted:
        raise HTTPException(status_code=404, detail=f"lesson {lesson_id} not found")
    return {"ok": True, "lesson_id": lesson_id}
