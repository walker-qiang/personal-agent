"""Session management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/sessions")
async def list_sessions(request: Request):
    """List recent chat sessions."""
    from ...store import SessionStore

    store: SessionStore = request.app.state.chat.store
    sessions = store.list_sessions(limit=20)
    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
async def get_session(request: Request, session_id: str):
    """Get session metadata."""
    from ...store import SessionStore

    store: SessionStore = request.app.state.chat.store
    session = store.get_session(session_id)
    if session is None:
        return {"error": "session not found"}, 404
    return {"session": session}


@router.delete("/sessions/{session_id}")
async def delete_session(request: Request, session_id: str):
    """Delete a session and all its messages."""
    from ...store import SessionStore

    store: SessionStore = request.app.state.chat.store
    deleted = store.delete_session(session_id)
    if not deleted:
        return {"error": "session not found"}, 404
    return {"ok": True}


@router.get("/sessions/{session_id}/messages")
async def get_messages(request: Request, session_id: str):
    """Get all messages for a session."""
    from ...store import SessionStore

    store: SessionStore = request.app.state.chat.store
    messages = store.get_history(session_id, max_turns=999)
    return {"messages": messages}


@router.get("/skills")
async def list_skills(request: Request):
    """List available skills."""
    from ...chat import ChatService

    chat: ChatService = request.app.state.chat
    skills = [
        {
            "name": s.name,
            "title": s.title,
            "description": s.description,
            "trigger_keywords": s.trigger_keywords[:3],
        }
        for s in chat.skills
    ]
    return {"skills": skills}