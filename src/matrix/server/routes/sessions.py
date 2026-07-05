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
    from ...skills import render_workflow

    chat: ChatService = request.app.state.chat
    skills = [
        {
            "name": s.name,
            "title": s.title,
            "description": s.description,
            "trigger_keywords": s.trigger_keywords,
            "workflow": s.workflow,
            "workflow_text": render_workflow(s.workflow),
            "output_format": s.output_format,
        }
        for s in chat.skills
    ]
    return {"skills": skills}


@router.post("/skills")
async def create_skill(request: Request):
    """Create a new skill (writes to Markdown file with YAML frontmatter)."""
    from ...chat import ChatService
    from ...skills import SkillDefinition, render_skill

    chat: ChatService = request.app.state.chat
    payload = await request.json()
    name = str(payload.get("name", "")).strip()
    title = str(payload.get("title", "")).strip()
    description = str(payload.get("description", "")).strip()
    keywords = payload.get("trigger_keywords", [])
    if not isinstance(keywords, list):
        keywords = []

    if not name or not title:
        return {"error": "name and title are required"}, 400

    safe_name = "".join(c for c in name if c.isalnum() or c in "_-").lower()
    if not safe_name:
        return {"error": "invalid name"}, 400

    md_path = chat.skills_dir / f"{safe_name}.md"
    if md_path.exists():
        return {"error": "skill already exists"}, 409

    skill = SkillDefinition(
        name=safe_name,
        title=title,
        description=description,
        trigger_keywords=keywords,
        output_format=str(payload.get("output_format", "")).strip(),
    )
    md_path.write_text(render_skill(skill), encoding="utf-8")
    chat.reload_skills()
    return {"ok": True, "name": safe_name}


@router.put("/skills/{skill_name}")
async def update_skill(request: Request, skill_name: str):
    """Update a skill's Markdown file with YAML frontmatter."""
    from ...chat import ChatService
    from ...skills import SkillDefinition, render_skill

    chat: ChatService = request.app.state.chat
    payload = await request.json()
    title = str(payload.get("title", "")).strip()
    description = str(payload.get("description", "")).strip()
    keywords = payload.get("trigger_keywords", [])
    if not isinstance(keywords, list):
        keywords = []

    md_path = chat.skills_dir / f"{skill_name}.md"
    if not md_path.exists():
        return {"error": "skill not found"}, 404

    skill = SkillDefinition(
        name=skill_name,
        title=title or skill_name,
        description=description,
        trigger_keywords=keywords,
        output_format=str(payload.get("output_format", "")).strip(),
    )
    md_path.write_text(render_skill(skill), encoding="utf-8")
    chat.reload_skills()
    return {"ok": True}


@router.delete("/skills/{skill_name}")
async def delete_skill(request: Request, skill_name: str):
    """Delete a skill's Markdown file."""
    from ...chat import ChatService

    chat: ChatService = request.app.state.chat
    md_path = chat.skills_dir / f"{skill_name}.md"
    if not md_path.exists():
        return {"error": "skill not found"}, 404
    md_path.unlink()
    chat.reload_skills()
    return {"ok": True}