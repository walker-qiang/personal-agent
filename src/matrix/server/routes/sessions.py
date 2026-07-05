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
            "workflow": s.workflow,
            "workflow_text": render_workflow(s.workflow),
            "output_format": s.output_format,
            "knowledge_files": s.knowledge_files,
            "script_files": s.script_files,
        }
        for s in chat.skills
    ]
    return {"skills": skills}


@router.post("/skills")
async def create_skill(request: Request):
    """Create a new skill directory with SKILL.md."""
    from ...chat import ChatService
    from ...skills import SkillDefinition, create_skill_dir

    chat: ChatService = request.app.state.chat
    payload = await request.json()
    name = str(payload.get("name", "")).strip()
    title = str(payload.get("title", "")).strip()
    description = str(payload.get("description", "")).strip()

    if not name or not title:
        return {"error": "name and title are required"}, 400

    safe_name = "".join(c for c in name if c.isalnum() or c in "_-").lower()
    if not safe_name:
        return {"error": "invalid name"}, 400

    skill_dir = chat.skills_dir / safe_name
    if skill_dir.exists():
        return {"error": "skill already exists"}, 409

    skill = SkillDefinition(
        name=safe_name,
        title=title,
        description=description,
        output_format=str(payload.get("output_format", "")).strip(),
    )
    create_skill_dir(chat.skills_dir, skill)
    chat.reload_skills()
    return {"ok": True, "name": safe_name}


@router.put("/skills/{skill_name}")
async def update_skill(request: Request, skill_name: str):
    """Update a skill's SKILL.md."""
    from ...chat import ChatService
    from ...skills import SkillDefinition, update_skill_dir

    chat: ChatService = request.app.state.chat
    payload = await request.json()
    title = str(payload.get("title", "")).strip()
    description = str(payload.get("description", "")).strip()

    if not (chat.skills_dir / skill_name).is_dir():
        return {"error": "skill not found"}, 404

    skill = SkillDefinition(
        name=skill_name,
        title=title or skill_name,
        description=description,
        output_format=str(payload.get("output_format", "")).strip(),
    )
    update_skill_dir(chat.skills_dir, skill_name, skill)
    chat.reload_skills()
    return {"ok": True}


@router.delete("/skills/{skill_name}")
async def delete_skill(request: Request, skill_name: str):
    """Delete a skill directory entirely."""
    from ...chat import ChatService
    from ...skills import delete_skill_dir

    chat: ChatService = request.app.state.chat
    if not (chat.skills_dir / skill_name).is_dir():
        return {"error": "skill not found"}, 404
    delete_skill_dir(chat.skills_dir, skill_name)
    chat.reload_skills()
    return {"ok": True}


# ---- Knowledge & Script file management ----

@router.get("/skills/{skill_name}/knowledge")
async def list_knowledge(request: Request, skill_name: str):
    """List knowledge files for a skill."""
    from ...chat import ChatService

    chat: ChatService = request.app.state.chat
    skill = next((s for s in chat.skills if s.name == skill_name), None)
    if not skill:
        return {"error": "skill not found"}, 404
    knowledge = skill.read_knowledge(chat.skills_dir / skill_name)
    return {"knowledge": knowledge}


@router.put("/skills/{skill_name}/knowledge/{filename:path}")
async def write_knowledge(request: Request, skill_name: str, filename: str):
    """Write a knowledge file."""
    from ...chat import ChatService
    from ...skills import write_knowledge

    chat: ChatService = request.app.state.chat
    payload = await request.json()
    content = str(payload.get("content", ""))
    write_knowledge(chat.skills_dir, skill_name, filename, content)
    chat.reload_skills()
    return {"ok": True}


@router.delete("/skills/{skill_name}/knowledge/{filename:path}")
async def remove_knowledge(request: Request, skill_name: str, filename: str):
    """Delete a knowledge file."""
    from ...chat import ChatService
    from ...skills import delete_knowledge

    chat: ChatService = request.app.state.chat
    delete_knowledge(chat.skills_dir, skill_name, filename)
    chat.reload_skills()
    return {"ok": True}


@router.put("/skills/{skill_name}/scripts/{filename:path}")
async def write_script(request: Request, skill_name: str, filename: str):
    """Write a script file."""
    from ...chat import ChatService
    from ...skills import write_script

    chat: ChatService = request.app.state.chat
    payload = await request.json()
    content = str(payload.get("content", ""))
    write_script(chat.skills_dir, skill_name, filename, content)
    chat.reload_skills()
    return {"ok": True}


@router.delete("/skills/{skill_name}/scripts/{filename:path}")
async def remove_script(request: Request, skill_name: str, filename: str):
    """Delete a script file."""
    from ...chat import ChatService
    from ...skills import delete_script

    chat: ChatService = request.app.state.chat
    delete_script(chat.skills_dir, skill_name, filename)
    chat.reload_skills()
    return {"ok": True}