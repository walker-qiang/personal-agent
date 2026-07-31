"""Session management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


def _get_user_id(request: Request) -> str:
    """Extract user_id from request state (set by AuthMiddleware)."""
    return getattr(request.state, "user_id", "")


@router.get("/sessions")
async def list_sessions(request: Request, include_hidden: bool = False):
    """List recent chat sessions for the authenticated user.

    Query params:
        include_hidden: if true, include archived (hidden) sessions
    """
    from ...store import SessionStore

    store: SessionStore = request.app.state.chat.store
    user_id = _get_user_id(request)
    sessions = store.list_sessions(user_id=user_id, limit=20, include_hidden=include_hidden)
    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
async def get_session(request: Request, session_id: str):
    """Get session metadata."""
    from ...store import SessionStore

    store: SessionStore = request.app.state.chat.store
    session = store.get_session(session_id, user_id=_get_user_id(request))
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return {"session": session}


@router.delete("/sessions/{session_id}")
async def delete_session(request: Request, session_id: str):
    """Delete a session and all its messages."""
    from ...store import SessionStore

    store: SessionStore = request.app.state.chat.store
    deleted = store.delete_session(session_id, user_id=_get_user_id(request))
    if not deleted:
        raise HTTPException(status_code=404, detail="session not found")
    return {"ok": True}


@router.post("/sessions/batch-delete")
async def batch_delete_sessions(request: Request):
    """Delete multiple sessions and their messages."""
    from ...store import SessionStore

    store: SessionStore = request.app.state.chat.store
    payload = await request.json()
    session_ids = payload.get("session_ids", [])
    if not isinstance(session_ids, list) or not session_ids:
        raise HTTPException(status_code=400, detail="session_ids must be a non-empty list")
    count = store.batch_delete_sessions(session_ids, user_id=_get_user_id(request))
    return {"ok": True, "deleted": count}


@router.post("/sessions/batch-archive")
async def batch_archive_sessions(request: Request):
    """Archive (hide) multiple sessions."""
    from ...store import SessionStore

    store: SessionStore = request.app.state.chat.store
    payload = await request.json()
    session_ids = payload.get("session_ids", [])
    if not isinstance(session_ids, list) or not session_ids:
        raise HTTPException(status_code=400, detail="session_ids must be a non-empty list")
    count = store.batch_set_hidden(session_ids, hidden=True, user_id=_get_user_id(request))
    return {"ok": True, "archived": count}


@router.post("/sessions/batch-unarchive")
async def batch_unarchive_sessions(request: Request):
    """Unarchive (unhide) multiple sessions."""
    from ...store import SessionStore

    store: SessionStore = request.app.state.chat.store
    payload = await request.json()
    session_ids = payload.get("session_ids", [])
    if not isinstance(session_ids, list) or not session_ids:
        raise HTTPException(status_code=400, detail="session_ids must be a non-empty list")
    count = store.batch_set_hidden(session_ids, hidden=False, user_id=_get_user_id(request))
    return {"ok": True, "unarchived": count}


@router.get("/sessions/{session_id}/messages")
async def get_messages(request: Request, session_id: str):
    """Get all messages for a session."""
    from ...store import SessionStore

    store: SessionStore = request.app.state.chat.store
    messages = store.get_history(
        session_id, max_turns=999, user_id=_get_user_id(request),
    )
    if not messages and store.get_session(session_id, user_id=_get_user_id(request)) is None:
        raise HTTPException(status_code=404, detail="session not found")
    return {"messages": messages}


@router.post("/sessions/{session_id}/branch")
async def branch_session(request: Request, session_id: str):
    """Branch a session from a specific message.

    Moves the session's leaf_id to the specified message, creating a
    fork point. New messages will be appended as children of that message.
    Existing messages on the old branch are preserved.
    """
    from ...store import SessionStore

    store: SessionStore = request.app.state.chat.store
    payload = await request.json()
    from_message_id = str(payload.get("from_message_id", "")).strip()
    if not from_message_id:
        raise HTTPException(status_code=400, detail="from_message_id is required")
    ok = store.branch(session_id, from_message_id, user_id=_get_user_id(request))
    if not ok:
        raise HTTPException(
            status_code=404,
            detail=f"Message {from_message_id} not found in session {session_id}",
        )
    return {"session_id": session_id, "leaf_id": from_message_id, "branched": True}


@router.get("/sessions/{session_id}/branches")
async def get_branches(request: Request, session_id: str):
    """List all fork points in a session tree."""
    from ...store import SessionStore

    store: SessionStore = request.app.state.chat.store
    branches = store.get_branches(session_id, user_id=_get_user_id(request))
    if not branches and store.get_session(session_id, user_id=_get_user_id(request)) is None:
        raise HTTPException(status_code=404, detail="session not found")
    return {"session_id": session_id, "branches": branches}


@router.get("/sessions/{session_id}/leaf")
async def get_leaf(request: Request, session_id: str):
    """Get the current leaf_id for a session."""
    from ...store import SessionStore

    store: SessionStore = request.app.state.chat.store
    leaf_id = store.get_leaf_id(session_id, user_id=_get_user_id(request))
    if leaf_id is None and store.get_session(session_id, user_id=_get_user_id(request)) is None:
        raise HTTPException(status_code=404, detail="session not found")
    return {"session_id": session_id, "leaf_id": leaf_id}


# ---- Skills CRUD (adapts to multi-agent skills structure) ----

def _get_skills_domain_dir(request: Request, domain: str = "investment") -> Path:
    """Get the skills directory for a domain (e.g. skills_base_dir / investment)."""
    from pathlib import Path
    from ...chat import ChatService
    chat: ChatService = request.app.state.chat
    return chat.config.skills_base_dir / domain


def _list_all_skills(request: Request):
    """List all skills using the agent registry."""
    from ...chat import ChatService
    from ...skills import render_workflow
    from ...agent import AgentRegistry

    chat: ChatService = request.app.state.chat
    registry: AgentRegistry = chat.agent_registry
    registry.reload_skills()

    all_skills = []
    seen = set()
    for skill in registry.list_all_skills():
        if skill.name not in seen:
            seen.add(skill.name)
            # Find which agents use this skill
            agents = []
            for agent_def in registry.list_all():
                if skill.name in agent_def.all_skills:
                    agents.append({"id": agent_def.id, "name": agent_def.name})
            all_skills.append({
                "name": skill.name,
                "title": skill.title,
                "description": skill.description,
                "agents": agents,
                "workflow": skill.workflow,
                "workflow_text": render_workflow(skill.workflow),
                "output_format": skill.output_format,
                "knowledge_files": skill.knowledge_files,
                "script_files": skill.script_files,
            })
    return all_skills


@router.get("/skills")
async def list_skills(request: Request):
    """List all available skills across all domains."""
    return {"skills": _list_all_skills(request)}


@router.post("/skills")
async def create_skill(request: Request):
    """Create a new skill directory with SKILL.md. Defaults to investment domain."""
    from ...chat import ChatService
    from ...skills import SkillDefinition, create_skill_dir

    chat: ChatService = request.app.state.chat
    payload = await request.json()
    name = str(payload.get("name", "")).strip()
    title = str(payload.get("title", "")).strip()
    description = str(payload.get("description", "")).strip()
    domain = str(payload.get("domain", "investment")).strip() or "investment"

    if not name or not title:
        raise HTTPException(status_code=400, detail="name and title are required")

    safe_name = "".join(c for c in name if c.isalnum() or c in "_-").lower()
    if not safe_name:
        raise HTTPException(status_code=400, detail="invalid name")

    skills_dir = _get_skills_domain_dir(request, domain)
    skill_dir = skills_dir / safe_name
    if skill_dir.exists():
        raise HTTPException(status_code=409, detail="skill already exists")

    skill = SkillDefinition(
        name=safe_name,
        title=title,
        description=description,
        output_format=str(payload.get("output_format", "")).strip(),
    )
    create_skill_dir(skills_dir, skill)
    chat.agent_registry.reload_skills()
    return {"ok": True, "name": safe_name, "domain": domain}


@router.put("/skills/{skill_name}")
async def update_skill(request: Request, skill_name: str):
    """Update a skill's SKILL.md. Defaults to investment domain."""
    from ...chat import ChatService
    from ...skills import SkillDefinition, update_skill_dir

    chat: ChatService = request.app.state.chat
    payload = await request.json()
    title = str(payload.get("title", "")).strip()
    description = str(payload.get("description", "")).strip()
    domain = str(payload.get("domain", "investment")).strip() or "investment"

    skills_dir = _get_skills_domain_dir(request, domain)
    if not (skills_dir / skill_name).is_dir():
        raise HTTPException(status_code=404, detail="skill not found")

    skill = SkillDefinition(
        name=skill_name,
        title=title or skill_name,
        description=description,
        output_format=str(payload.get("output_format", "")).strip(),
    )
    update_skill_dir(skills_dir, skill_name, skill)
    chat.agent_registry.reload_skills()
    return {"ok": True}


@router.delete("/skills/{skill_name}")
async def delete_skill(request: Request, skill_name: str):
    """Delete a skill directory entirely. Defaults to investment domain."""
    from ...chat import ChatService
    from ...skills import delete_skill_dir

    chat: ChatService = request.app.state.chat
    domain = str(request.query_params.get("domain", "investment")).strip() or "investment"
    skills_dir = _get_skills_domain_dir(request, domain)
    if not (skills_dir / skill_name).is_dir():
        raise HTTPException(status_code=404, detail="skill not found")
    delete_skill_dir(skills_dir, skill_name)
    chat.agent_registry.reload_skills()
    return {"ok": True}


# ---- Knowledge & Script file management ----

def _find_skill_dir(request: Request, skill_name: str) -> Path | None:
    """Find the skill directory across all domains."""
    from ...chat import ChatService
    chat: ChatService = request.app.state.chat
    base = chat.config.skills_base_dir
    for domain_dir in base.iterdir():
        if not domain_dir.is_dir():
            continue
        skill_dir = domain_dir / skill_name
        if skill_dir.is_dir():
            return skill_dir
    return None


@router.get("/skills/{skill_name}/knowledge")
async def list_knowledge(request: Request, skill_name: str):
    """List knowledge files for a skill."""
    from ...skills import SkillDefinition

    skill_dir = _find_skill_dir(request, skill_name)
    if not skill_dir:
        raise HTTPException(status_code=404, detail="skill not found")
    knowledge = SkillDefinition.read_knowledge_static(skill_dir)
    return {"knowledge": knowledge}


@router.put("/skills/{skill_name}/knowledge/{filename:path}")
async def write_knowledge(request: Request, skill_name: str, filename: str):
    """Write a knowledge file."""
    from ...skills import write_knowledge

    chat = request.app.state.chat
    skill_dir = _find_skill_dir(request, skill_name)
    if not skill_dir:
        raise HTTPException(status_code=404, detail="skill not found")

    payload = await request.json()
    content = str(payload.get("content", ""))
    write_knowledge(skill_dir.parent, skill_name, filename, content)
    chat.agent_registry.reload_skills()
    return {"ok": True}


@router.get("/skills/{skill_name}/knowledge/{filename:path}")
async def get_knowledge_file(request: Request, skill_name: str, filename: str):
    """Read a single knowledge file."""
    from ...skills import SkillDefinition

    skill_dir = _find_skill_dir(request, skill_name)
    if not skill_dir:
        raise HTTPException(status_code=404, detail="skill not found")
    content = SkillDefinition.read_knowledge_file_static(skill_dir, filename)
    return {"filename": filename, "content": content}


@router.delete("/skills/{skill_name}/knowledge/{filename:path}")
async def remove_knowledge(request: Request, skill_name: str, filename: str):
    """Delete a knowledge file."""
    from ...skills import delete_knowledge

    chat = request.app.state.chat
    skill_dir = _find_skill_dir(request, skill_name)
    if not skill_dir:
        raise HTTPException(status_code=404, detail="skill not found")
    delete_knowledge(skill_dir.parent, skill_name, filename)
    chat.agent_registry.reload_skills()
    return {"ok": True}


@router.get("/skills/{skill_name}/scripts/{filename:path}")
async def get_script_file(request: Request, skill_name: str, filename: str):
    """Read a single script file."""
    from ...skills import SkillDefinition

    skill_dir = _find_skill_dir(request, skill_name)
    if not skill_dir:
        raise HTTPException(status_code=404, detail="skill not found")
    content = SkillDefinition.read_script_static(skill_dir, filename)
    return {"filename": filename, "content": content}


@router.put("/skills/{skill_name}/scripts/{filename:path}")
async def write_script(request: Request, skill_name: str, filename: str):
    """Write a script file."""
    from ...skills import write_script

    chat = request.app.state.chat
    skill_dir = _find_skill_dir(request, skill_name)
    if not skill_dir:
        raise HTTPException(status_code=404, detail="skill not found")

    payload = await request.json()
    content = str(payload.get("content", ""))
    write_script(skill_dir.parent, skill_name, filename, content)
    chat.agent_registry.reload_skills()
    return {"ok": True}


@router.delete("/skills/{skill_name}/scripts/{filename:path}")
async def remove_script(request: Request, skill_name: str, filename: str):
    """Delete a script file."""
    from ...skills import delete_script

    chat = request.app.state.chat
    skill_dir = _find_skill_dir(request, skill_name)
    if not skill_dir:
        raise HTTPException(status_code=404, detail="skill not found")
    delete_script(skill_dir.parent, skill_name, filename)
    chat.agent_registry.reload_skills()
    return {"ok": True}
