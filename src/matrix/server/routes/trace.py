"""Trace API routes — queryable access to the structured trace store."""

from __future__ import annotations

from fastapi import APIRouter, Request, Query

router = APIRouter(prefix="/api/trace", tags=["trace"])


@router.get("/sessions")
async def list_sessions(request: Request, limit: int = Query(50, ge=1, le=200)):
    """List recent trace sessions with summary stats."""
    trace = request.app.state.trace
    return trace.sessions(limit=limit)


@router.get("/sessions/{session_id}")
async def session_detail(request: Request, session_id: str):
    """Get all trace events for a session."""
    trace = request.app.state.trace
    events = trace.session_detail(session_id)
    return {"session_id": session_id, "events": events}


@router.get("/events")
async def query_events(
    request: Request,
    session_id: str | None = Query(None),
    event_type: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Query trace events with optional filters."""
    trace = request.app.state.trace
    return trace.query(
        session_id=session_id,
        event_type=event_type,
        limit=limit,
        offset=offset,
    )


@router.get("/stats")
async def trace_stats(request: Request):
    """Get overall trace statistics."""
    trace = request.app.state.trace
    return trace.stats()