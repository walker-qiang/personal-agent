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


@router.get("/spans")
async def query_spans(
    request: Request,
    trace_id: str | None = Query(None),
    session_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    """Query OTel-standardized spans by trace_id or session_id."""
    trace = request.app.state.trace
    if not hasattr(trace, "query_spans"):
        return {"spans": []}
    spans = trace.query_spans(
        trace_id=trace_id,
        session_id=session_id,
        limit=limit,
    )
    return {"spans": spans}


@router.get("/otlp/export")
async def otlp_export_buffer(request: Request):
    """Get buffered OTLP exports (for debugging/testing).

    This endpoint returns the raw OTLP/JSON payloads that would be sent
    to an external OTLP receiver (Jaeger, Tempo, etc.). Only available
    when OTLP export is enabled in configuration.
    """
    trace = request.app.state.trace
    if not hasattr(trace, "export_otlp"):
        return {"exports": [], "enabled": False}
    return {
        "exports": trace.export_otlp(),
        "enabled": getattr(trace, "_otlp_export", False),
    }