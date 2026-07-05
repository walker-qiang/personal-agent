"""FastAPI application factory for the Agent HTTP server."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from ..chat import ChatService
from ..config import AgentConfig, load_config
from ..observability.trace import TraceLogger
from ..tools import ToolRegistry
from ..tools.finance import register_all as register_finance_tools
from .routes import chat, health, sessions, tools

# Pre-load the Web UI HTML content at module level
_INDEX_HTML = ""
_index_path = Path(__file__).parent / "static" / "index.html"
if _index_path.exists():
    _INDEX_HTML = _index_path.read_text(encoding="utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize application state on startup."""
    config: AgentConfig = app.state.config
    tools_registry = ToolRegistry()
    register_finance_tools(tools_registry, config.cache_path)
    trace = TraceLogger(config.trace_path)
    app.state.tools = tools_registry
    app.state.trace = trace
    app.state.chat = ChatService(config, tools_registry, trace)
    print(f"matrix agent listening on http://{config.host}:{config.port}")
    print(f"mode=read-only cache={config.cache_path} trace={config.trace_path}")
    yield


def create_app(config: AgentConfig | None = None) -> FastAPI:
    """Create the FastAPI application with all routes and middleware."""
    cfg = config or load_config()

    app = FastAPI(
        title="Project Matrix Agent",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.config = cfg

    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routes FIRST (before any catch-all routes)
    app.include_router(tools.router)
    app.include_router(chat.router)
    app.include_router(health.router)
    app.include_router(sessions.router)

    # Serve Web UI at root (LAST, so API routes take priority)
    @app.get("/", include_in_schema=False)
    async def serve_ui():
        if _INDEX_HTML:
            return HTMLResponse(_INDEX_HTML)
        return HTMLResponse("<h1>Project Matrix</h1><p>UI not found.</p>", status_code=404)

    return app