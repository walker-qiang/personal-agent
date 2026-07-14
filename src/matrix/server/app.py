"""FastAPI application factory for the Agent HTTP server."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from ..chat import ChatService
from ..config import AgentConfig, load_config
from ..logging_config import RequestIdFilter, get_logger, setup_logging
from ..observability.trace import TraceLogger
from ..tools import ToolRegistry
from ..tools.finance import register_all as register_finance_tools
from ..tools.web import register_all as register_web_tools
from ..tools.agnes import register_all as register_agnes_tools
from .routes import auth, chat, health, provider, sessions, tools, upload
from .middleware import AuthMiddleware

logger = get_logger("matrix")

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
    register_web_tools(tools_registry)
    register_agnes_tools(tools_registry)
    trace = TraceLogger(config.trace_path)
    app.state.tools = tools_registry
    app.state.trace = trace
    app.state.chat = ChatService(config, tools_registry, trace)
    # Bootstrap admin user on first run (no users in DB yet)
    if config.admin_password_hash:
        if app.state.chat.store.user_count() == 0:
            created = app.state.chat.store.create_user("admin", config.admin_password_hash)
            if created:
                logger.info("Created admin user (first-run bootstrap)")
    else:
        if app.state.chat.store.user_count() == 0:
            logger.warning(
                "No users exist and ADMIN_PASSWORD is not set. "
                "Create a user via the API or set ADMIN_PASSWORD in .env."
            )
    logger.info("matrix agent listening on http://%s:%s", config.host, config.port)
    logger.info("mode=read-only cache=%s trace=%s", config.cache_path, config.trace_path)
    # Sync user profiles from personal-assets on startup
    sync_path = config.memory_sync_path
    if sync_path and Path(sync_path).is_dir():
        synced = 0
        for json_file in Path(sync_path).glob("*.json"):
            uid = json_file.stem
            count = app.state.chat.store.sync_profile_from_file(uid, str(json_file))
            if count > 0:
                logger.info("memory_sync: user=%s entries=%d", uid, count)
                synced += 1
        if synced:
            logger.info("memory_sync: %d user(s) synced from %s", synced, sync_path)

    # Initialize RAG retriever if docs path is configured
    app.state.retriever = None
    if config.rag_docs_path and Path(config.rag_docs_path).is_dir():
        try:
            from ..rag.embedder import LocalEmbedder
            from ..rag.retriever import HybridRetriever
            from ..rag.indexer import DocumentIndexer

            embedder = LocalEmbedder(model_name=config.rag_embed_model)
            indexer = DocumentIndexer(
                embedder=embedder,
                persist_dir=config.rag_persist_dir,
            )
            chunk_count = indexer.index_directory(config.rag_docs_path)
            logger.info(
                "rag: indexed %d chunks from %s (persist=%s)",
                chunk_count, config.rag_docs_path, config.rag_persist_dir,
            )
            app.state.retriever = HybridRetriever(
                embedder=embedder,
                persist_dir=config.rag_persist_dir,
            )
            app.state.chat.retriever = app.state.retriever
            logger.info("rag: retriever ready")
        except Exception as exc:
            logger.warning("rag: initialization failed (will run without RAG): %s", exc)

    yield


def create_app(config: AgentConfig | None = None) -> FastAPI:
    """Create the FastAPI application with all routes and middleware."""
    cfg = config or load_config()

    # Initialize structured logging
    setup_logging(level=cfg.log_level, log_dir=cfg.log_dir)

    app = FastAPI(
        title="Matrix",
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

    # Request ID middleware — injects a unique ID per request into logs
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        RequestIdFilter.set_request_id(rid)
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000)
        logger.info(
            "request=%s %s status=%d duration=%dms",
            request.method, request.url.path, response.status_code, duration_ms,
        )
        response.headers["X-Request-ID"] = rid
        response.headers["X-Response-Time"] = f"{duration_ms}ms"
        RequestIdFilter.set_request_id(None)
        return response

    # Register API routes FIRST (before any catch-all routes)
    app.include_router(auth.router)
    app.include_router(tools.router)
    app.include_router(upload.router)
    app.include_router(chat.router)
    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(provider.router)

    # Auth middleware — verify JWT on protected routes
    app.add_middleware(AuthMiddleware)

    # Serve Web UI at root (LAST, so API routes take priority)
    @app.get("/", include_in_schema=False)
    async def serve_ui():
        if _INDEX_HTML:
            return HTMLResponse(_INDEX_HTML)
        return HTMLResponse("<h1>Matrix</h1><p>UI not found.</p>", status_code=404)

    # Mount static files (JS, CSS, etc.) — after all routes
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=False), name="static")

    return app