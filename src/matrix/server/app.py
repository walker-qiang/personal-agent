"""FastAPI application factory for the Agent HTTP server."""

from __future__ import annotations

import os
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from .stream_ticket import StreamTicketStore

from ..chat import ChatService
from ..config import AgentConfig, load_config
from ..guardrails import GuardrailPipeline, GuardConfig
from ..logging_config import RequestIdFilter, get_logger, setup_logging
from ..observability.trace import TraceLogger
from ..tools import ToolRegistry
from ..tools.finance import register_all as register_finance_tools
from ..tools.web import register_all as register_web_tools
from ..tools.agnes import register_all as register_agnes_tools
from ..tools.adapters.personal_os import register_all as register_personal_os_tools
from ..tools.rag import register_all as register_rag_tools
from .routes import auth, chat, health, memory, provider, runtime, sessions, tools, trace, upload
from .middleware import AuthMiddleware

logger = get_logger("matrix")

_REACT_INDEX_HTML = ""
_react_index_path = Path(__file__).parent / "static" / "react-app" / "index.html"
if _react_index_path.exists():
    _REACT_INDEX_HTML = _react_index_path.read_text(encoding="utf-8")


def _build_rag(config: AgentConfig, tools_registry: ToolRegistry) -> tuple[object, object | None]:
    """在工作线程中构建 RAG，避免阻塞 Agent HTTP 服务启动。"""
    from ..rag.embedder import LocalEmbedder
    from ..rag.retriever import HybridRetriever
    from ..rag.indexer import DocumentIndexer
    from ..rag.knowledge_graph import KnowledgeGraph, EntityExtractor, GraphRetriever

    embedder = LocalEmbedder(model_name=config.rag_embed_model)
    kg_persist_path = os.path.join(config.rag_persist_dir, "knowledge_graph.json")
    knowledge_graph = KnowledgeGraph(persist_path=kg_persist_path)
    graph_retriever = GraphRetriever(knowledge_graph)

    indexer = DocumentIndexer(
        embedder=embedder,
        persist_dir=config.rag_persist_dir,
        knowledge_graph=knowledge_graph,
    )
    chunk_count = indexer.index_directory(config.rag_docs_path)
    logger.info(
        "rag: indexed %d chunks from %s (persist=%s)",
        chunk_count, config.rag_docs_path, config.rag_persist_dir,
    )

    if not knowledge_graph.is_empty:
        if knowledge_graph.dirty:
            knowledge_graph.save()
        else:
            logger.info("rag: knowledge graph unchanged, skip save")
        logger.info("rag: knowledge graph — %s", knowledge_graph.stats)

    retriever = HybridRetriever(
        embedder=embedder,
        persist_dir=config.rag_persist_dir,
        rebuild_bm25=indexer.last_index_changed,
    )

    agentic_search = None
    if config.llm_available:
        try:
            from ..rag.agentic_search import AgenticSearch
            from ..llm import build_llm_client

            pipeline_llm = build_llm_client(
                provider=config.pipeline_provider,
                deepseek_api_key=config.deepseek_api_key,
                anthropic_api_key=config.anthropic_api_key,
                agnes_api_key=config.agnes_api_key,
                model=config.pipeline_model,
                deepseek_base_url=(
                    config.agnes_base_url
                    if config.pipeline_provider == "agnes"
                    else config.deepseek_base_url
                ),
                codex_bin=config.codex_bin,
                codex_workdir=config.codex_workdir,
                codex_sandbox=config.codex_sandbox,
                codex_reasoning_effort=config.codex_reasoning_effort,
                max_tokens=config.agent_max_tokens,
                timeout_sec=config.agent_model_timeout_sec,
            )

            def _web_fallback(query: str) -> dict:
                web_tool = tools_registry.get("web_search")
                if web_tool and web_tool.handler:
                    return web_tool.handler(query=query)
                return {"results": []}

            agentic_search = AgenticSearch(
                retriever=retriever,
                llm=pipeline_llm,
                web_search_fn=_web_fallback,
                graph_retriever=graph_retriever,
            )
            logger.info("rag: agentic search enabled (query rewriting + grading + knowledge graph)")
        except Exception as agentic_exc:
            logger.warning("rag: agentic search init failed (using simple retrieval): %s", agentic_exc)

    return retriever, agentic_search


async def _initialize_rag(
    app: FastAPI,
    config: AgentConfig,
    tools_registry: ToolRegistry,
) -> None:
    """异步初始化 RAG；同步模型/索引工作放入线程池。"""
    try:
        retriever, agentic_search = await asyncio.to_thread(
            _build_rag, config, tools_registry,
        )
        app.state.retriever = retriever
        register_rag_tools(
            tools_registry,
            retriever=retriever,
            agentic_search=agentic_search,
        )
        app.state.rag_status = "ready"
        logger.info("rag: retriever ready")
    except Exception as exc:
        app.state.rag_error = str(exc)
        app.state.rag_status = "failed"
        logger.warning("rag: initialization failed (will run without RAG): %s", exc)


def _start_rag_warmup(
    app: FastAPI,
    config: AgentConfig,
    tools_registry: ToolRegistry,
) -> str:
    """Schedule the single RAG initialization task when warmup is requested."""
    status = getattr(app.state, "rag_status", "disabled")
    task = getattr(app.state, "rag_task", None)
    if status == "ready" or (task is not None and not task.done()):
        return status
    if not config.rag_docs_path or not Path(config.rag_docs_path).is_dir():
        app.state.rag_status = "disabled"
        return "disabled"

    app.state.rag_error = ""
    app.state.rag_status = "initializing"
    app.state.rag_task = asyncio.create_task(
        _initialize_rag(app, config, tools_registry),
    )
    logger.info("rag: warmup scheduled")
    return "initializing"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize application state on startup."""
    config: AgentConfig = app.state.config
    tools_registry = ToolRegistry()
    register_finance_tools(tools_registry, config.cache_path)
    register_web_tools(tools_registry)
    register_agnes_tools(tools_registry)
    register_personal_os_tools(tools_registry)

    # ---- GUARDRAILS ----
    guard_config = GuardConfig.from_env()
    guardrails = GuardrailPipeline(guard_config)
    # Wire TraceSanitizer to TraceLogger (now with OTel support)
    trace = TraceLogger(
        config.trace_path,
        sanitizer=guardrails.privacy,
        service_name="matrix-agent",
        otlp_endpoint=config.otel_exporter_endpoint,
        otlp_export=config.otel_export,
    )
    # Wire ToolGuard to ToolRegistry
    if guardrails.tool:
        tools_registry.set_guard(guardrails.tool)
    # Wire IndirectInjectionGuard to ToolRegistry
    if guardrails.injection:
        tools_registry.set_injection_guard(guardrails.injection)
    # ---- END GUARDRAILS ----

    app.state.tools = tools_registry
    app.state.trace = trace
    app.state.guardrails = guardrails
    app.state.stream_tickets = StreamTicketStore()
    app.state.chat = ChatService(
        config, tools_registry, trace,
        output_guard=guardrails.output,
    )
    app.state.runtime_store = app.state.chat._runtime_store
    runtime_store = app.state.chat._runtime_store
    recovered = runtime_store.recover_incomplete()
    if recovered:
        logger.warning(
            "runtime: marked %d interrupted operation(s) as recovery_required; "
            "no side effects were replayed",
            len(recovered),
        )
    incomplete = runtime_store.list_incomplete()
    waiting = [item for item in incomplete if item.phase.value == "waiting_approval"]
    if waiting:
        logger.info(
            "runtime: retained %d approval-waiting operation(s) for user resume",
            len(waiting),
        )
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

    # Keep startup light; the product client requests warmup after its first load.
    app.state.retriever = None
    app.state.rag_error = ""
    app.state.rag_status = "disabled"
    app.state.rag_task = None
    app.state.start_rag_warmup = lambda: _start_rag_warmup(
        app, config, tools_registry,
    )
    if config.rag_docs_path and Path(config.rag_docs_path).is_dir():
        app.state.rag_status = "pending"

    # ---- CODE SANDBOX ----
    if config.code_sandbox_enabled:
        from ..tools.code import register_all as register_code_tools

        code_guard = register_code_tools(
            tools_registry,
            timeout_sec=config.code_sandbox_timeout_sec,
            max_memory_mb=config.code_sandbox_max_memory_mb,
            max_output_chars=config.code_sandbox_max_output_chars,
            network_enabled=config.code_sandbox_network,
        )
        tools_registry.set_code_guard(code_guard)
        logger.info(
            "code: sandbox enabled (timeout=%ds, memory=%dMB)",
            config.code_sandbox_timeout_sec,
            config.code_sandbox_max_memory_mb,
        )
    # ---- END CODE SANDBOX ----

    # ---- MCP CLIENT ----
    # Connect to external MCP servers and register their tools
    app.state.mcp_client = None
    app.state.mcp_error = ""
    try:
        from ..tools.mcp import init_mcp_client, register_mcp_tools

        mcp_manager = init_mcp_client(config.mcp_config_path)
        if mcp_manager is not None:
            mcp_count = register_mcp_tools(tools_registry, mcp_manager)
            app.state.mcp_client = mcp_manager
            logger.info("mcp: %d tools registered", mcp_count)
    except Exception as exc:
        app.state.mcp_error = str(exc)
        logger.warning("mcp: initialization failed (will run without MCP): %s", exc)
    # ---- END MCP CLIENT ----

    try:
        yield
    finally:
        # ---- Cleanup ----
        rag_task = getattr(app.state, "rag_task", None)
        if rag_task is not None and not rag_task.done():
            try:
                await rag_task
            except Exception as exc:
                logger.warning("rag: shutdown while initialization was pending: %s", exc)
        # Disconnect MCP servers on shutdown.
        if app.state.mcp_client is not None:
            try:
                app.state.mcp_client.stop()
            except Exception as exc:
                logger.warning("mcp: shutdown error: %s", exc)
        if getattr(app.state, "chat", None) is not None:
            try:
                app.state.chat.close()
            except Exception as exc:
                logger.warning("chat: shutdown error: %s", exc)
        if getattr(app.state, "trace", None) is not None:
            try:
                app.state.trace.close()
            except Exception as exc:
                logger.warning("trace: shutdown error: %s", exc)


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
    app.include_router(trace.router)
    app.include_router(memory.router)
    app.include_router(runtime.router)

    # MCP server management routes
    from .routes import mcp as mcp_routes
    app.include_router(mcp_routes.router)

    # Auth middleware — verify JWT on protected routes
    app.add_middleware(AuthMiddleware)

    # Serve React SPA at root (LAST, so API routes take priority)
    @app.get("/", include_in_schema=False)
    async def serve_ui():
        if _REACT_INDEX_HTML:
            return HTMLResponse(_REACT_INDEX_HTML, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
        return HTMLResponse("<h1>Matrix</h1><p>React SPA not found.</p>", status_code=404)

    # Serve React SPA at /react-app/
    @app.get("/react-app/", include_in_schema=False)
    @app.get("/react-app", include_in_schema=False)
    async def serve_react_app():
        if _REACT_INDEX_HTML:
            return HTMLResponse(_REACT_INDEX_HTML, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
        return HTMLResponse("<h1>Matrix</h1><p>React SPA not found.</p>", status_code=404)


    # Mount React SPA static files (assets/) — after all routes
    # NOTE: StaticFiles at "/" overrides the @app.get("/") route in Starlette.
    # Mount at a sub-path instead to serve only asset files, not the root index.
    react_dir = Path(__file__).parent / "static" / "react-app"
    if react_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(react_dir / "assets"), html=False), name="static")

    return app
