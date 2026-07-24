"""Chat orchestration: Commander + Domain Agents multi-agent flow."""

from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Iterator, Protocol

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from ..agent import AgentRegistry
from ..agent.commander import COMMANDER
from ..agent.domain_agents import INVESTMENT_ANALYST, MEDIA_GENERATOR
from ..config import AgentConfig, IMAGE_MODELS, KNOWN_MODELS, VIDEO_MODELS, default_model
from ..llm import LLMClient, LLMError, build_llm_client
from ..llm.http import set_rate_limiter
from ..orchestration.anti_hallucination import _strip_all_verification_tags
from ..orchestration import build_graph
from ..orchestration.state import AgentState
from ..rate_limiter import TokenBucketRateLimiter
from ..store import SessionStore
from ..tools import FinanceToolError, ToolRegistry, ToolDefinition
from ..context import ToolResultRefStore, make_get_stored_data_tool
from ..memory import EvolutionConfig, MemoryEvolution
from ._utils import(
    MEMORY_EXTRACTION_PROMPT,
    _drain_queue,
    preview_json,
    result_count,
    timestamp,
)


class TraceSink(Protocol):
    def record(self, event: dict[str, Any]) -> None:
        ...


logger = logging.getLogger("matrix.chat")


class ChatService:
    """LangGraph-based chat orchestration: classify → react/plan/skill → summarize → reflection."""

    def __init__(
        self,
        config: AgentConfig,
        tools: ToolRegistry,
        trace: TraceSink | None = None,
        llm: LLMClient | None = None,
        agent_registry: AgentRegistry | None = None,
        output_guard: object | None = None,
    ):
        self.config = config
        self.tools = tools
        self.trace = trace
        self._output_guard = output_guard  # OutputGuard or None
        self._default_llm = llm or build_llm_client(
            provider=config.agent_provider,
            deepseek_api_key=config.deepseek_api_key,
            anthropic_api_key=config.anthropic_api_key,
            agnes_api_key=config.agnes_api_key,
            model=config.agent_model,
            deepseek_base_url=config.deepseek_base_url,
            agnes_base_url=config.agnes_base_url,
            max_tokens=config.agent_max_tokens,
            timeout_sec=config.agent_model_timeout_sec,
            max_message_chars=config.max_message_chars,
        )
        self._default_provider = config.agent_provider
        self._llm_cache: dict[str, LLMClient] = {}  # per-provider+model cache

        # Pipeline LLM: fixed model for internal tasks (classify, plan, reflection)
        # When an explicit LLM is injected (e.g. tests), reuse it as pipeline_llm
        if llm is not None:
            self._pipeline_llm = llm
        else:
            self._pipeline_llm = build_llm_client(
                provider=config.pipeline_provider,
                deepseek_api_key=config.deepseek_api_key,
                anthropic_api_key=config.anthropic_api_key,
                agnes_api_key=config.agnes_api_key,
                model=config.pipeline_model,
                deepseek_base_url=config.deepseek_base_url,
                agnes_base_url=config.agnes_base_url,
                max_tokens=config.agent_max_tokens,
                timeout_sec=config.agent_model_timeout_sec,
                max_message_chars=config.max_message_chars,
            )
        # Initialize AgentRegistry
        self.agent_registry = agent_registry or _build_default_registry(config)
        self.store = SessionStore(config.store_path)
        self.store.backfill_titles()

        # L1-L4: Context management tools
        self._ref_store = ToolResultRefStore(
            config.root_path / "var" / "agent" / "tool_results.db"
        )
        # Per-user working memory insights (user_id → list[str])
        self._wm_insights: dict[str, list[str]] = {}
        # Memory evolution pipeline (consolidation, conflict resolution, forgetting)
        self._evolution = MemoryEvolution(
            self.store,
            config=EvolutionConfig(
                enable_llm_consolidation=self.config.llm_available,
            ),
            llm=self._pipeline_llm if self.config.llm_available else None,
        )
        self._register_internal_tools()

        # Configure rate limiter for LLM API calls
        if config.rate_limit_per_sec > 0:
            set_rate_limiter(TokenBucketRateLimiter(config.rate_limit_per_sec))

        # Pre-build and compile the LangGraph graph once
        self._graph = build_graph()
        self._checkpoint_conn = sqlite3.connect(
            config.checkpoint_path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._checkpointer = SqliteSaver(self._checkpoint_conn)
        self._compiled_graph = self._graph.compile(checkpointer=self._checkpointer)

        # Store pending confirmations for HITL resume
        self._pending_confirms: dict[str, dict[str, Any]] = {}

    def __enter__(self) -> "ChatService":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close resources: checkpoint database connection and session store."""
        if hasattr(self, "_checkpoint_conn") and self._checkpoint_conn:
            self._checkpoint_conn.close()
        if hasattr(self, "store") and self.store:
            self.store.close()
        if hasattr(self, "_ref_store") and self._ref_store:
            self._ref_store.close()

    # ---- Public API ----

    @property
    def available_providers(self) -> list[dict[str, Any]]:
        """List available providers with their models."""
        providers = []
        if self.config.deepseek_api_key:
            providers.append({"id": "deepseek", "name": "DeepSeek", "models": KNOWN_MODELS.get("deepseek", [])})
        if self.config.anthropic_api_key:
            providers.append({"id": "anthropic", "name": "Anthropic", "models": KNOWN_MODELS.get("anthropic", [])})
        if self.config.agnes_api_key:
            providers.append({"id": "agnes", "name": "Agnes AI", "models": KNOWN_MODELS.get("agnes", [])})
        return providers

    @property
    def available_image_models(self) -> list[dict[str, Any]]:
        """List available image generation models."""
        models = []
        if self.config.agnes_api_key:
            models.append({"provider": "agnes", "name": "Agnes AI", "models": IMAGE_MODELS.get("agnes", [])})
        return models

    @property
    def available_video_models(self) -> list[dict[str, Any]]:
        """List available video generation models."""
        models = []
        if self.config.agnes_api_key:
            models.append({"provider": "agnes", "name": "Agnes AI", "models": VIDEO_MODELS.get("agnes", [])})
        return models

    def get_provider(self, session_id: str | None = None, user_id: str = "default") -> dict[str, str]:
        """Get the LLM provider and model for a session, falling back to default."""
        if session_id:
            provider = self.store.get_provider(session_id)
            model = self.store.get_model(session_id)
            if provider:
                return {"provider": provider, "model": model or default_model(provider)}
        return {"provider": self._default_provider, "model": default_model(self._default_provider)}

    def switch_provider(self, session_id: str, provider: str, model: str = "", user_id: str = "default") -> dict[str, Any]:
        """Set the LLM provider and model for a specific session.

        Args:
            session_id: Session to configure.
            provider: One of 'deepseek', 'anthropic', 'agnes'.
            model: Specific model ID (optional, falls back to provider default).
            user_id: Authenticated user ID.

        Returns:
            dict with 'ok', 'provider', and 'model' fields.
        """
        if provider not in {"deepseek", "anthropic", "agnes"}:
            return {"ok": False, "error": f"unsupported provider: {provider}"}
        self.store.set_provider(session_id, provider, model, user_id=user_id)
        return {"ok": True, "provider": provider, "model": model or default_model(provider)}

    def _build_llm(self, provider: str, model: str | None = None) -> LLMClient:
        """Build (or return cached) LLM client for a provider+model."""
        cache_key = f"{provider}:{model or ''}"
        if cache_key not in self._llm_cache:
            self._llm_cache[cache_key] = build_llm_client(
                provider=provider,
                deepseek_api_key=self.config.deepseek_api_key,
                anthropic_api_key=self.config.anthropic_api_key,
                agnes_api_key=self.config.agnes_api_key,
                model=model or default_model(provider),
                deepseek_base_url=self.config.deepseek_base_url,
                agnes_base_url=self.config.agnes_base_url,
                max_tokens=self.config.agent_max_tokens,
                timeout_sec=self.config.agent_model_timeout_sec,
                max_message_chars=self.config.max_message_chars,
            )
        return self._llm_cache[cache_key]

    def _get_llm(self, session_id: str | None) -> LLMClient:
        """Get the LLM client for a session, using stored provider/model."""
        if session_id:
            provider = self.store.get_provider(session_id)
            if provider:
                model = self.store.get_model(session_id)
                return self._build_llm(provider, model or None)
        return self._default_llm

    def reset(self, session_id: str) -> None:
        if session_id:
            self.store.reset(session_id)

    def _load_file_content(self, file_id: str) -> str | dict[str, Any]:
        """Load uploaded file content for injection into chat messages.

        Returns:
            - str: text content for text/PDF files
            - dict: {"type": "image", "mime_type": "...", "base64": "..."} for images
        """
        upload_dir = self.config.root_path.parent / "var" / "uploads"
        for ext in (".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf"):
            file_path = upload_dir / f"{file_id}{ext}"
            if file_path.exists():
                if ext in (".txt", ".md", ".csv", ".json", ".yaml", ".yml"):
                    return file_path.read_text(encoding="utf-8", errors="replace")
                elif ext == ".pdf":
                    try:
                        import PyPDF2
                        reader = PyPDF2.PdfReader(str(file_path))
                        pages = []
                        for page in reader.pages:
                            text = page.extract_text()
                            if text:
                                pages.append(text)
                        return "\n\n".join(pages)
                    except ImportError:
                        return f"[PDF: {file_path.name}]"
                else:
                    # Image: return base64 data for vision model
                    import base64
                    content = file_path.read_bytes()
                    mime_map = {
                        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".webp": "image/webp", ".gif": "image/gif",
                    }
                    return {
                        "type": "image",
                        "mime_type": mime_map.get(ext, "image/png"),
                        "base64": base64.b64encode(content).decode("utf-8"),
                    }
        return ""

    def stream_chat(self, message: str, session_id: str | None = None, user_id: str = "default", file_id: str | None = None) -> Iterator[dict[str, Any]]:
        """LangGraph-based streaming chat with classify → react/plan/skill → summarize → reflection."""
        started = time.perf_counter()
        sid = session_id or uuid.uuid4().hex
        text = message.strip()
        if not text:
            yield {"type": "error", "message": "message is required"}
            yield {"type": "done", "session_id": sid, "duration_ms": 0}
            return

        # Inject uploaded file content into the message
        attachments: list[dict[str, Any]] = []
        if file_id:
            file_content = self._load_file_content(file_id)
            if file_content:
                if isinstance(file_content, dict) and file_content.get("type") == "image":
                    attachments.append(file_content)
                    # Add a hint in the text so classification still works
                    if not text:
                        text = "请描述这张图片的内容"
                else:
                    # Text/PDF: inject as before
                    content_str = file_content if isinstance(file_content, str) else ""
                    if content_str:
                        text = f"[文件内容]\n{content_str}\n\n[用户问题]\n{text}"

        if not self.config.llm_available:
            yield {
                "type": "error",
                "message": f"LLM unavailable: {self.config.llm_unavailable_reason}",
            }
            yield {"type": "done", "session_id": sid, "duration_ms": 0}
            return

        # Load conversation history for context injection into LLM calls
        history = self._get_history(sid, user_id)
        call_id = str(uuid.uuid4())

        # Clean up stale checkpoint from previous call (P0-4: prevents reducer merge)
        self._cleanup_stale_checkpoint(sid, call_id)

        initial_state = AgentState(
            user_message=text, session_id=sid, call_id=call_id,
            reflexion_max=self.config.reflexion_max_attempts,
            attachments=attachments,
        )

        interrupted = False
        try:
            session_llm = self._get_llm(sid)
            logger.debug(
                "llm_request: provider=%s model=%s message_len=%d",
                session_llm.provider if hasattr(session_llm, 'provider') else "?",
                session_llm.model if hasattr(session_llm, 'model') else "?",
                len(text),
            )

            graph_config = self._build_graph_config(sid, session_llm, history, text, user_id, attachments)

            try:
                final_state = yield from self._stream_graph_events(
                    initial_state, graph_config, emit_classify=True,
                )
                yield from self._finalize_stream(final_state, sid, text, session_llm, history, user_id)

            except GraphInterrupt as gi:
                interrupted = True
                yield from self._handle_hitl_interrupt(gi, sid, graph_config, session_llm, user_id)
                return

        except Exception as err:
            logger.error("stream_chat error: %s\n%s", err, traceback.format_exc())
            yield {"type": "error", "message": f"agent error: {err}"}
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000)
            if not interrupted:
                # Keep latest checkpoint for recovery (P0-4: 断点恢复)
                self._prune_checkpoints(sid, keep_latest=True)
            yield {"type": "done", "session_id": sid, "duration_ms": duration_ms}

    def resume_chat(self, session_id: str, decision: str = "approve") -> Iterator[dict[str, Any]]:
        """Resume a paused graph after user confirmation.

        Args:
            session_id: The session ID of the paused graph.
            decision: User's decision: 'approve' or 'skip'.

        Yields:
            SSE events from the resumed graph execution.
        """
        started = time.perf_counter()
        pending = self._pending_confirms.pop(session_id, None)
        if not pending:
            yield {"type": "error", "message": "no pending confirmation for this session"}
            yield {"type": "done", "session_id": session_id, "duration_ms": 0}
            return

        graph_config = pending["config"]
        # Ensure the resumed graph has an event queue for real-time streaming
        if "event_queue" not in graph_config.get("configurable", {}):
            graph_config.setdefault("configurable", {})["event_queue"] = queue.Queue()
        session_llm = pending["session_llm"]
        user_id = pending["user_id"]

        logger.info("hitl: resuming session=%s decision=%s", session_id, decision)

        try:
            final_state = yield from self._stream_graph_events(
                Command(resume=decision), graph_config, emit_classify=False,
            )

            # Yield final answer
            final_answer = final_state.get("final_answer", "")
            # FINAL SAFETY NET: strip any leaked verification tags (same as normal path)
            if final_answer:
                final_answer = _strip_all_verification_tags(final_answer)
            if final_answer and not final_answer.startswith("所有领域专家"):
                yield {"type": "token", "content": final_answer}

        except GraphInterrupt:
            # Another confirmation needed (unlikely but handle gracefully)
            yield {"type": "error", "message": "additional confirmation required"}
        except Exception as err:
            yield {"type": "error", "message": f"resume error: {err}"}
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000)
            self._prune_checkpoints(session_id, keep_latest=False)
            yield {"type": "done", "session_id": session_id, "duration_ms": duration_ms}

    # ---- Internal ----

    def _handle_working_memory(self, action: str, content: str, user_id: str = "default") -> dict:
        """Handle working_memory tool calls from the LLM.

        Used by the LLM to record key insights that survive context compression.
        Insights are isolated per user_id to prevent cross-user leakage.
        """
        if action == "add_insight" and content:
            insights = self._wm_insights.setdefault(user_id, [])
            insights.insert(0, content)
            self._wm_insights[user_id] = insights[:20]  # cap at 20 insights
            return {"ok": True, "recorded": content, "total_insights": len(self._wm_insights[user_id])}
        return {"ok": False, "error": f"Unknown action: {action}"}

    def _register_internal_tools(self) -> None:
        """Register context management tools (P0-P2: get_stored_data, working_memory).

        Extracted from __init__ to keep the constructor focused on wiring.
        """
        self.tools.register(ToolDefinition(
            name="get_stored_data",
            description=(
                "Retrieve full data that was externalized from context. "
                "Use when you see a __refId reference and need the complete data. "
                "Pass the refId from the __refId field."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "refId": {"type": "string", "description": "The reference ID from a __refId field"},
                },
                "required": ["refId"],
            },
            handler=make_get_stored_data_tool(self._ref_store),
        ))
        self.tools.register(ToolDefinition(
            name="working_memory",
            description=(
                "Record a key insight or finding that should survive context compression. "
                "Use this when you discover a critical piece of information (value, ID, "
                "constraint, decision) that future steps need to know."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add_insight"], "description": "Always 'add_insight' to record a finding"},
                    "content": {"type": "string", "description": "The insight to record. Be specific: include values, IDs, names."},
                },
                "required": ["action", "content"],
            },
            handler=self._handle_working_memory,
        ))

    def _cleanup_stale_checkpoint(self, thread_id: str, call_id: str) -> None:
        """Delete stale checkpoints from a previous call to prevent reducer merge.

        If any checkpoint exists from a previous call with the same thread_id,
        we delete it. The call_id check ensures we only clean up when starting
        a genuinely new call (not when resuming an interrupted one).
        """
        try:
            conn = self._checkpoint_conn
            if conn is None:
                return
            row = conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            if row and row[0] > 0:
                logger.info(
                    "cleanup_stale_checkpoint: removing %d stale checkpoints "
                    "thread_id=%s call_id=%s",
                    row[0], thread_id, call_id,
                )
                self._prune_checkpoints(thread_id, keep_latest=False)
        except Exception:
            # Best-effort cleanup; if it fails, the graph will still run
            pass

    def _prune_checkpoints(self, thread_id: str, keep_latest: bool = True) -> None:
        """Clean up checkpoints per thread to prevent unbounded growth.

        LangGraph's SqliteSaver writes a checkpoint after every node execution.
        Each user question generates 5-6 checkpoints. Over time this accumulates
        useless history.

        Two modes:
        - keep_latest=True (default): keeps only the latest checkpoint, used for
          HITL (interrupt/resume) where the paused state must be preserved.
        - keep_latest=False: deletes ALL checkpoints for this thread, used for
          normal call completion. This is critical because operator.add reducers
          on AgentState fields (agent_results, tool_results, tool_call_count)
          would otherwise merge stale checkpointed state into the next call,
          causing duplicate results and incorrect routing.
        """
        try:
            conn = self._checkpoint_conn
            if conn is None:
                return
            if keep_latest:
                # Delete all but the latest checkpoint for this thread
                conn.execute(
                    """DELETE FROM checkpoints
                       WHERE (thread_id, checkpoint_ns, checkpoint_id) IN (
                         SELECT thread_id, checkpoint_ns, checkpoint_id
                         FROM checkpoints
                         WHERE thread_id = ?
                         ORDER BY checkpoint_id DESC
                         LIMIT -1 OFFSET 1
                       )""",
                    (thread_id,),
                )
            else:
                # Delete all checkpoints for this thread
                conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?",
                    (thread_id,),
                )
            # Clean orphaned writes (no matching checkpoint)
            conn.execute(
                """DELETE FROM writes
                   WHERE (thread_id, checkpoint_ns, checkpoint_id) NOT IN (
                     SELECT thread_id, checkpoint_ns, checkpoint_id FROM checkpoints
                   )""",
            )
            conn.execute("PRAGMA optimize")
        except Exception:
            pass  # Pruning is best-effort; never fail the chat for it

    # ---- Stream event helpers (shared between stream_chat and resume_chat) ----

    def _stream_graph_events(
        self,
        graph_input: Any,
        graph_config: dict[str, Any],
        *,
        emit_classify: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Stream LangGraph events with common agent/tool/error emission.

        Args:
            graph_input: Initial state (for stream_chat) or Command(resume=...) (for resume_chat).
            graph_config: LangGraph config dict with configurable and thread_id.
            emit_classify: If True, emit a classify event when delegation_plan appears.

        Yields:
            SSE event dicts (classify, progress, tool_call, tool_result, agent_result, error, token).
        """
        emitted_tool_count = 0
        emitted_agent_count = 0
        classify_emitted = False
        _queue_emitted: set[tuple[str, str]] = set()
        final_state: dict[str, Any] = {}

        for event in self._compiled_graph.stream(
            graph_input, stream_mode="values", config=graph_config,
        ):
            yield from _drain_queue(graph_config["configurable"]["event_queue"], _queue_emitted)
            if not isinstance(event, dict):
                continue
            final_state = event

            # Emit classify event (stream_chat only)
            if emit_classify and not classify_emitted:
                delegation_plan = event.get("delegation_plan")
                if delegation_plan and len(delegation_plan) > 0:
                    classify_emitted = True
                    intent = "delegate" if len(delegation_plan) > 1 or (
                        delegation_plan and delegation_plan[0].get("agent_id") != "commander"
                    ) else "simple"
                    yield {"type": "classify", "intent": intent, "delegation_plan": delegation_plan}

            # Emit agent events
            agent_results = event.get("agent_results", [])
            if len(agent_results) > emitted_agent_count:
                emitted_agent_count = yield from self._emit_agent_events(
                    agent_results, emitted_agent_count,
                )

            # Emit tool events
            tool_results = event.get("tool_results", [])
            if len(tool_results) > emitted_tool_count:
                emitted_tool_count = yield from self._emit_tool_events(
                    tool_results, emitted_tool_count, _queue_emitted,
                )

            # Emit error
            error = event.get("error", "")
            if error:
                yield {"type": "error", "message": error}

        return final_state

    def _emit_agent_events(
        self, agent_results: list[dict], emitted_agent_count: int,
    ) -> int:
        """Yield agent_result events for newly emitted agents. Returns new count."""
        new_count = emitted_agent_count
        for i in range(emitted_agent_count, len(agent_results)):
            ar = agent_results[i]
            yield {
                "type": "agent_result",
                "agent_id": ar.get("agent_id", ""),
                "task": ar.get("task", ""),
                "result": ar.get("result", "")[:500],
                "error": ar.get("error", ""),
            }
            new_count += 1
        return new_count

    def _emit_tool_events(
        self, tool_results: list[dict], emitted_tool_count: int,
        queue_emitted: set[tuple[str, str]],
    ) -> int:
        """Yield tool_call + tool_result events for new tools. Returns new count."""
        new_count = emitted_tool_count
        for i in range(emitted_tool_count, len(tool_results)):
            tr = tool_results[i]
            if tr.get("duplicate") or tr.get("name") == "_knowledge":
                continue
            tr_key = (tr.get("name", ""), json.dumps(tr.get("arguments", {}), sort_keys=True))
            if tr_key in queue_emitted:
                continue
            yield {
                "type": "tool_call",
                "name": tr.get("name", ""),
                "args": tr.get("arguments", {}),
            }
            yield {
                "type": "tool_result",
                "name": tr.get("name", ""),
                "preview": preview_json(
                    tr.get("error", tr.get("result", {})),
                    limit=2000,
                ),
            }
            new_count += 1
        return new_count

    def _build_graph_config(
        self, sid: str, session_llm: LLMClient, history: list[dict],
        user_message: str = "", user_id: str = "default",
        attachments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build the LangGraph config dict for a streaming session."""
        return {
            "configurable": {
                "llm": session_llm,
                "pipeline_llm": self._pipeline_llm,
                "agent_registry": self.agent_registry,
                "full_tools": self.tools,
                "trace": self.trace,
                "history": history,
                "event_queue": queue.Queue(),
                "ref_store": self._ref_store,
                "attachments": attachments or [],
                "working_memory": {
                    "pinned": user_message,
                    "insights": list(self._wm_insights.get(user_id, [])),
                },
            },
            "thread_id": sid,
        }

    def _finalize_stream(
        self, final_state: dict[str, Any], sid: str, text: str,
        session_llm: LLMClient, history: list[dict], user_id: str,
    ) -> Iterator[dict[str, Any]]:
        """Handle output after graph streaming completes: summarize or direct answer."""
        if final_state.get("needs_summary"):
            answer_parts: list[str] = []
            for event in self._stream_summarize(final_state, sid, text, session_llm, history):
                yield event
                if event["type"] == "token":
                    answer_parts.append(event["content"])
            answer = "".join(answer_parts)
            # ---- OUTPUT GUARD ----
            if answer and self._output_guard:
                result = self._output_guard.check(answer, user_id=user_id)
                if result.had_pii:
                    logger.warning("output_pii_detected: flags=%s session=%s", result.flags, sid)
                answer = result.sanitized
            # ---- END OUTPUT GUARD ----
            if answer:
                self._remember(sid, text, answer, user_id=user_id)
        else:
            final_answer = final_state.get("final_answer", "")
            # ---- FINAL SAFETY NET: strip any leaked verification tags ----
            # This is the last gate before output reaches the user. Regardless
            # of which internal path produced this answer (ReAct, aggregate,
            # reflection revision, commander pass-through, error fallback),
            # strip ALL verification markup here so it can NEVER leak to UI.
            if final_answer:
                final_answer = _strip_all_verification_tags(final_answer)
            # ---- END SAFETY NET ----
            # ---- OUTPUT GUARD ----
            if final_answer and self._output_guard:
                result = self._output_guard.check(final_answer, user_id=user_id)
                if result.had_pii:
                    logger.warning("output_pii_detected: flags=%s session=%s", result.flags, sid)
                final_answer = result.sanitized
            # ---- END OUTPUT GUARD ----
            if final_answer and not final_answer.startswith("所有领域专家"):
                yield {"type": "token", "content": final_answer}
                self._remember(sid, text, final_answer, user_id=user_id)

    def _handle_hitl_interrupt(
        self, gi: Any, sid: str, graph_config: dict[str, Any],
        session_llm: LLMClient, user_id: str,
    ) -> Iterator[dict[str, Any]]:
        """Handle GraphInterrupt: store pending state and yield HITL events."""
        interrupt_value = gi.args[0] if gi.args else {}
        pending_actions = interrupt_value.get("actions", [])
        logger.info(
            "hitl: confirm_required session=%s actions=%d",
            sid, len(pending_actions),
        )
        self._pending_confirms[sid] = {
            "config": graph_config,
            "session_llm": session_llm,
            "user_id": user_id,
        }
        self._prune_checkpoints(sid, keep_latest=True)
        yield {
            "type": "confirm_required",
            "actions": pending_actions,
            "session_id": sid,
        }

    def reload_skills(self) -> None:
        """Reload skills from disk (after CRUD)."""
        self.agent_registry.reload_skills()

    def _get_history(self, session_id: str, user_id: str = "default") -> list[dict[str, str]]:
        """Return conversation history with layered user profile injected as context."""
        history = self.store.get_history(session_id, self.config.memory_max_turns)
        formatted = self.store.get_profile_formatted(user_id)
        if formatted:
            history.insert(0, {"role": "system", "content": formatted})
        return history

    def _remember(self, session_id: str, question: str, answer: str, user_id: str = "default") -> None:
        self.store.save_message(session_id, "user", question, user_id=user_id)
        self.store.save_message(session_id, "assistant", answer, user_id=user_id)
        self.store.update_title(session_id, question[:30].strip())
        # Extract memories in background thread (non-blocking)
        threading.Thread(
            target=self._extract_memories,
            args=(question, answer, user_id),
            daemon=True,
        ).start()

    def _extract_memories(self, question: str, answer: str, user_id: str) -> None:
        """Extract key facts from conversation and store in user profile."""
        try:
            prompt = MEMORY_EXTRACTION_PROMPT.format(
                question=question[:500], answer=answer[:1000],
            )
            data = self._pipeline_llm.complete_json(
                prompt, [{"role": "user", "content": "Extract memories from this Q&A."}],
            )
            updated = False
            for mem in data.get("memories", []):
                key = mem["key"].strip()
                value = mem["value"].strip()
                mem_type = mem.get("type", "preference").strip()
                if key and value:
                    self.store.upsert_profile(user_id, key, value, memory_type=mem_type)
                    logger.debug("memory_upsert: user=%s key=%s type=%s", user_id, key, mem_type)
                    updated = True
            if updated and self.config.memory_sync_path:
                json_path = Path(self.config.memory_sync_path) / f"{user_id}.json"
                self.store.sync_profile_to_file(user_id, str(json_path))
        except Exception:
            pass  # Memory extraction is best-effort

        # Run memory evolution (conflict resolution, consolidation, forgetting)
        try:
            report = self._evolution.evolve(user_id)
            if report.total_before != report.total_after:
                logger.info(
                    "memory_evolved: user=%s %s", user_id, str(report),
                )
        except Exception:
            pass  # Evolution is best-effort

    def _stream_summarize(
        self, state: dict[str, Any], session_id: str, original_text: str, llm: LLMClient,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream the LLM summarization token by token via SSE."""
        user_msg = state.get("user_message", original_text)
        tool_results = state.get("tool_results", [])
        attachments = state.get("attachments", [])

        system_prompt = """You are a helpful AI assistant. Answer the user's question using only the provided data.
Rules:
- Use only the provided data, never fabricate
- Money is CNY unless stated otherwise, format large numbers with commas
- Keep answers concise and well-structured
- Reply in the same language as the user
- Use Markdown formatting: **bold** for key figures, bullet lists for breakdowns
- If the result contains an image URL, display it using ![description](URL) format
- Do NOT include execution process review, agent status tables, or step-by-step workflow
- Your output is for the end user, not an internal log"""

        # Build conversation history context for multi-turn awareness
        history_context = ""
        if history:
            recent = history[-6:]  # last 3 turns
            lines = []
            for h in recent:
                role_label = "用户" if h["role"] == "user" else "助手"
                lines.append(f"[{role_label}]: {h['content'][:300]}")
            history_context = "对话历史：\n" + "\n".join(lines) + "\n\n"

        user_message_text = f"""User question: {user_msg}

Tool results:
{json.dumps(tool_results, ensure_ascii=False, indent=2)}

Please answer the user's question using only the provided data."""

        # Build multi-modal user message if attachments present
        if attachments:
            content_blocks: list[dict[str, Any]] = [
                {"type": "text", "text": history_context + user_message_text},
            ]
            for att in attachments:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{att['mime_type']};base64,{att['base64']}"},
                })
            user_message: str | list[dict[str, Any]] = content_blocks
        else:
            user_message = history_context + user_message_text

        full_answer: list[str] = []
        try:
            for token in llm.stream_complete(
                system_prompt, [{"role": "user", "content": user_message}]
            ):
                full_answer.append(token)
                yield {"type": "token", "content": token}
        except LLMError as err:
            yield {"type": "error", "message": f"LLM error: {err}"}
            full_answer = ["无法生成回答，请查看原始数据。"]
            yield {"type": "token", "content": full_answer[0]}
        except Exception:
            full_answer = ["无法生成回答，请查看原始数据。"]
            yield {"type": "token", "content": full_answer[0]}

        return "".join(full_answer).strip()


# ---- Module-level helpers ----

def _build_default_registry(config: AgentConfig) -> AgentRegistry:
    """Build the default AgentRegistry with commander and domain agents."""
    registry = AgentRegistry(skills_base_dir=config.skills_base_dir)
    registry.register_all([
        COMMANDER,
        INVESTMENT_ANALYST,
        MEDIA_GENERATOR,
    ])
    return registry


