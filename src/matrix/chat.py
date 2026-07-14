"""Chat orchestration: Commander + Domain Agents multi-agent flow."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Protocol

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from .agent import AgentRegistry
from .agent.commander import COMMANDER
from .agent.domain_agents import INVESTMENT_ANALYST, MEDIA_GENERATOR
from .config import AgentConfig, IMAGE_MODELS, KNOWN_MODELS, VIDEO_MODELS, default_model
from .llm import LLMClient, LLMError, build_llm_client
from .llm.http import set_rate_limiter
from .orchestration import build_graph
from .orchestration.state import AgentState
from .rate_limiter import TokenBucketRateLimiter
from .store import SessionStore
from .tools import FinanceToolError, ToolRegistry


class TraceSink(Protocol):
    def record(self, event: dict[str, Any]) -> None:
        ...


logger = logging.getLogger("matrix.chat")

MEMORY_EXTRACTION_PROMPT = """从以下对话中提取用户的关键信息，以 JSON 格式返回。
只提取明确陈述的事实，不要推测。返回格式：{"memories": [{"key": "简短键名", "value": "事实描述"}]}

可提取的信息类型：
- 用户偏好（如"喜欢简洁回答"、"使用中文"）
- 关键实体（如"我持有腾讯股票"、"我的投资目标是XX"）
- 常用指令（如"每天早上查看持仓"）
- 个人信息（如"我是软件工程师"、"我在北京"）

如果没有新信息，返回 {"memories": []}。

对话：
用户：{question}
助手：{answer}"""


class ChatService:
    """LangGraph-based chat orchestration: classify → react/plan/skill → summarize → reflection."""

    def __init__(
        self,
        config: AgentConfig,
        tools: ToolRegistry,
        trace: TraceSink | None = None,
        llm: LLMClient | None = None,
        agent_registry: AgentRegistry | None = None,
        retriever: Any = None,
    ):
        self.config = config
        self.tools = tools
        self.trace = trace
        self.retriever = retriever
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

    def _load_file_content(self, file_id: str) -> str:
        """Load uploaded file content for injection into chat messages."""
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
                    return f"[图片文件: {file_path.name}]"
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
        if file_id:
            file_content = self._load_file_content(file_id)
            if file_content:
                text = f"[文件内容]\n{file_content}\n\n[用户问题]\n{text}"

        if not self.config.llm_available:
            yield {
                "type": "error",
                "message": f"LLM unavailable: {self.config.llm_unavailable_reason}",
            }
            yield {"type": "done", "session_id": sid, "duration_ms": 0}
            return

        # Load conversation history for context injection into LLM calls
        history = self._get_history(sid, user_id)
        initial_state: AgentState = {
            "messages": [],
            "user_message": text,
            "session_id": sid,
            "intent": "",
            "delegation_plan": [],
            "current_step": 0,
            "agent_results": [],
            "tool_results": [],
            "tool_call_count": 0,
            "react_iteration": 0,
            "final_answer": "",
            "needs_summary": False,
            "error": "",
        }

        try:
            emitted_tool_count = 0
            emitted_agent_count = 0
            classify_emitted = False
            final_state: dict[str, Any] = {}
            session_llm = self._get_llm(sid)
            logger.debug(
                "llm_request: provider=%s model=%s message_len=%d",
                session_llm.provider if hasattr(session_llm, 'provider') else "?",
                session_llm.model if hasattr(session_llm, 'model') else "?",
                len(text),
            )

            graph_config = {
                "configurable": {
                    "llm": session_llm,
                    "pipeline_llm": self._pipeline_llm,
                    "agent_registry": self.agent_registry,
                    "full_tools": self.tools,
                    "trace": self.trace,
                    "history": history,
                    "retriever": self.retriever,
                },
                "thread_id": sid,
            }

            try:
                for event in self._compiled_graph.stream(
                    initial_state,
                    stream_mode="values",
                    config=graph_config,
                ):
                    if not isinstance(event, dict):
                        continue
                    final_state = event

                    # Emit classify event when delegation_plan is first set by commander_plan
                    delegation_plan = event.get("delegation_plan")
                    if delegation_plan is not None and not classify_emitted:
                        classify_emitted = True
                        # Determine intent: commander-only plan = simple, multi-agent = delegate
                        intent = "delegate" if len(delegation_plan) > 1 or (
                            delegation_plan and delegation_plan[0].get("agent_id") != "commander"
                        ) else "simple"
                        yield {
                            "type": "classify",
                            "intent": intent,
                            "delegation_plan": delegation_plan,
                        }

                    # Emit agent delegation events
                    agent_results = event.get("agent_results", [])
                    if len(agent_results) > emitted_agent_count:
                        for i in range(emitted_agent_count, len(agent_results)):
                            ar = agent_results[i]
                            yield {
                                "type": "agent_result",
                                "agent_id": ar.get("agent_id", ""),
                                "task": ar.get("task", ""),
                                "result": ar.get("result", "")[:500],
                                "error": ar.get("error", ""),
                            }
                        emitted_agent_count = len(agent_results)

                    # Yield only NEW tool calls (skip duplicates)
                    tool_results = event.get("tool_results", [])
                    new_count = len(tool_results)
                    if new_count > emitted_tool_count:
                        for i in range(emitted_tool_count, new_count):
                            tr = tool_results[i]
                            if tr.get("duplicate") or tr.get("name") == "_knowledge":
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
                        emitted_tool_count = new_count

                    # Yield error
                    error = event.get("error", "")
                    if error:
                        yield {"type": "error", "message": error}

                # Streaming summarization
                if final_state.get("needs_summary"):
                    answer_parts: list[str] = []
                    for event in self._stream_summarize(final_state, sid, text, session_llm):
                        yield event
                        if event["type"] == "token":
                            answer_parts.append(event["content"])
                    answer = "".join(answer_parts)
                    if answer:
                        self._remember(sid, text, answer, user_id=user_id)
                else:
                    # Non-streaming path: yield once after reflection, then save
                    final_answer = final_state.get("final_answer", "")
                    if final_answer and not final_answer.startswith("所有领域专家"):
                        yield {"type": "token", "content": final_answer}
                        self._remember(sid, text, final_answer, user_id=user_id)

            except GraphInterrupt as gi:
                # HITL: graph paused at confirm_node, waiting for user input
                interrupt_value = gi.args[0] if gi.args else {}
                pending_actions = interrupt_value.get("actions", [])
                logger.info(
                    "hitl: confirm_required session=%s actions=%d",
                    sid, len(pending_actions),
                )
                # Store config for later resume
                self._pending_confirms[sid] = {
                    "config": graph_config,
                    "session_llm": session_llm,
                    "user_id": user_id,
                }
                yield {
                    "type": "confirm_required",
                    "actions": pending_actions,
                    "session_id": sid,
                }
                # Don't yield "done" — the stream is paused
                return

        except Exception as err:
            yield {"type": "error", "message": f"agent error: {err}"}
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000)
            self._prune_checkpoints(sid)
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
        session_llm = pending["session_llm"]
        user_id = pending["user_id"]

        logger.info(
            "hitl: resuming session=%s decision=%s",
            session_id, decision,
        )

        emitted_tool_count = 0
        emitted_agent_count = 0
        final_state: dict[str, Any] = {}

        try:
            for event in self._compiled_graph.stream(
                Command(resume=decision),
                stream_mode="values",
                config=graph_config,
            ):
                if not isinstance(event, dict):
                    continue
                final_state = event

                # Emit agent delegation events
                agent_results = event.get("agent_results", [])
                if len(agent_results) > emitted_agent_count:
                    for i in range(emitted_agent_count, len(agent_results)):
                        ar = agent_results[i]
                        yield {
                            "type": "agent_result",
                            "agent_id": ar.get("agent_id", ""),
                            "task": ar.get("task", ""),
                            "result": ar.get("result", "")[:500],
                            "error": ar.get("error", ""),
                        }
                    emitted_agent_count = len(agent_results)

                # Yield tool calls
                tool_results = event.get("tool_results", [])
                new_count = len(tool_results)
                if new_count > emitted_tool_count:
                    for i in range(emitted_tool_count, new_count):
                        tr = tool_results[i]
                        if tr.get("duplicate") or tr.get("name") == "_knowledge":
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
                    emitted_tool_count = new_count

                # Yield error
                error = event.get("error", "")
                if error:
                    yield {"type": "error", "message": error}

            # Yield final answer
            final_answer = final_state.get("final_answer", "")
            if final_answer and not final_answer.startswith("所有领域专家"):
                yield {"type": "token", "content": final_answer}

        except GraphInterrupt:
            # Another confirmation needed (unlikely but handle gracefully)
            yield {"type": "error", "message": "additional confirmation required"}
        except Exception as err:
            yield {"type": "error", "message": f"resume error: {err}"}
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000)
            self._prune_checkpoints(session_id)
            yield {"type": "done", "session_id": session_id, "duration_ms": duration_ms}

    # ---- Internal ----

    def _prune_checkpoints(self, thread_id: str) -> None:
        """Keep only the latest checkpoint per thread to prevent unbounded growth.

        LangGraph's SqliteSaver writes a checkpoint after every node execution.
        Each user question generates 5-6 checkpoints. Over time this accumulates
        useless history — only the latest checkpoint is needed to resume a
        conversation. This method deletes all but the most recent checkpoint for
        the given thread and cleans up the corresponding writes table.
        """
        try:
            conn = self._checkpoint_conn
            if conn is None:
                return
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

    def reload_skills(self) -> None:
        """Reload skills from disk (after CRUD)."""
        self.agent_registry.reload_skills()

    def _get_history(self, session_id: str, user_id: str = "default") -> list[dict[str, str]]:
        """Return conversation history with user profile injected as context."""
        history = self.store.get_history(session_id, self.config.memory_max_turns)
        profile = self.store.get_profile(user_id)
        if profile:
            # Build a compact profile summary and inject as system message
            lines = ["[用户画像]"]
            for k, v in profile.items():
                lines.append(f"- {k}: {v}")
            summary = "\n".join(lines)
            history.insert(0, {"role": "system", "content": summary})
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
            result = self._pipeline_llm.complete(prompt)
            data = json.loads(result.content)
            updated = False
            for mem in data.get("memories", []):
                key = mem["key"].strip()
                value = mem["value"].strip()
                if key and value:
                    self.store.upsert_profile(user_id, key, value)
                    logger.debug("memory_upsert: user=%s key=%s", user_id, key)
                    updated = True
            if updated and self.config.memory_sync_path:
                json_path = Path(self.config.memory_sync_path) / f"{user_id}.json"
                self.store.sync_profile_to_file(user_id, str(json_path))
        except Exception:
            pass  # Memory extraction is best-effort

    def _stream_summarize(
        self, state: dict[str, Any], session_id: str, original_text: str, llm: LLMClient
    ) -> Iterator[dict[str, Any]]:
        """Stream the LLM summarization token by token via SSE."""
        user_msg = state.get("user_message", original_text)
        tool_results = state.get("tool_results", [])

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

        user_message = f"""User question: {user_msg}

Tool results:
{json.dumps(tool_results, ensure_ascii=False, indent=2)}

Please answer the user's question using only the provided data."""

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


def preview_json(value: Any, limit: int = 1200) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def result_count(result: Any) -> int:
    """Count results from a tool return value."""
    if isinstance(result, list):
        return len(result)
    if isinstance(result, dict):
        for key in ("count", "holding_count", "bucket_count"):
            value = result.get(key)
            if isinstance(value, int):
                return value
        for key in ("assets", "snapshots", "holdings", "buckets"):
            value = result.get(key)
            if isinstance(value, list):
                return len(value)
        return 0
    return 0


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())