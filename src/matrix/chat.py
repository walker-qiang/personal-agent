"""Chat orchestration: LangGraph-based classify → react/plan/skill → summarize."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Protocol

from langgraph.checkpoint.memory import MemorySaver

from .config import AgentConfig
from .llm import LLMClient, build_llm_client
from .orchestration import build_graph
from .orchestration.state import AgentState
from .role import INVESTMENT_ANALYST, RoleDefinition
from .skills import SkillDefinition, load_skills
from .store import SessionStore
from .tools import FinanceToolError, ToolRegistry

# Default skills directory relative to the project root
_DEFAULT_SKILLS_DIR = Path("skills/investment")


class TraceSink(Protocol):
    def record(self, event: dict[str, Any]) -> None:
        ...


class ChatService:
    """LangGraph-based chat orchestration: classify → react/plan/skill → summarize."""

    def __init__(
        self,
        config: AgentConfig,
        tools: ToolRegistry,
        trace: TraceSink | None = None,
        llm: LLMClient | None = None,
        role: RoleDefinition | None = None,
        skills: list[SkillDefinition] | None = None,
        skills_dir: str | Path | None = None,
    ):
        self.config = config
        self.tools = tools
        self.trace = trace
        self.llm = llm or build_llm_client(
            provider=config.agent_provider,
            deepseek_api_key=config.deepseek_api_key,
            anthropic_api_key=config.anthropic_api_key,
            model=config.agent_model,
            deepseek_base_url=config.deepseek_base_url,
            max_tokens=config.agent_max_tokens,
            timeout_sec=config.agent_model_timeout_sec,
        )
        self.role = role or INVESTMENT_ANALYST
        self.skills = skills if skills is not None else _load_default_skills(skills_dir)
        self.store = SessionStore(config.store_path)
        self.store.backfill_titles()  # migrate existing sessions

        # Pre-build and compile the LangGraph graph once
        self._graph = build_graph()
        self._checkpointer = MemorySaver()
        self._compiled_graph = self._graph.compile(checkpointer=self._checkpointer)

    # ---- Public API ----

    def reset(self, session_id: str) -> None:
        if session_id:
            self.store.reset(session_id)

    def stream_chat(self, message: str, session_id: str | None = None) -> Iterator[dict[str, Any]]:
        """LangGraph-based streaming chat with classify → react/plan/skill → summarize."""
        started = time.perf_counter()
        sid = session_id or uuid.uuid4().hex
        text = message.strip()
        if not text:
            yield {"type": "error", "message": "message is required"}
            yield {"type": "done", "session_id": sid, "duration_ms": 0}
            return

        if not self.config.llm_available:
            yield {
                "type": "error",
                "message": f"LLM unavailable: {self.config.llm_unavailable_reason}",
            }
            yield {"type": "done", "session_id": sid, "duration_ms": 0}
            return

        # Inject conversation history into user message for context
        history = self._get_history(sid)
        full_message = text
        if history:
            history_text = "\n".join(
                f"[{h['role']}]: {h['content']}" for h in history[-4:]
            )
            full_message = f"对话历史:\n{history_text}\n\n当前问题: {text}"

        initial_state: AgentState = {
            "messages": [],
            "user_message": full_message,
            "session_id": sid,
            "intent": "",
            "skill_name": "",
            "tool_results": [],
            "tool_call_count": 0,
            "current_plan": [],
            "react_iteration": 0,
            "findings": [],
            "final_answer": "",
            "error": "",
        }

        try:
            emitted_tool_count = 0
            for event in self._compiled_graph.stream(
                initial_state,
                stream_mode="values",
                config={
                    "configurable": {
                        "llm": self.llm,
                        "tools": self.tools,
                        "role": self.role,
                        "skills": self.skills,
                        "trace": self.trace,
                    },
                    "thread_id": sid,
                },
            ):
                if not isinstance(event, dict):
                    continue

                # Yield only NEW tool calls (skip duplicates)
                tool_results = event.get("tool_results", [])
                new_count = len(tool_results)
                if new_count > emitted_tool_count:
                    for i in range(emitted_tool_count, new_count):
                        tr = tool_results[i]
                        if tr.get("duplicate"):
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
                                limit=500,
                            ),
                        }
                    emitted_tool_count = new_count

                # Yield error
                error = event.get("error", "")
                if error:
                    yield {"type": "error", "message": error}

                # Yield final answer
                answer = event.get("final_answer", "")
                if answer:
                    yield {"type": "token", "content": answer}
                    self._remember(sid, text, answer)
        except Exception as err:
            yield {"type": "error", "message": f"agent error: {err}"}
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000)
            yield {"type": "done", "session_id": sid, "duration_ms": duration_ms}

    # ---- Internal ----

    def _get_history(self, session_id: str) -> list[dict[str, str]]:
        return self.store.get_history(session_id, self.config.memory_max_turns)

    def _remember(self, session_id: str, question: str, answer: str) -> None:
        self.store.save_message(session_id, "user", question)
        self.store.save_message(session_id, "assistant", answer)
        # Auto-title: use first 30 chars of first user message
        self.store.update_title(session_id, question[:30].strip())


# ---- Module-level helpers ----

def _load_default_skills(skills_dir: str | Path | None) -> list[SkillDefinition]:
    path = Path(skills_dir) if skills_dir else _DEFAULT_SKILLS_DIR
    if not path.exists():
        return []
    return load_skills(path)


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