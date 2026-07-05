"""Chat orchestration: Planner-Final two-phase engine with streaming SSE output."""

from __future__ import annotations

import json
import re
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


READ_ONLY_SYSTEM_PROMPT = """You are the read-only personal-os finance agent.

Rules:
- Use finance tools for factual answers about assets, holdings, allocation, balances, targets, or snapshots.
- Never modify data or suggest that you changed data. If the user asks for writes, tell them to use the existing UI.
- Money is CNY unless a tool result explicitly says otherwise.
- Do not describe balance changes as investment returns unless the user provided return data.
- Cite durable asset ids or asset codes when they matter.
- Keep answers concise and reply in the same language as the user.
"""

PLANNER_PROMPT = (
    READ_ONLY_SYSTEM_PROMPT
    + """
You are choosing read-only tools before the final answer.
Return exactly one JSON object and no prose:
{"tool_calls":[{"name":"finance.bucket_allocation","arguments":{}}]}

Use only the listed tool names and valid JSON arguments. Return {"tool_calls":[]} when no more tool data is needed.
Use finance.recent_snapshots for latest snapshot questions, optionally filtered by asset code, name, bucket, or asset type.
For snapshot history, call finance.asset_lookup first if the user gave only a code or name. Call finance.snapshot_history only with an ast_* durable asset id.
"""
)

FINAL_PROMPT = (
    READ_ONLY_SYSTEM_PROMPT
    + """
Answer using only the provided tool results and conversation context. If tool data is missing, say what is missing instead of guessing.
"""
)

# Default skills directory relative to the project root
_DEFAULT_SKILLS_DIR = Path("skills/investment")


class TraceSink(Protocol):
    def record(self, event: dict[str, Any]) -> None:
        ...


class ToolCall:
    """A parsed tool call from the planner LLM response."""

    __slots__ = ("name", "arguments")

    def __init__(self, name: str, arguments: dict[str, Any]) -> None:
        self.name = name
        self.arguments = arguments

    def __hash__(self) -> int:
        return hash((self.name, json.dumps(self.arguments, ensure_ascii=False, sort_keys=True)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCall):
            return NotImplemented
        return self.name == other.name and self.arguments == other.arguments


class ChatService:
    """Planner-Final two-phase chat orchestration engine.

    Supports two modes:
    - ``mode=planner`` (default): legacy Planner-Final two-phase
    - ``mode=graph``: LangGraph-based classify → react/plan/skill → summarize
    """

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

        # Pre-build and compile the LangGraph graph once
        self._graph = build_graph()
        self._checkpointer = MemorySaver()
        self._compiled_graph = self._graph.compile(checkpointer=self._checkpointer)

    # ---- Public API ----

    def reset(self, session_id: str) -> None:
        if session_id:
            self.store.reset(session_id)

    def stream_chat(self, message: str, session_id: str | None = None) -> Iterator[dict[str, Any]]:
        """Legacy Planner-Final streaming chat (mode=planner)."""
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

        answer = ""
        tool_results: list[dict[str, Any]] = []
        seen_calls: set[ToolCall] = set()
        try:
            for _ in range(3):
                planner_messages = self._planner_messages(sid, text, tool_results)
                plan_text = self.llm.complete(PLANNER_PROMPT, planner_messages)
                try:
                    calls = parse_tool_calls(plan_text, self.tools.tool_names())
                except ValueError:
                    if tool_results:
                        break
                    raise
                calls = [call for call in calls if call not in seen_calls]
                if not calls:
                    break
                for call in calls[:4]:
                    seen_calls.add(call)
                    yield {
                        "type": "tool_call",
                        "name": call.name,
                        "args": call.arguments,
                    }
                    result = self._call_tool(call.name, call.arguments)
                    tool_results.append(
                        {
                            "name": call.name,
                            "arguments": call.arguments,
                            "result": result,
                        }
                    )
                    yield {
                        "type": "tool_result",
                        "name": call.name,
                        "preview": preview_json(result),
                    }

            answer = self.llm.complete(
                FINAL_PROMPT, self._final_messages(sid, text, tool_results)
            ).strip()
            if not answer:
                answer = "没有生成可用回答。"
            yield {"type": "token", "content": answer}
            self._remember(sid, text, answer)
        except (FinanceToolError, ValueError) as err:
            yield {"type": "error", "message": str(err)}
        except Exception as err:
            yield {"type": "error", "message": f"agent error: {err}"}
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000)
            yield {"type": "done", "session_id": sid, "duration_ms": duration_ms}

    def stream_chat_graph(self, message: str, session_id: str | None = None) -> Iterator[dict[str, Any]]:
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
            yield {"type": "error", "message": f"graph agent error: {err}"}
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000)
            yield {"type": "done", "session_id": sid, "duration_ms": duration_ms}

    # ---- Internal ----

    def _planner_messages(
        self, session_id: str, message: str, tool_results: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        payload = {
            "available_tools": self.tools.list_tools(),
            "conversation": self._get_history(session_id),
            "question": message,
            "tool_results": compact_tool_results(tool_results),
        }
        return [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]

    def _final_messages(
        self, session_id: str, message: str, tool_results: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        payload = {
            "conversation": self._get_history(session_id),
            "question": message,
            "tool_results": tool_results,
        }
        return [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]

    def _get_history(self, session_id: str) -> list[dict[str, str]]:
        return self.store.get_history(session_id, self.config.memory_max_turns)

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            result = self.tools.call(name, arguments)
            self._trace(
                {
                    "ok": True,
                    "chat": True,
                    "tool": name,
                    "arguments": arguments,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                    "result_count": result_count(result),
                    "ts": timestamp(),
                }
            )
            return result
        except FinanceToolError as err:
            self._trace(
                {
                    "ok": False,
                    "chat": True,
                    "tool": name,
                    "arguments": arguments,
                    "error": str(err),
                    "ts": timestamp(),
                }
            )
            raise

    def _trace(self, event: dict[str, Any]) -> None:
        if self.trace is not None:
            self.trace.record(event)

    def _remember(self, session_id: str, question: str, answer: str) -> None:
        self.store.save_message(session_id, "user", question)
        self.store.save_message(session_id, "assistant", answer)


# ---- Module-level helpers ----

def _load_default_skills(skills_dir: str | Path | None) -> list[SkillDefinition]:
    path = Path(skills_dir) if skills_dir else _DEFAULT_SKILLS_DIR
    if not path.exists():
        return []
    return load_skills(path)


def parse_tool_calls(text: str, allowed: set[str]) -> list[ToolCall]:
    """Parse tool calls from planner LLM response, validating against allowed set."""
    payload = extract_json_object(text)
    calls = payload.get("tool_calls", [])
    if not isinstance(calls, list):
        raise ValueError("planner response tool_calls must be a list")
    parsed: list[ToolCall] = []
    for item in calls[:4]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name not in allowed:
            raise ValueError(f"planner selected unknown tool: {name}")
        arguments = item.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ValueError(f"planner arguments for {name} must be an object")
        parsed.append(ToolCall(name=name, arguments=arguments))
    return parsed


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM output, handling markdown fences."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    elif not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("planner response must be a JSON object")
    return data


def compact_tool_results(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": row["name"],
            "arguments": row["arguments"],
            "result_preview": preview_json(row["result"], limit=3000),
        }
        for row in tool_results
    ]


def preview_json(value: Any, limit: int = 1200) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def result_count(result: dict[str, Any]) -> int:
    for key in ("count", "holding_count", "bucket_count"):
        value = result.get(key)
        if isinstance(value, int):
            return value
    for key in ("assets", "snapshots", "holdings", "buckets"):
        value = result.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())