"""Agent-as-Tool: wrap domain agents as callable tools.

Enables hierarchical agent architectures where one agent can delegate
sub-tasks to another agent via the standard tool-calling interface.

Recursion control:
    Uses contextvars to track call depth. Max depth = 2.
    Agent A -> Agent B -> Agent C is allowed.
    Agent A -> Agent B -> Agent C -> Agent D is blocked.

Design notes:
    - The handler is a bound method of ``AgentToolWrapper``, which captures
      the ``AgentDefinition`` and a ``cfg_factory`` callable.  No global
      mutable state beyond the contextvar used for recursion tracking.
    - ``_run_domain_agent_react`` is imported lazily inside the handler to
      avoid a circular dependency (orchestration imports from tools).
    - All error paths return ``tool_error(...)`` dicts so the calling LLM
      can self-correct.
"""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

from .base import ToolDefinition, tool_error

if TYPE_CHECKING:
    from ..agent.base import AgentDefinition
    from ..agent.registry import AgentRegistry
    from .registry import ToolRegistry

logger = logging.getLogger(__name__)

# ── Recursion depth tracking (async-safe via contextvars) ────────────────────

_call_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_agent_tool_depth", default=0,
)
_MAX_DEPTH = 2

# Max ReAct iterations for agent-tool delegation (subtask-level budget).
_MAX_AGENT_TOOL_ITERATIONS = 10

# Truncate agent answers to prevent context explosion in the caller.
_MAX_ANSWER_LENGTH = 4000


# ── Protocol ─────────────────────────────────────────────────────────────────


class CfgFactory(Protocol):
    """Callable that returns the current LangGraph configurable dict.

    Since ``cfg`` is derived from the LangGraph ``RunnableConfig`` at runtime
    (and may change between graph invocations), we inject a factory callable
    rather than capturing ``cfg`` directly.  This guarantees the handler
    always operates on the latest config — e.g. updated working memory,
    circuit breaker, or history.
    """

    def __call__(self) -> dict[str, Any]:
        ...  # pragma: no cover


# ── AgentToolWrapper ─────────────────────────────────────────────────────────


@dataclass
class AgentToolWrapper:
    """Wraps an ``AgentDefinition`` as a callable tool.

    When called (i.e. when another agent invokes this tool), the wrapper:

    1. Validates the ``task`` argument.
    2. Checks recursion depth via ``contextvars`` (max depth = 2).
    3. Obtains the runtime ``cfg`` from ``cfg_factory``.
    4. Builds the agent's filtered ``ToolRegistry``.
    5. Runs a ReAct loop via ``_run_domain_agent_react``.
    6. Truncates the answer and returns a result dict.

    The wrapper holds no mutable state beyond ``agent_def`` and
    ``cfg_factory`` — no session-level caching, no implicit globals.
    """

    agent_def: AgentDefinition
    cfg_factory: CfgFactory

    # ── Tool metadata ────────────────────────────────────────────────────

    @property
    def tool_name(self) -> str:
        """Tool name: ``agent_{agent_id}`` (e.g. ``agent_investment_analyst``)."""
        return f"agent_{self.agent_def.id}"

    @property
    def tool_description(self) -> str:
        """Tool description sourced from the agent's ``description`` field."""
        return self.agent_def.description

    @property
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for the tool's single ``task`` parameter."""
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        f"要委派给 {self.agent_def.name} 完成的子任务描述。"
                        "请提供清晰、具体的任务指令。"
                    ),
                },
            },
            "required": ["task"],
        }

    @property
    def capabilities(self) -> list[str]:
        """Capability tags used by the commander for task-tool matching."""
        return ["agent_delegation"]

    # ── Handler ──────────────────────────────────────────────────────────

    def __call__(self, task: str = "") -> dict[str, Any]:
        """Tool handler: delegate *task* to the wrapped agent.

        Args:
            task: The sub-task to delegate to this agent.

        Returns:
            On success::

                {"result": str, "agent": str, "tool_results_count": int}

            On failure::

                {"error": str, "agent": str}
        """
        # ── 1. Validate input ────────────────────────────────────────────
        if not task or not task.strip():
            return tool_error(
                self.tool_name,
                "委派子任务",
                "task 参数为空",
                "请提供要委派给该 Agent 的具体任务描述",
            )

        # ── 2. Recursion depth check (contextvars) ───────────────────────
        current_depth = _call_depth.get()
        if current_depth >= _MAX_DEPTH:
            return tool_error(
                self.tool_name,
                "委派子任务",
                (
                    f"已达到最大 Agent 委派深度 ({_MAX_DEPTH})，"
                    f"当前深度: {current_depth}。继续委派可能导致无限递归。"
                ),
                "请直接使用已有工具结果回答用户，不要继续委派给其他 Agent",
                context={"agent": self.agent_def.id, "depth": current_depth},
            )

        # ── 3. Obtain runtime cfg ────────────────────────────────────────
        try:
            cfg = self.cfg_factory()
        except Exception as exc:
            return tool_error(
                self.tool_name,
                "获取运行时配置",
                f"cfg_factory 调用失败: {type(exc).__name__}: {exc}",
                "请检查 LangGraph 配置是否正确初始化",
            )

        if not cfg:
            return tool_error(
                self.tool_name,
                "获取运行时配置",
                "cfg_factory 返回空配置",
                "请确保 LangGraph configurable dict 已正确初始化",
            )

        # ── 4. Extract dependencies from cfg ─────────────────────────────
        agent_registry: AgentRegistry | None = cfg.get("agent_registry")
        full_tools: ToolRegistry | None = cfg.get("full_tools")

        if agent_registry is None:
            return tool_error(
                self.tool_name,
                "获取 Agent 注册表",
                "cfg 中缺少 'agent_registry' 键",
                "请检查 LangGraph configurable 是否包含 agent_registry",
            )
        if full_tools is None:
            return tool_error(
                self.tool_name,
                "获取工具注册表",
                "cfg 中缺少 'full_tools' 键",
                "请检查 LangGraph configurable 是否包含 full_tools",
            )

        # Build the agent's filtered tool registry
        try:
            agent_tools = agent_registry.build_tool_registry(
                self.agent_def.id, full_tools,
            )
        except ValueError as exc:
            return tool_error(
                self.tool_name,
                "构建 Agent 工具集",
                f"agent_registry.build_tool_registry 失败: {exc}",
                f"请确认 agent '{self.agent_def.id}' 已在 AgentRegistry 中注册",
            )

        # Wire circuit breaker from session config into the tool registry
        breaker = cfg.get("circuit_breaker")
        if breaker is not None:
            agent_tools.set_circuit_breaker(breaker)

        # ── 5. Run ReAct loop (increment depth) ──────────────────────────
        token = _call_depth.set(current_depth + 1)
        try:
            # Lazy import: orchestration.nodes.commander imports from tools,
            # so importing at module level would create a circular dependency.
            from ..orchestration.nodes.commander import _run_domain_agent_react

            session_id = cfg.get("session_id", "")
            result = _run_domain_agent_react(
                agent_def=self.agent_def,
                task=task,
                tools=agent_tools,
                skill_results=[],
                cfg=cfg,
                session_id=session_id,
                agent_id=self.agent_def.id,
                max_iterations=_MAX_AGENT_TOOL_ITERATIONS,
            )
        except ImportError as exc:
            logger.error(
                "agent_tool[%s]: failed to import _run_domain_agent_react: %s",
                self.agent_def.id, exc,
            )
            return tool_error(
                self.tool_name,
                "导入 ReAct 引擎",
                f"无法导入 _run_domain_agent_react: {exc}",
                "请检查 orchestration 模块是否正确安装且无循环导入",
            )
        except Exception as exc:
            logger.error(
                "agent_tool[%s]: _run_domain_agent_react failed: %s: %s",
                self.agent_def.id, type(exc).__name__, str(exc)[:200],
            )
            return tool_error(
                self.tool_name,
                "执行 Agent ReAct 循环",
                f"{type(exc).__name__}: {str(exc)[:200]}",
                "请稍后重试，或尝试将任务拆分为更小的子任务",
                context={"agent": self.agent_def.id},
            )
        finally:
            _call_depth.reset(token)

        # ── 6. Process result ────────────────────────────────────────────
        answer = result.get("answer", "")
        tool_results = result.get("tool_results", [])

        if not answer:
            return tool_error(
                self.tool_name,
                "获取 Agent 回答",
                f"Agent '{self.agent_def.id}' 返回了空回答",
                "请尝试重新描述任务，或使用其他工具/Agent",
                context={
                    "agent": self.agent_def.id,
                    "tool_results_count": len(tool_results),
                },
            )

        # Truncate to prevent context explosion in the calling agent
        truncated = False
        if len(answer) > _MAX_ANSWER_LENGTH:
            answer = answer[:_MAX_ANSWER_LENGTH] + "\n...[已截断]"
            truncated = True

        response: dict[str, Any] = {
            "result": answer,
            "agent": self.agent_def.id,
            "tool_results_count": len(tool_results),
        }
        if truncated:
            response["truncated"] = True
        return response

    # ── Conversion ───────────────────────────────────────────────────────

    def to_tool_definition(self) -> ToolDefinition:
        """Convert this wrapper into a ``ToolDefinition`` for registration.

        The ``handler`` is a bound method (``self.__call__``) which closes
        over ``agent_def`` and ``cfg_factory``.
        """
        return ToolDefinition(
            name=self.tool_name,
            description=self.tool_description,
            input_schema=self.input_schema,
            handler=self.__call__,
            capabilities=self.capabilities,
        )


# ── Factory function ─────────────────────────────────────────────────────────


def make_agent_tool(
    agent_def: AgentDefinition,
    cfg_factory: CfgFactory,
) -> ToolDefinition:
    """Create a ``ToolDefinition`` that wraps a domain agent as a callable tool.

    Args:
        agent_def: The agent definition to wrap.
        cfg_factory: Callable that returns the current cfg dict at runtime.
            This is needed because cfg comes from the LangGraph
            ``RunnableConfig`` and is only available at invocation time.

    Returns:
        A ``ToolDefinition`` with name ``agent_{agent_id}``, ready for
        registration in a ``ToolRegistry``.
    """
    wrapper = AgentToolWrapper(agent_def=agent_def, cfg_factory=cfg_factory)
    return wrapper.to_tool_definition()


# ── Batch registration ──────────────────────────────────────────────────────


def _has_agent_tool_dependency(agent_def: AgentDefinition) -> bool:
    """Check if an agent's tool list includes agent-delegation patterns.

    Agents whose ``tools`` list references ``agent_*`` tools could create
    circular delegation chains when their own agent-tool is registered.
    This is a conservative static check — the runtime depth limit
    (``_MAX_DEPTH``) is the ultimate safety net.

    Patterns checked:
    - Exact: ``agent_xxx`` (direct reference to another agent-tool)
    - Prefix: ``agent.*`` (wildcard matching all agent-tools)
    """
    for pattern in agent_def.tools:
        if pattern.startswith("agent_"):
            return True
        if pattern == "agent.*":
            return True
    return False


def register_agent_tools(
    registry: ToolRegistry,
    agent_registry: AgentRegistry,
    cfg_factory: CfgFactory,
) -> int:
    """Register all domain agents as agent-tools in the tool registry.

    Iterates over all non-commander agents (via
    ``agent_registry.list_domain_agents()``) and creates an agent-tool
    for each.  Skips agents that would create circular dependencies
    (i.e. their own ``tools`` list includes ``agent_*`` patterns).

    Args:
        registry: The ``ToolRegistry`` to register agent-tools into.
        agent_registry: The ``AgentRegistry`` to read agent definitions from.
        cfg_factory: Callable that returns the current cfg dict at runtime.

    Returns:
        The number of agent-tools successfully registered.
    """
    domain_agents = agent_registry.list_domain_agents()
    count = 0

    for agent_def in domain_agents:
        # Skip agents with agent-tool dependencies (circular prevention)
        if _has_agent_tool_dependency(agent_def):
            logger.info(
                "register_agent_tools: skipping '%s' — tool list includes "
                "agent_* patterns, would risk circular delegation",
                agent_def.id,
            )
            continue

        tool = make_agent_tool(agent_def, cfg_factory)

        try:
            registry.register(tool)
            count += 1
            logger.info(
                "register_agent_tools: registered agent-tool '%s' for agent '%s'",
                tool.name,
                agent_def.id,
            )
        except ValueError:
            logger.warning(
                "register_agent_tools: tool '%s' already registered, skipping",
                tool.name,
            )

    logger.info(
        "register_agent_tools: registered %d/%d agent-tools",
        count,
        len(domain_agents),
    )
    return count
