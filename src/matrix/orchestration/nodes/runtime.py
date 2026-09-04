"""LangGraph adapter for one Runtime-managed domain-agent operation."""

from __future__ import annotations

from typing import Any

from langgraph.types import RunnableConfig, interrupt

from ...runtime import AgentRuntime, ExecutionPolicy, RunRequest
from ...runtime.adapters.model import MatrixModelAdapter
from ...runtime.adapters.tools import MatrixToolAdapter, tool_specs
from ...runtime.adapters.context import MatrixContextAdapter
from ...runtime.domain.messages import Message
from ...runtime.domain.results import RunOutcome
from ...runtime.domain.tools import ToolResult
from ...runtime.testing.memory_store import MemoryOperationStore
from ._helpers import (
    _build_history_context,
    _inject_agent_guidelines,
    _inject_data_index,
    _inject_lessons,
    _inject_working_memory,
    _push_event,
    _today_cn,
    DOMAIN_AGENT_REACT_SYSTEM,
    _get_configurable,
)
from ..state import AgentState
from ..runtime_adapter import build_multimodal_content, run_dag_step


def runtime_agent_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Execute an eligible single step through the independent Runtime."""

    cfg = _get_configurable(config)
    agent_registry = cfg["agent_registry"]
    full_tools = cfg["full_tools"]
    plan = state.get("delegation_plan", [])
    current_step = state.get("current_step", 0)
    step = plan[current_step] if current_step < len(plan) else {}
    agent_id = step.get("agent_id", "commander")
    task = step.get("task", state.get("user_message", ""))
    agent_def = agent_registry.get(agent_id)
    if agent_def is None:
        return {"agent_results": [{"agent_id": agent_id, "task": task, "error": f"Agent not found: {agent_id}"}]}

    agent_tools = agent_registry.build_tool_registry(agent_id, full_tools)
    breaker = cfg.get("circuit_breaker")
    if breaker is not None:
        agent_tools.set_circuit_breaker(breaker)
    history_context = _build_history_context(cfg.get("history", []))
    system_prompt = DOMAIN_AGENT_REACT_SYSTEM.format(
        agent_name=agent_def.name,
        persona=agent_def.persona,
        task=task,
        today=_today_cn(),
    )
    system_prompt = _inject_working_memory(
        system_prompt, state.get("working_memory", {}), state.get("messages", []),
    )
    system_prompt = _inject_agent_guidelines(system_prompt, agent_def)
    system_prompt = _inject_data_index(system_prompt, cfg.get("ref_store"), state.get("messages", []))
    system_prompt = _inject_lessons(system_prompt, task, agent_id, cfg)

    request = RunRequest(
        owner_id=state.get("owner_id", cfg.get("user_id", "default")),
        session_id=state.get("session_id", ""),
        agent_id=agent_id,
        messages=[Message(
            role="user",
            content=build_multimodal_content(
                history_context + f"请完成以下任务：{task}",
                cfg.get("attachments", []),
            ),
        )],
        system_prompt=system_prompt,
        model=getattr(cfg["llm"], "model", ""),
        tools=tool_specs(agent_tools),
        tool_context={"session_id": state.get("session_id", "")},
        execution_policy=cfg.get("execution_policy", ExecutionPolicy()),
        orchestration_run_id=state.get("orchestration_run_id", ""),
    )
    runtime = AgentRuntime(
        cfg.get("runtime_store") or MemoryOperationStore(),
        model=MatrixModelAdapter(cfg["llm"]),
        tools=MatrixToolAdapter(
            agent_tools,
            session_id=state.get("session_id", ""),
            owner_id=state.get("owner_id", cfg.get("user_id", "default")),
            mode=cfg.get("execution_policy", ExecutionPolicy()).mode,
            allow_external_effects=cfg.get("execution_policy", ExecutionPolicy()).allow_external_effects,
        ),
        context=cfg.get("runtime_context") or MatrixContextAdapter(),
    )
    handle = runtime.start(request)
    _push_event(cfg, "progress", {"message": "独立 Runtime 正在执行 Agent 任务...", "operation_id": handle.operation_id})
    events = list(handle.events())
    result = handle.result()
    if cfg.get("execution_policy", ExecutionPolicy()).debug_trace:
        for trace_event in handle.debug_trace():
            _push_event(cfg, "debug_trace", {
                "operation_id": handle.operation_id,
                "event": trace_event,
            })
    for event in events:
        if event.event_type.value == "tool_start":
            _push_event(cfg, "tool_call", {
                "name": event.payload.get("name", ""),
                "args": {},
                "operation_id": handle.operation_id,
            })
        elif event.event_type.value == "tool_end":
            _push_event(cfg, "tool_result", {
                "name": event.payload.get("name", ""),
                "error": event.payload.get("error", ""),
                "operation_id": handle.operation_id,
            })

    if result.outcome.value == "suspended" and result.suspension is not None:
        action = {
            "approval_id": result.suspension.approval_id,
            "approval_set_id": result.suspension.approval_set_id,
            "approval_ids": list(result.suspension.approval_ids),
            "operation_id": handle.operation_id,
            "name": result.suspension.payload.get("tool_name", ""),
            "args": result.suspension.payload.get("arguments", {}),
            "risk": "runtime approval required",
        }
        _push_event(cfg, "confirm_required", {
            "actions": [action], "session_id": state.get("session_id", ""),
        })
        return {
            "needs_confirmation": True,
            "pending_actions": [action],
            "runtime_operation_ids": [{
                "step": step.get("step", current_step + 1),
                "operation_id": handle.operation_id,
            }],
        }

    runtime_tool_results = [
        _tool_result_dict(item) for item in result.tool_results
    ]
    agent_result = {
        "agent_id": agent_id,
        "task": task,
        "result": result.final_message,
        "tool_results": runtime_tool_results,
        "operation_id": handle.operation_id,
    }
    if result.outcome is not RunOutcome.COMPLETED:
        agent_result["error"] = result.error or result.outcome.value
    return {
        "agent_results": [agent_result],
        "tool_results": runtime_tool_results,
        "tool_call_count": len(runtime_tool_results),
        "completed_steps": [step.get("step", current_step + 1)],
        "completed_step_refs": [
            f"{state.get('plan_revision', 0)}:{step.get('step', current_step + 1)}"
        ],
    }


def runtime_confirm_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Pause LangGraph while the durable Runtime approval is pending."""
    cfg = _get_configurable(config)
    actions = state.get("pending_actions", [])
    decision = interrupt({"type": "confirm", "actions": actions})
    approval_decision = (
        "approve" if decision in (True, "approve", "confirmed") else "skip"
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for action in actions:
        if isinstance(action, dict) and action.get("operation_id"):
            grouped.setdefault(str(action["operation_id"]), []).append(action)
    if not grouped:
        for item in state.get("runtime_operation_ids", []):
            if isinstance(item, dict) and item.get("operation_id"):
                grouped.setdefault(str(item["operation_id"]), []).append(item)
    if not grouped:
        return {"error": "Runtime operation not found", "confirmed": True}

    agent_results: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    runtime_operation_ids: list[dict[str, Any]] = []
    completed_steps: list[int] = []
    completed_step_refs: list[str] = []
    for operation_id, operation_actions in grouped.items():
        operation = cfg["runtime_store"].load(cfg.get("user_id", "default"), operation_id)
        if operation is None:
            return {"error": "Runtime operation not found", "confirmed": True}
        agent_def = cfg["agent_registry"].get(operation.agent_id)
        if agent_def is None:
            return {"error": "Agent not found", "confirmed": True}
        agent_tools = cfg["agent_registry"].build_tool_registry(operation.agent_id, cfg["full_tools"])
        runtime = AgentRuntime(
            cfg["runtime_store"], model=MatrixModelAdapter(cfg["llm"]),
            tools=MatrixToolAdapter(
                agent_tools, session_id=operation.session_id, owner_id=operation.owner_id,
                mode=str(operation.state.get("execution_policy", {}).get("mode", "read_only")),
                allow_external_effects=bool(operation.state.get("execution_policy", {}).get("allow_external_effects", False)),
            ),
            context=cfg.get("runtime_context") or MatrixContextAdapter(),
        )
        first = operation_actions[0] if operation_actions else {}
        approval_id = str(first.get("approval_id", ""))
        approval_set_id = str(first.get("approval_set_id", ""))
        approval_ids = list(first.get("approval_ids", []))
        handle = runtime.resume(
            cfg.get("user_id", "default"), operation_id,
            __import__("matrix.runtime.domain.requests", fromlist=["ResumeInput"]).ResumeInput(
                kind="approval",
                decision=approval_decision,
                payload={
                    "approval_id": approval_id,
                    "approval_set_id": approval_set_id,
                    "approval_ids": approval_ids,
                    "expected_operation_version": operation.version,
                },
            ),
        )
        events = list(handle.events())
        result = handle.result()
        if cfg.get("execution_policy", ExecutionPolicy()).debug_trace:
            for trace_event in handle.debug_trace():
                _push_event(cfg, "debug_trace", {
                    "operation_id": handle.operation_id,
                    "event": trace_event,
                })
        step_number: int | None = None
        if operation.step_id:
            try:
                step_number = int(operation.step_id)
            except ValueError:
                step_number = None
        if step_number is None:
            plan = state.get("delegation_plan", [])
            current_step = state.get("current_step", 0)
            if 0 <= current_step < len(plan):
                raw_step = plan[current_step].get("step")
                if isinstance(raw_step, int):
                    step_number = raw_step
        task = ""
        if step_number is not None:
            for item in state.get("delegation_plan", []):
                if item.get("step") == step_number:
                    task = item.get("task", "")
                    break
        agent_result: dict[str, Any] = {
            "agent_id": operation.agent_id,
            "task": task,
            "result": result.final_message,
            "operation_id": operation_id,
        }
        if step_number is not None:
            agent_result["step"] = step_number
        if result.error:
            agent_result["error"] = result.error
        agent_results.append(agent_result)
        tool_results.extend(_tool_result_dict(item) for item in result.tool_results)
        runtime_operation_ids.append({"step": step_number, "operation_id": operation_id})
        if step_number is not None and result.outcome is RunOutcome.COMPLETED:
            completed_steps.append(step_number)
            completed_step_refs.append(
                f"{state.get('plan_revision', 0)}:{step_number}"
            )

    result_data: dict[str, Any] = {
        "confirmed": True,
        "needs_confirmation": False,
        "pending_actions": [],
        "agent_results": agent_results,
        "tool_results": tool_results,
        "runtime_operation_ids": runtime_operation_ids,
    }
    if completed_steps:
        result_data["completed_steps"] = completed_steps
        result_data["completed_step_refs"] = completed_step_refs
    return result_data


def runtime_delegate_node(state: AgentState, *, config: RunnableConfig) -> dict[str, Any]:
    """Execute a ready multi-agent DAG step through its own Runtime operation."""
    cfg = _get_configurable(config)
    plan = state.get("delegation_plan", [])
    current_step = state.get("current_step", 0)
    step = plan[current_step] if current_step < len(plan) else {}
    return run_dag_step(state, cfg, step)


def _tool_result_dict(result: ToolResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "arguments": {},
        "result": result.result,
        "error": result.error,
        "call_id": result.call_id,
    }
