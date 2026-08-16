"""Durable application-level workflow for Codex Deep Research.

This workflow is deliberately above Runtime Core.  It owns the fixed evidence
collection plan and the synthesis call, while Runtime provides the durable
operation/event lifecycle underneath it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
import time
import uuid
from typing import Any, Callable, Iterator

from ..domain.events import RuntimeEvent, RuntimeEventType
from ..domain.operations import OperationPhase, OperationState, StateTransition
from ..domain.results import RunOutcome
from ..ports.store import OperationStorePort


def _build_multimodal_content(
    text: str, attachments: list[dict[str, Any]],
) -> str | list[dict[str, Any]]:
    """Build provider-neutral synthesis content for image attachments."""
    image_attachments = [
        item for item in attachments
        if item.get("type") == "image" and item.get("base64")
    ]
    if not image_attachments:
        return text
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for item in image_attachments:
        blocks.append({
            "type": "image_url",
            "image_url": {
                "url": (
                    f"data:{item.get('mime_type', 'image/png')};"
                    f"base64,{item['base64']}"
                ),
            },
        })
    return blocks


@dataclass(frozen=True)
class DeepResearchEvent:
    runtime_event: RuntimeEvent
    ui_event: dict[str, Any]


@dataclass(frozen=True)
class DeepResearchResult:
    outcome: RunOutcome
    operation_id: str
    final_message: str = ""
    error: str = ""


class DeepResearchHandle:
    def __init__(self, workflow: "DeepResearchWorkflow", operation: OperationState,
                 question: str, name: str, research_date: str,
                 attachments: list[dict[str, Any]] | None = None) -> None:
        self.workflow = workflow
        self.operation = operation
        self.question = question
        self.name = name
        self.research_date = research_date
        self.attachments = attachments or []
        self._started = False
        self._events: list[DeepResearchEvent] = []
        self._result: DeepResearchResult | None = None

    @property
    def operation_id(self) -> str:
        return self.operation.operation_id

    def events(self) -> Iterator[DeepResearchEvent]:
        if self._started:
            yield from self._events
            return
        self._started = True
        yield from self._run()

    def result(self) -> DeepResearchResult:
        if not self._started:
            list(self.events())
        assert self._result is not None
        return self._result

    def _run(self) -> Iterator[DeepResearchEvent]:
        current = self.operation
        try:
            code, object_type = self._parse_request()
            if not code:
                yield from self._fail(current, "深度研究缺少标的代码")
                return

            current, event = self.workflow._commit(
                current, OperationPhase.PREPARING, RuntimeEventType.RUN_START,
                {"workflow": "deep_research", "code": code},
                {"type": "classify", "intent": "deep-research-prefetch"},
            )
            self._events.append(event)
            yield event

            evidence: list[dict[str, Any]] = []
            agent_tools = self.workflow.agent_tools
            calls = self._tool_calls(code)
            for tool_name, arguments in calls:
                if tool_name not in agent_tools.tool_names():
                    result = {"error": "tool unavailable"}
                    evidence.append({"tool": tool_name, "result": result})
                    current, event = self.workflow._commit(
                        current, OperationPhase.EXECUTING_TOOLS,
                        RuntimeEventType.TOOL_END,
                        {"tool": tool_name, "arguments": arguments, "result": result},
                        {"type": "tool_result", "name": tool_name, "error": "tool unavailable"},
                    )
                    self._events.append(event)
                    yield event
                    continue

                thinking = {
                    "type": "thinking",
                    "content": f"正在调用 {tool_name} 获取数据…",
                }
                current, event = self.workflow._commit(
                    current, OperationPhase.EXECUTING_TOOLS,
                    RuntimeEventType.TOOL_START,
                    {"tool": tool_name, "arguments": arguments},
                    thinking,
                )
                self._events.append(event)
                yield event
                call_event = {"type": "tool_call", "name": tool_name, "args": arguments}
                current, event = self.workflow._commit(
                    current, OperationPhase.EXECUTING_TOOLS,
                    RuntimeEventType.TOOL_UPDATE,
                    {"tool": tool_name, "arguments": arguments},
                    call_event,
                )
                self._events.append(event)
                yield event

                result, error = self._call_with_retry(tool_name, arguments)
                if error:
                    result = {"error": error}
                evidence.append({"tool": tool_name, "result": result})
                result_event = {
                    "type": "tool_result", "name": tool_name, "result": result,
                    "error": error, "preview": self.workflow.preview_json(result),
                }
                current, event = self.workflow._commit(
                    current, OperationPhase.EXECUTING_TOOLS,
                    RuntimeEventType.TOOL_END,
                    {"tool": tool_name, "arguments": arguments, "result": result,
                     "error": error},
                    result_event,
                )
                self._events.append(event)
                yield event

            info = next(
                (item["result"] for item in evidence
                 if item["tool"] == "personal_os.information_search"
                 and isinstance(item.get("result"), dict)),
                {},
            )
            sources = info.get("items", []) if isinstance(info, dict) else []
            official_url = self.workflow.select_latest_official_url(
                sources, self.research_date[:4] or time.strftime("%Y")
            )
            if official_url and "personal_os.web_fetch" in agent_tools.tool_names():
                tool_name = "personal_os.web_fetch"
                arguments = {"url": official_url}
                thinking = {"type": "thinking", "content": f"正在调用 {tool_name} 核验官方正文…"}
                current, event = self.workflow._commit(
                    current, OperationPhase.EXECUTING_TOOLS,
                    RuntimeEventType.TOOL_START,
                    {"tool": tool_name, "arguments": arguments}, thinking,
                )
                self._events.append(event)
                yield event
                current, event = self.workflow._commit(
                    current, OperationPhase.EXECUTING_TOOLS,
                    RuntimeEventType.TOOL_UPDATE,
                    {"tool": tool_name, "arguments": arguments},
                    {"type": "tool_call", "name": tool_name, "args": arguments},
                )
                self._events.append(event)
                yield event
                result, error = self._call_with_retry(tool_name, arguments)
                if error:
                    result = {"error": error}
                evidence.append({"tool": tool_name, "result": result})
                result_event = {
                    "type": "tool_result", "name": tool_name, "result": result,
                    "error": error, "preview": self.workflow.preview_json(result),
                }
                current, event = self.workflow._commit(
                    current, OperationPhase.EXECUTING_TOOLS,
                    RuntimeEventType.TOOL_END,
                    {"tool": tool_name, "arguments": arguments, "result": result,
                     "error": error}, result_event,
                )
                self._events.append(event)
                yield event

            evidence_text = json.dumps(evidence, ensure_ascii=False, default=str)[:60000]
            current, event = self.workflow._commit(
                current, OperationPhase.REQUESTING_MODEL,
                RuntimeEventType.MESSAGE_START,
                {"stage": "synthesis", "evidence_count": len(evidence)},
                {"type": "thinking", "content": "正在汇总研究结论…"},
            )
            self._events.append(event)
            yield event
            result = self.workflow.llm.complete_json(
                self.workflow.synthesis_prompt(evidence_text),
                [{
                    "role": "user",
                    "content": _build_multimodal_content(self.question, self.attachments),
                }],
            )
            if not isinstance(result, dict):
                raise ValueError("深度研究汇总结果不是 JSON 对象")
            result = self.workflow.normalize_result(
                result, evidence, code=code, name=self.name,
                object_type=object_type, research_date=self.research_date,
            )
            official_period = self.workflow.extract_official_report_period(evidence)
            if official_period and official_period > str(result.get("latest_report_period") or ""):
                result["latest_report_period"] = official_period
                metrics = result.get("metrics")
                if not isinstance(metrics, list):
                    metrics = []
                    result["metrics"] = metrics
                metrics.append({
                    "name": "官方公告最新报告期", "value": official_period,
                    "period": self.research_date, "source": "personal_os.web_fetch",
                })
            answer = json.dumps(result, ensure_ascii=False)
            current, event = self.workflow._commit(
                current, OperationPhase.REQUESTING_MODEL,
                RuntimeEventType.MESSAGE_DELTA,
                {"content": answer, "stage": "synthesis"},
                {"type": "token", "content": answer},
            )
            self._events.append(event)
            yield event
            current, event, final = self.workflow._finish(
                current, RunOutcome.COMPLETED, answer,
                {"outcome": RunOutcome.COMPLETED.value, "message": answer},
                {"type": "done"},
            )
            wrapped = DeepResearchEvent(event, {"type": "done"})
            self._events.append(wrapped)
            self._result = final
            yield wrapped
        except Exception as exc:
            yield from self._fail(current, str(exc) or exc.__class__.__name__)

    def _parse_request(self) -> tuple[str, str]:
        code_match = re.search(r"标的代码[:：]\s*([A-Za-z0-9_.-]+)", self.question)
        name_match = re.search(r"研究对象[:：]\s*(.+)", self.question)
        code = code_match.group(1).strip() if code_match else ""
        self.name = name_match.group(1).strip() if name_match else self.name
        object_type = "fund" if "对象类型：fund" in self.question else "stock"
        return code, object_type

    def _tool_calls(self, code: str) -> list[tuple[str, dict[str, Any]]]:
        year = self.research_date[:4] or time.strftime("%Y")
        return [
            ("personal_os.market_quote", {"code": code}),
            ("personal_os.financials", {"code": code, "periods": 8}),
            ("personal_os.profile", {"code": code}),
            ("personal_os.dividend", {"code": code, "years": 5}),
            ("personal_os.valuation", {"code": code}),
            ("personal_os.peers", {"code": code}),
            ("personal_os.research_context", {"code": code, "name": self.name}),
            ("personal_os.information_search", {
                "query": f"{self.name} {code} {year} 最新季报 二季度 业绩公告 年报 经营风险",
                "limit": 12,
            }),
        ]

    def _call_with_retry(self, name: str, arguments: dict[str, Any]) -> tuple[Any, str]:
        last_error = ""
        for _ in range(2):
            try:
                return self.workflow.agent_tools.call(
                    name, arguments, session_id=self.operation.session_id,
                ), ""
            except Exception as exc:
                last_error = str(exc) or exc.__class__.__name__
        return None, last_error

    def _fail(self, operation: OperationState, message: str) -> Iterator[DeepResearchEvent]:
        _, event, result = self.workflow._finish(
            operation, RunOutcome.FAILED, message,
            {"outcome": RunOutcome.FAILED.value, "error": message},
            {"type": "error", "message": message},
        )
        wrapped = DeepResearchEvent(event, {"type": "error", "message": message})
        self._events.append(wrapped)
        self._result = result
        yield wrapped


class DeepResearchWorkflow:
    def __init__(
        self,
        store: OperationStorePort,
        llm: Any,
        agent_tools: Any,
        *,
        normalize_result: Callable[..., dict[str, Any]],
        preview_json: Callable[[Any], str],
        select_latest_official_url: Callable[[list[Any], str], str],
        extract_official_report_period: Callable[[list[dict[str, Any]]], str],
    ) -> None:
        self.store = store
        self.llm = llm
        self.agent_tools = agent_tools
        self.normalize_result = normalize_result
        self.preview_json = preview_json
        self.select_latest_official_url = select_latest_official_url
        self.extract_official_report_period = extract_official_report_period

    def start(
        self, *, owner_id: str, session_id: str, question: str,
        name: str = "", research_date: str = "",
        attachments: list[dict[str, Any]] | None = None,
    ) -> DeepResearchHandle:
        operation = OperationState(
            operation_id=uuid.uuid4().hex,
            owner_id=owner_id,
            session_id=session_id,
            agent_id="deep-research",
            operation_scope="top_level",
            state={
                "workflow": "deep_research",
                "question": question,
                "research_date": research_date,
            },
        )
        self.store.create(operation)
        return DeepResearchHandle(
            self, operation, question, name, research_date, attachments,
        )

    def synthesis_prompt(self, evidence_text: str) -> str:
        return """你是筋斗云的深度投资研究员。
你只能使用下方 personal-os 工具证据，不得使用记忆补数字，不得编造来源。
请输出一个合法 JSON 对象，不要 Markdown，不要解释，不要代码围栏。
必须包含 schema_version=2、type=investment-research、status、object_type、
subject、research_date、data_date、latest_report_period、
information_completeness、decision、summary、highlights、thesis、antithesis、
risks、metrics、triggers、sources、tags。
如果证据不足，status 必须是 incomplete，information_completeness 必须是 low，
并在 risks 中说明缺口；不要伪装成 deep。
如果 personal_os.financials.metadata.stale 为 true，必须明确标记财报过期，
不得把该期间称为当前最新经营数据，并优先使用最新季报/业绩公告补充。
如果官方公告正文包含比 personal_os.financials 更新的报告期，必须使用官方公告
报告期作为 latest_report_period，并在指标或风险中区分“聚合财报最新期”和“官方公告最新期”。
metrics 必须是对象数组，每项至少包含 name 和 value；禁止把 Python map[...]、
整段工具返回或未拆解的对象放进 value。
sources 必须是对象数组，每项包含 title、url、date、source_type；url 和 date
只能填写证据中真实出现的内容，找不到就留空。

PERSONAL-OS EVIDENCE:
""" + evidence_text
    def _commit(
        self, operation: OperationState, phase: OperationPhase,
        event_type: RuntimeEventType, payload: dict[str, Any],
        ui_event: dict[str, Any],
    ) -> tuple[OperationState, DeepResearchEvent]:
        event = RuntimeEvent(
            event_id=uuid.uuid4().hex, owner_id=operation.owner_id,
            operation_id=operation.operation_id, session_id=operation.session_id,
            sequence=operation.last_event_sequence + 1,
            event_type=event_type, timestamp=time.time(), payload=payload,
        )
        state = dict(operation.state)
        state["last_stage"] = payload.get("stage") or payload.get("tool") or event_type.value
        next_operation = replace(
            operation, phase=phase, version=operation.version + 1,
            last_event_sequence=event.sequence, state=state,
        )
        self.store.commit(StateTransition(
            previous_version=operation.version, new_state=next_operation,
            events=(event,),
        ))
        return next_operation, DeepResearchEvent(event, ui_event)

    def _finish(
        self, operation: OperationState, outcome: RunOutcome, message: str,
        payload: dict[str, Any], ui_event: dict[str, Any],
    ) -> tuple[OperationState, RuntimeEvent, DeepResearchResult]:
        phase = {
            RunOutcome.COMPLETED: OperationPhase.COMPLETED,
            RunOutcome.FAILED: OperationPhase.FAILED,
        }[outcome]
        event_type = RuntimeEventType.RUN_END if outcome is RunOutcome.COMPLETED else RuntimeEventType.RUN_FAILED
        event = RuntimeEvent(
            event_id=uuid.uuid4().hex, owner_id=operation.owner_id,
            operation_id=operation.operation_id, session_id=operation.session_id,
            sequence=operation.last_event_sequence + 1,
            event_type=event_type, timestamp=time.time(), payload=payload,
        )
        state = dict(operation.state)
        state["final_message"] = message if outcome is RunOutcome.COMPLETED else ""
        next_operation = replace(
            operation, phase=phase, version=operation.version + 1,
            last_event_sequence=event.sequence, state=state,
        )
        self.store.commit(StateTransition(
            previous_version=operation.version, new_state=next_operation,
            events=(event,),
        ))
        return next_operation, event, DeepResearchResult(
            outcome=outcome, operation_id=operation.operation_id,
            final_message=message if outcome is RunOutcome.COMPLETED else "",
            error=message if outcome is not RunOutcome.COMPLETED else "",
        )
