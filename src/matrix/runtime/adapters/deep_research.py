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

            announcement_payload = next(
                (item["result"] for item in evidence
                 if item["tool"] == "personal_os.announcements"
                 and isinstance(item.get("result"), dict)),
                {},
            )
            announcement_sources = announcement_payload.get("items", []) if isinstance(announcement_payload, dict) else []
            info = next(
                (item["result"] for item in evidence
                 if item["tool"] == "personal_os.information_search"
                 and isinstance(item.get("result"), dict)),
                {},
            )
            info_sources = info.get("items", []) if isinstance(info, dict) else []
            sources = list(announcement_sources) + list(info_sources)
            financials = next(
                (item["result"] for item in evidence
                 if item["tool"] == "personal_os.financials"
                 and isinstance(item.get("result"), dict)),
                {},
            )
            financial_data = financials.get("data", {}) if isinstance(financials, dict) else {}
            reports = financial_data.get("reports", []) if isinstance(financial_data, dict) else []
            for report in reports if isinstance(reports, list) else []:
                if not isinstance(report, dict):
                    continue
                source = report.get("source")
                candidates = source.get("official_source_candidates", []) if isinstance(source, dict) else []
                for candidate in candidates if isinstance(candidates, list) else []:
                    if not isinstance(candidate, dict) or not candidate.get("source_url"):
                        continue
                    sources.append({
                        **candidate,
                        "url": candidate["source_url"],
                        "tier": candidate.get("source_level", ""),
                    })
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
                self.workflow.synthesis_prompt(evidence_text, evidence),
                [{
                    "role": "user",
                    "content": _build_multimodal_content(self.question, self.attachments),
                }],
            )
            if not isinstance(result, dict):
                raise ValueError("深度研究汇总结果不是 JSON 对象")
            official_period = self.workflow.extract_official_report_period(evidence)
            if official_period:
                for entry in evidence:
                    if entry.get("tool") != "personal_os.web_fetch":
                        continue
                    payload = entry.get("result")
                    if isinstance(payload, dict) and payload.get("content"):
                        payload["report_period"] = official_period
                        payload["verification_status"] = "verified"
            result = self.workflow.normalize_result(
                result, evidence, code=code, name=self.name,
                object_type=object_type, research_date=self.research_date,
            )
            issues = self.workflow.validate_result(
                result, evidence, research_date=self.research_date,
            )
            if issues:
                current, event = self.workflow._commit(
                    current, OperationPhase.REQUESTING_MODEL,
                    RuntimeEventType.MESSAGE_START,
                    {"stage": "validation_repair", "issues": issues},
                    {"type": "thinking", "content": "首轮研究结果未通过质量校验，正在修正…"},
                )
                self._events.append(event)
                yield event
                repaired = self.workflow.llm.complete_json(
                    self.workflow.repair_prompt(
                        evidence_text, result, issues, evidence,
                    ),
                    [{
                        "role": "user",
                        "content": _build_multimodal_content(self.question, self.attachments),
                    }],
                )
                if not isinstance(repaired, dict):
                    raise ValueError("研究结果修复响应不是 JSON 对象")
                result = self.workflow.normalize_result(
                    repaired, evidence, code=code, name=self.name,
                    object_type=object_type, research_date=self.research_date,
                )
                issues = self.workflow.validate_result(
                    result, evidence, research_date=self.research_date,
                )
                if issues:
                    raise ValueError("研究结果质量闸门未通过：" + "；".join(issues))
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
            ("personal_os.announcements", {"code": code, "limit": 10}),
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
        try:
            _, event, result = self.workflow._finish(
                operation, RunOutcome.FAILED, message,
                {"outcome": RunOutcome.FAILED.value, "error": message},
                {"type": "error", "message": message},
            )
        except Exception as persist_error:
            # A concurrent request may have advanced the CAS version while the
            # workflow was failing.  Never replace the useful user-facing
            # error with a second unhandled version-conflict exception.
            latest = self.workflow.store.load(operation.owner_id, operation.operation_id)
            base = latest or operation
            event = RuntimeEvent(
                event_id=uuid.uuid4().hex,
                owner_id=base.owner_id,
                operation_id=base.operation_id,
                session_id=base.session_id,
                sequence=base.last_event_sequence + 1,
                event_type=RuntimeEventType.RUN_FAILED,
                timestamp=time.time(),
                payload={
                    "outcome": RunOutcome.FAILED.value,
                    "error": message,
                    "settlement_error": str(persist_error),
                },
            )
            result = DeepResearchResult(
                outcome=RunOutcome.FAILED,
                operation_id=base.operation_id,
                error=message,
            )
        wrapped = DeepResearchEvent(event, {"type": "error", "message": message})
        self._events.append(wrapped)
        self._result = result
        yield wrapped


def _fact_check_summary(evidence: list[dict[str, Any]]) -> str:
    """Give synthesis a compact, deterministic view of key returned fields."""
    facts: list[dict[str, Any]] = []
    for entry in evidence:
        tool = entry.get("tool")
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        if tool == "personal_os.valuation":
            facts.append({
                "tool": tool,
                "pe_approx": result.get("pe_approx"),
                "pb_approx": result.get("pb_approx"),
                "price": result.get("price"),
            })
        elif tool == "personal_os.dividend":
            data = result.get("data")
            facts.append({"tool": tool, "has_data": bool(data), "years": result.get("years")})
        elif tool == "personal_os.peers":
            companies = result.get("companies")
            peer_rows = []
            if isinstance(companies, list):
                for company in companies:
                    if not isinstance(company, dict) or company.get("error"):
                        continue
                    peer_rows.append({
                        "code": company.get("code"),
                        "name": company.get("name"),
                        "price": company.get("price"),
                        "pe_approx": company.get("pe_approx"),
                        "pb_approx": company.get("pb_approx"),
                        "latest_report_period": company.get("latest_report_period"),
                    })
            facts.append({
                "tool": tool,
                "status": result.get("status", "configured"),
                "industry": result.get("industry"),
                "company_count": len(companies) if isinstance(companies, list) else 0,
                "successful_company_count": len(peer_rows),
                "companies": peer_rows,
            })
        elif tool == "personal_os.financials":
            data = result.get("data")
            metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
            facts.append({
                "tool": tool,
                "latest_report_period": metadata.get("latest_report_period"),
                "stale": metadata.get("stale"),
            })
        elif tool == "personal_os.web_fetch":
            facts.append({
                "tool": tool,
                "verified": result.get("verification_status") == "verified",
                "report_period": result.get("report_period"),
                "has_content": bool(result.get("content") or result.get("text")),
            })
    return json.dumps(facts, ensure_ascii=False)


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
        validate_result: Callable[..., list[str]] | None = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.agent_tools = agent_tools
        self.normalize_result = normalize_result
        self.preview_json = preview_json
        self.select_latest_official_url = select_latest_official_url
        self.extract_official_report_period = extract_official_report_period
        self.validate_result = validate_result or (lambda result, evidence, **kwargs: [])

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

    def synthesis_prompt(
        self, evidence_text: str, evidence: list[dict[str, Any]] | None = None,
    ) -> str:
        availability: list[dict[str, str]] = []
        for entry in evidence or []:
            tool = str(entry.get("tool") or "")
            if not tool:
                continue
            result = entry.get("result")
            unavailable = bool(entry.get("error")) or (
                isinstance(result, dict) and bool(result.get("error"))
            )
            availability.append({
                "tool": tool,
                "status": "unavailable" if unavailable else "available",
            })
        availability_text = json.dumps(availability, ensure_ascii=False)
        fact_summary = _fact_check_summary(evidence or [])
        return """你是筋斗云的深度投资研究员。
你只能使用下方 personal-os 工具证据，不得使用记忆补数字，不得编造来源。
工具可用性摘要中的 available 表示该工具已成功返回；禁止把 available 工具说成“未提供、缺失或不可用”。
如果工具已返回数据但本次没有使用，必须说“本次未采用该工具结果”，不得说“没有该数据”。
请输出一个合法 JSON 对象，不要 Markdown，不要解释，不要代码围栏。
必须包含 schema_version=2、type=investment-research、status、object_type、
subject、research_date、data_date、latest_report_period、
information_completeness、decision、summary、highlights、thesis、antithesis、
risks、metrics、triggers、sources、tags。
只有硬性证据缺失时才将 information_completeness 设为 low：例如没有可靠行情、
没有多期财务报告、财务数据过期，或没有官方正文/对账证据。核心数据齐全但缺少
PB、资本开支、资产负债表细节或同业组时，使用 standard，并在 risks/triggers 中
明确列出缺口；不要把可选字段缺失夸大为 low。只有行情、核心财务、估值、分红、
同业组和最新官方正文都可核验时，才允许使用 deep。
如果 personal_os.financials.data.metadata.stale 为 true 或 personal_os.financials.metadata.stale 为 true，必须说明结构化财报缓存的报告期；
如果已有更晚且通过核验的官方业绩公告，则使用该公告作为最新经营数据，不得因此把 information_completeness 降为 low，
但要在 risks 或 data_quality warning 中说明结构化财报尚未同步。没有更晚官方证据时，才将财报过期作为硬性缺口。
如果 financials.data.reports 存在，必须优先使用其中的多期数据分析趋势，不得只根据 latest 单期摘要推断趋势。
如果 personal_os.peers 返回 status=not_configured，必须说明同业组未配置，不得输出为系统故障，也不得自行猜测同业名单；这本身不是将核心研究降为 low 的理由。
如果 personal_os.peers 返回 status=ok 或 partial 且 companies 中有成功数据，必须使用其中真实的公司名称和估值字段进行至少一条同业比较；
不得说“同业数据未显示”或“无法比较”。对缺失的单项 PE/PB 要明确标记为不可计算，不得补猜。
如果官方公告正文包含比 personal_os.financials 更新的报告期，必须使用官方公告
报告期作为 latest_report_period，并在指标或风险中区分“聚合财报最新期”和“官方公告最新期”。
metrics 必须是对象数组，每项至少包含 name 和 value；禁止把 Python map[...]、
整段工具返回或未拆解的对象放进 value。
sources 必须是对象数组，每项包含 title、url、date、source_type；url 和 date
只能填写证据中真实出现的内容，找不到就留空。

TOOL AVAILABILITY:
        """ + availability_text + "\n\nDETERMINISTIC FACT CHECKS:\n" + fact_summary + "\n\nPERSONAL-OS EVIDENCE:\n" + evidence_text

    def repair_prompt(
        self, evidence_text: str, previous: dict[str, Any], issues: list[str],
        evidence: list[dict[str, Any]] | None = None,
    ) -> str:
        return (
            "你正在修复一份投资研究 JSON。只输出一个合法 JSON 对象，不要 Markdown。\n"
            "必须保留 schema_version=2、type=investment-research、subject、research_date、"
            "data_date、latest_report_period、decision、summary。以下数组绝对不能为空："
            "highlights、thesis、antithesis、risks、metrics、triggers、sources。"
            "highlights/thesis/antithesis/risks 使用字符串数组；metrics 使用包含 name/value 的对象数组；"
            "triggers 使用包含 type/condition 的对象数组；sources 使用包含 title/url/date/source_type 的对象数组。"
            "只能根据 PERSONAL-OS EVIDENCE 填写，缺少的事实必须明确写成待核实，不得编造数字。\n\n"
            "上一版结果未通过以下确定性校验：\n"
            + json.dumps(issues, ensure_ascii=False)
            + "\n上一版 JSON：\n"
            + json.dumps(previous, ensure_ascii=False, default=str)
            + "\n\nPERSONAL-OS EVIDENCE:\n"
            + evidence_text
        )


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
