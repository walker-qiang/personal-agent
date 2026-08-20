"""Durable application-level workflow for Codex Deep Research.

This workflow is deliberately above Runtime Core.  It owns the fixed evidence
collection plan and the synthesis call, while Runtime provides the durable
operation/event lifecycle underneath it.
"""

from __future__ import annotations

import copy
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


RESEARCH_MINIMUM_ITEMS = {
    "highlights": 3,
    "thesis": 3,
    "antithesis": 2,
    "risks": 3,
    "metrics": 4,
    "triggers": 2,
    "sources": 2,
}

DEEP_RESEARCH_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version",
        "type",
        "status",
        "object_type",
        "subject",
        "research_date",
        "data_date",
        "latest_report_period",
        "information_completeness",
        "decision",
        "summary",
        "highlights",
        "thesis",
        "antithesis",
        "risks",
        "metrics",
        "triggers",
        "sources",
        "tags",
    ],
    "properties": {
        "schema_version": {"type": "integer", "enum": [2]},
        "type": {"type": "string", "enum": ["investment-research"]},
        "status": {"type": "string", "minLength": 1},
        "object_type": {"type": "string", "minLength": 1},
        "subject": {
            "type": "object",
            "required": ["code", "name"],
            "properties": {
                "code": {"type": "string", "minLength": 1},
                "name": {"type": "string", "minLength": 1},
            },
        },
        "research_date": {"type": "string", "minLength": 10},
        "data_date": {"type": "string", "minLength": 10},
        "latest_report_period": {"type": "string", "minLength": 10},
        "information_completeness": {"type": "string", "minLength": 1},
        "decision": {"type": "object"},
        "summary": {"type": "string", "minLength": 1},
        "highlights": {
            "type": "array",
            "minItems": RESEARCH_MINIMUM_ITEMS["highlights"],
            "items": {"type": "string", "minLength": 1},
        },
        "thesis": {
            "type": "array",
            "minItems": RESEARCH_MINIMUM_ITEMS["thesis"],
            "items": {"type": "string", "minLength": 1},
        },
        "antithesis": {
            "type": "array",
            "minItems": RESEARCH_MINIMUM_ITEMS["antithesis"],
            "items": {"type": "string", "minLength": 1},
        },
        "risks": {
            "type": "array",
            "minItems": RESEARCH_MINIMUM_ITEMS["risks"],
            "items": {"type": "string", "minLength": 1},
        },
        "metrics": {
            "type": "array",
            "minItems": RESEARCH_MINIMUM_ITEMS["metrics"],
            "items": {
                "type": "object",
                "required": ["name", "value"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "value": {},
                },
            },
        },
        "triggers": {
            "type": "array",
            "minItems": RESEARCH_MINIMUM_ITEMS["triggers"],
            "items": {
                "type": "object",
                "required": ["type", "condition"],
                "properties": {
                    "type": {"type": "string", "minLength": 1},
                    "condition": {"type": "string", "minLength": 1},
                },
            },
        },
        "sources": {
            "type": "array",
            "minItems": RESEARCH_MINIMUM_ITEMS["sources"],
            "items": {
                "type": "object",
                "required": ["title", "url", "date", "source_type"],
                "properties": {
                    "title": {"type": "string", "minLength": 1},
                    "url": {
                        "type": "string",
                        "pattern": r"^https?://",
                    },
                    "date": {"type": "string"},
                    "source_type": {"type": "string", "minLength": 1},
                },
            },
        },
        "tags": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
}


def _result_shape_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Return debug-safe result shape without persisting report prose."""
    list_fields: dict[str, dict[str, Any]] = {}
    for field in RESEARCH_MINIMUM_ITEMS:
        value = result.get(field)
        list_fields[field] = {
            "type": type(value).__name__,
            "count": len(value) if isinstance(value, list) else 0,
        }
    sources = result.get("sources")
    source_items = sources if isinstance(sources, list) else []
    usable_sources = [
        item for item in source_items
        if isinstance(item, dict)
        and str(item.get("url") or "").startswith(("http://", "https://"))
    ]
    official_sources = [
        item for item in usable_sources
        if str(item.get("source_type") or "").lower()
        in {"official", "web_fetch", "cninfo", "hkexnews"}
    ]
    return {
        "schema_version": result.get("schema_version"),
        "type": result.get("type"),
        "status": result.get("status"),
        "research_date": result.get("research_date"),
        "data_date": result.get("data_date"),
        "latest_report_period": result.get("latest_report_period"),
        "information_completeness": result.get("information_completeness"),
        "fields": list_fields,
        "usable_source_count": len(usable_sources),
        "official_source_count": len(official_sources),
    }


def _bounded_report_period(value: Any, research_date: str) -> str:
    """Accept an inferred report period only when it is real and not future."""
    import datetime as _datetime

    text = str(value or "").strip()[:10]
    upper_text = str(research_date or "").strip()[:10]
    try:
        period = _datetime.date.fromisoformat(text)
        upper = _datetime.date.fromisoformat(upper_text)
    except (TypeError, ValueError):
        return ""
    if period.year < 2000 or period > upper:
        return ""
    return period.isoformat()


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


def _report_verification_status(report: dict[str, Any]) -> str:
    source = report.get("source")
    source = source if isinstance(source, dict) else {}
    status = str(source.get("verification_status") or "").strip().lower()
    reconciliation = source.get("reconciliation")
    if isinstance(reconciliation, dict):
        status = str(reconciliation.get("status") or status).strip().lower()
    return status or "unverified"


def _redact_unverified_report(report: dict[str, Any]) -> dict[str, Any]:
    """Keep report identity while hiding unverified financial values from LLMs."""
    keep_fields = {
        "period_start", "period_end", "period_type", "report_type",
        "reported_at", "currency", "unit", "data_type",
    }
    redacted = {
        key: copy.deepcopy(value)
        for key, value in report.items()
        if key in keep_fields
    }
    source = report.get("source")
    if isinstance(source, dict):
        redacted["source"] = {
            key: copy.deepcopy(source[key])
            for key in (
                "verification_status", "reconciliation", "source_url",
                "source_name", "statement_coverage",
            )
            if key in source
        }
    redacted["data_type"] = "reported_unverified_metadata_only"
    return redacted


def _model_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove unverified financial values from the model-visible evidence view."""
    visible = copy.deepcopy(evidence)
    for entry in visible:
        tool = entry.get("tool")
        result = entry.get("result")
        if not isinstance(result, dict):
            continue

        if tool == "personal_os.financials":
            data = result.get("data")
            if not isinstance(data, dict):
                continue
            reports = data.get("reports")
            if isinstance(reports, list):
                data["reports"] = [
                    report
                    if not isinstance(report, dict)
                    or _report_verification_status(report) in {"verified", "reconciled"}
                    else _redact_unverified_report(report)
                    for report in reports
                ]
            latest = data.get("latest")
            if isinstance(latest, dict):
                latest_period = str(
                    latest.get("period")
                    or latest.get("period_end")
                    or ""
                ).strip()[:10]
                matching = next(
                    (
                        report for report in reports
                        if isinstance(report, dict)
                        and str(report.get("period_end") or "").strip()[:10]
                        == latest_period
                    ),
                    None,
                ) if isinstance(reports, list) else None
                if matching and _report_verification_status(matching) not in {
                    "verified", "reconciled",
                }:
                    data["latest"] = _redact_unverified_report(matching)
            metadata = data.get("metadata")
            if isinstance(metadata, dict):
                metadata["model_note"] = (
                    "未官方对账报告仅保留期间和状态，禁止作为确定性事实；"
                    "请使用 verified/reconciled 报告或明确写数据缺口。"
                )

        elif tool == "personal_os.research_context":
            observed = result.get("observed_facts")
            if not isinstance(observed, dict):
                continue
            reports = observed.get("reports")
            if isinstance(reports, list):
                observed["reports"] = [
                    report
                    if not isinstance(report, dict)
                    or _report_verification_status(report) in {"verified", "reconciled"}
                    else _redact_unverified_report(report)
                    for report in reports
                ]
    return visible


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

            model_evidence = _model_evidence(evidence)
            evidence_text = json.dumps(
                model_evidence, ensure_ascii=False, default=str,
            )[:60000]
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
            official_period = _bounded_report_period(
                self.workflow.extract_official_report_period(evidence),
                self.research_date,
            )
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
            result = self.workflow.sanitize_result(
                result, evidence, research_date=self.research_date,
            )
            issues = self.workflow.validate_result(
                result, evidence, research_date=self.research_date,
            )
            if issues:
                initial_issues = list(issues)
                current, event = self.workflow._commit(
                    current, OperationPhase.REQUESTING_MODEL,
                    RuntimeEventType.MESSAGE_START,
                    {
                        "stage": "validation_repair",
                        "attempt": 1,
                        "issues": initial_issues,
                        "result_shape": _result_shape_summary(result),
                    },
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
                    schema=DEEP_RESEARCH_RESULT_SCHEMA,
                    temperature=0,
                )
                if not isinstance(repaired, dict):
                    raise ValueError("研究结果修复响应不是 JSON 对象")
                result = self.workflow.normalize_result(
                    repaired, evidence, code=code, name=self.name,
                    object_type=object_type, research_date=self.research_date,
                )
                result = self.workflow.sanitize_result(
                    result, evidence, research_date=self.research_date,
                )
                issues = self.workflow.validate_result(
                    result, evidence, research_date=self.research_date,
                )
                if issues:
                    current, event = self.workflow._commit(
                        current, OperationPhase.REQUESTING_MODEL,
                        RuntimeEventType.MESSAGE_END,
                        {
                            "stage": "validation_failed",
                            "attempt": 1,
                            "initial_issues": initial_issues,
                            "issues": issues,
                            "result_shape": _result_shape_summary(result),
                        },
                        {
                            "type": "thinking",
                            "content": "修正结果仍未通过质量校验。",
                        },
                    )
                    self._events.append(event)
                    yield event
                    raise ValueError(
                        "研究结果质量闸门未通过（已修复 1 次）："
                        + "；".join(issues)
                    )
                current, event = self.workflow._commit(
                    current, OperationPhase.REQUESTING_MODEL,
                    RuntimeEventType.MESSAGE_END,
                    {
                        "stage": "validation_repair_passed",
                        "attempt": 1,
                        "initial_issues": initial_issues,
                        "issues": [],
                        "result_shape": _result_shape_summary(result),
                    },
                    {"type": "thinking", "content": "修正结果已通过质量校验。"},
                )
                self._events.append(event)
                yield event
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
            report_verification = []
            reports = data.get("reports", []) if isinstance(data, dict) else []
            for report in reports if isinstance(reports, list) else []:
                if not isinstance(report, dict):
                    continue
                source = report.get("source")
                source = source if isinstance(source, dict) else {}
                status = str(source.get("verification_status") or "unverified")
                reconciliation = source.get("reconciliation")
                if isinstance(reconciliation, dict):
                    status = str(reconciliation.get("status") or status)
                report_verification.append({
                    "period": report.get("period_end"),
                    "period_type": report.get("period_type"),
                    "verification_status": status,
                })
            facts.append({
                "tool": tool,
                "latest_report_period": metadata.get("latest_report_period"),
                "latest_period_type": metadata.get("latest_period_type"),
                "stale": metadata.get("stale"),
                "report_verification": report_verification,
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
        sanitize_result: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.agent_tools = agent_tools
        self.normalize_result = normalize_result
        self.preview_json = preview_json
        self.select_latest_official_url = select_latest_official_url
        self.extract_official_report_period = extract_official_report_period
        self.validate_result = validate_result or (lambda result, evidence, **kwargs: [])
        self.sanitize_result = sanitize_result or (lambda result, evidence, **kwargs: result)

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
financials.data.reports[].period_type 是期间口径的权威字段：Q2 表示第二季度单季，
H1 表示上半年累计。不得仅根据 period_end=06-30 把 Q2 写成 H1、上半年或中期；
指标名称、摘要和关键事实必须保持同一期间口径。
financials.data.reports[].source.verification_status 或 source.reconciliation.status
只有 verified/reconciled 才能作为 reported 确定性指标。unverified 报告期不得进入
metrics；正文如需提及，必须明确标注“第三方结构化数据，尚未官方对账/待核实”，
不得据此给出确定性同比或投资结论。
valuation 返回 estimated=true 时，PE/PB 必须标记为 Calculated/Estimated，
并在 metric name 或 source 中明确写“估算”，不得冒充官方披露指标。
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
        minimums = json.dumps(RESEARCH_MINIMUM_ITEMS, ensure_ascii=False)
        template = {
            "schema_version": 2,
            "type": "investment-research",
            "status": "complete",
            "object_type": "stock",
            "subject": {"code": "从上一版保留", "name": "从上一版保留"},
            "research_date": "从上一版保留",
            "data_date": "证据数据日期，YYYY-MM-DD",
            "latest_report_period": "证据中的最新报告期，YYYY-MM-DD",
            "information_completeness": "standard 或 deep",
            "decision": {
                "quality": "基于证据填写",
                "valuation": "基于证据填写",
                "portfolio_role": "基于证据填写",
                "action": "基于证据填写",
                "confidence": "基于证据填写",
            },
            "summary": "完整研究摘要",
            "highlights": ["关键事实 1", "关键事实 2", "关键事实 3"],
            "thesis": ["正向逻辑 1", "正向逻辑 2", "正向逻辑 3"],
            "antithesis": ["反方逻辑 1", "反方逻辑 2"],
            "risks": ["风险 1", "风险 2", "风险 3"],
            "metrics": [
                {"name": "指标 1", "value": "证据值", "period": "报告期", "source": "工具名"},
                {"name": "指标 2", "value": "证据值", "period": "报告期", "source": "工具名"},
                {"name": "指标 3", "value": "证据值", "period": "报告期", "source": "工具名"},
                {"name": "指标 4", "value": "证据值", "period": "报告期", "source": "工具名"},
            ],
            "triggers": [
                {"type": "positive", "condition": "可验证条件"},
                {"type": "negative", "condition": "可验证条件"},
            ],
            "sources": [
                {
                    "title": "官方来源标题",
                    "url": "证据中的真实 URL",
                    "date": "证据中的真实日期",
                    "source_type": "official",
                },
                {
                    "title": "补充来源标题",
                    "url": "证据中的真实 URL",
                    "date": "证据中的真实日期",
                    "source_type": "supplementary",
                },
            ],
            "tags": ["investment-research", "stock"],
        }
        return (
            "你正在完整重写一份未通过质量校验的投资研究 JSON。"
            "只输出一个合法 JSON 对象，不要 Markdown，不要解释。\n"
            "这是一次修复调用，不是摘要调用。即使上一版缺少字段，也必须补齐完整结构；"
            "任何数组都不允许省略、设为空或用 null 代替。输出前请逐项检查每个数组的条目数。"
            "缺少事实时，只能根据 PERSONAL-OS EVIDENCE 写出具体的“待核实事项”，"
            "不能用“暂无”“待补充”“N/A”等空泛占位语句。\n"
            "必须保留 schema_version=2、type=investment-research、subject、research_date、"
            "data_date、latest_report_period、decision、summary。以下数组绝对不能为空："
            "highlights、thesis、antithesis、risks、metrics、triggers、sources。"
            "highlights/thesis/antithesis/risks 使用字符串数组；metrics 使用包含 name/value 的对象数组；"
            "triggers 使用包含 type/condition 的对象数组；sources 使用包含 title/url/date/source_type 的对象数组。"
            "禁止复制上一版中的空数组，禁止使用占位符、map[...]、整段工具对象或虚构 URL。"
            "只能根据 PERSONAL-OS EVIDENCE 填写；缺少的事实要写成具体的待核实事项，不得编造数字。"
            "financials 报告只有 source.verification_status/reconciliation.status 为 verified 或 reconciled"
            "才能进入 metrics。unverified 报告必须从 metrics 删除；正文如提及必须明确标注"
            "第三方结构化数据尚未官方对账。estimated=true 的 PE/PB 必须在 name 或 source 标注估算。"
            "如果校验问题指出某个 highlights、summary、thesis、antithesis 或 risks 引用了"
            "尚未官方对账的报告期，必须删除或改写整条断言，改用已对账报告期的证据或明确的数据缺口；"
            "不能只在原句末尾添加“待核实”后继续保留确定性同比、增长或经营判断。"
            "如果校验问题指出 metrics 使用未对账报告期，必须删除该 metric 并用已对账证据补足，"
            "不能只修改 name、source 或 period 的文字来绕过校验。\n"
            "修复后每个数组仍必须满足最小条目数；如果正向证据不足，"
            "thesis 必须用已对账事实支持的条件性判断或明确的待验证路径补足至少 3 条，"
            "不得引用被隐藏的未对账期间，也不得用空泛口号充数。\n"
            "每个数组必须满足以下最小条目数：\n"
            + minimums
            + "\n机器校验还会强制执行以下规则：所有必填字段必须出现；字符串数组条目不得为空；"
            "metrics 每项必须有 name/value；triggers 每项必须有 type/condition；"
            "sources 每项必须有 title/url/date/source_type，url 必须以 http:// 或 https:// 开头。"
            + "\n输出必须遵循这个完整结构，示例文字必须替换为真实证据或具体待核实事项：\n"
            + json.dumps(template, ensure_ascii=False)
            + "\n\n"
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
