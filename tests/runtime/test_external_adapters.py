from __future__ import annotations

import json

from matrix.runtime.adapters.deep_research import DeepResearchWorkflow
from matrix.runtime.adapters.external_agent import ExternalAgentAdapter
from matrix.runtime.domain.events import RuntimeEventType
from matrix.runtime.domain.operations import OperationPhase
from matrix.runtime.domain.results import RunOutcome
from matrix.runtime.testing.memory_store import MemoryOperationStore


class FakeExternalAgent:
    def __init__(self, events):
        self.events = list(events)
        self.calls = []

    def stream_agent(self, system, messages):
        self.calls.append((system, messages))
        yield from self.events


def test_external_agent_maps_events_without_owning_the_inner_loop():
    store = MemoryOperationStore()
    client = FakeExternalAgent([
        {"type": "progress", "message": "分析中"},
        {"type": "tool_call", "name": "执行本地命令"},
        {"type": "message", "content": "完成"},
        {"type": "done"},
    ])
    handle = ExternalAgentAdapter(store, client).start(
        owner_id="owner-a", session_id="session-a", agent_id="codex-direct",
        system="system", messages=[{"role": "user", "content": "hello"}],
    )

    events = list(handle.events())
    result = handle.result()
    operation = store.load("owner-a", handle.operation_id)

    assert result.outcome is RunOutcome.COMPLETED
    assert result.final_message == "完成"
    assert operation.phase is OperationPhase.COMPLETED
    assert [event.runtime_event.sequence for event in events] == list(range(1, len(events) + 1))
    assert events[0].runtime_event.event_type is RuntimeEventType.RUN_START
    assert any(event.runtime_event.event_type is RuntimeEventType.TOOL_UPDATE for event in events)
    assert events[-1].runtime_event.event_type is RuntimeEventType.RUN_END
    assert client.calls


def test_external_agent_cancel_is_durable_and_does_not_run_again():
    store = MemoryOperationStore()
    client = FakeExternalAgent([{"type": "message", "content": "never"}])
    handle = ExternalAgentAdapter(store, client).start(
        owner_id="owner-a", session_id="session-cancel", agent_id="codex-direct",
        system="system", messages=[],
    )
    handle.cancel("user stopped")

    events = list(handle.events())
    result = handle.result()

    assert result.outcome is RunOutcome.ABORTED
    assert result.error == "user stopped"
    assert client.calls == []
    assert events[-1].runtime_event.event_type is RuntimeEventType.RUN_ABORTED
    assert store.load("owner-a", handle.operation_id).phase is OperationPhase.ABORTED


def test_external_agent_partial_stream_is_recovered_after_restart():
    store = MemoryOperationStore()
    client = FakeExternalAgent([{"type": "message", "content": "partial"}])
    handle = ExternalAgentAdapter(store, client).start(
        owner_id="owner-a", session_id="session-restart", agent_id="codex-direct",
        system="system", messages=[],
    )
    stream = handle.events()
    next(stream)
    stream.close()

    recovered = store.recover_incomplete()

    assert len(recovered) == 1
    assert recovered[0].phase is OperationPhase.RECOVERY_REQUIRED
    assert recovered[0].state["runtime_recovery"]["reason"] == "process_restart"


class FakeResearchTools:
    def __init__(self):
        self.calls = []
        self.fail_once = {"personal_os.profile"}

    def tool_names(self):
        return {
            "personal_os.market_quote", "personal_os.financials",
            "personal_os.profile", "personal_os.dividend", "personal_os.valuation",
            "personal_os.peers", "personal_os.research_context",
            "personal_os.announcements", "personal_os.information_search",
        }

    def call(self, name, arguments, session_id=""):
        self.calls.append((name, arguments, session_id))
        if name in self.fail_once:
            self.fail_once.remove(name)
            raise RuntimeError("temporary tool failure")
        if name in {"personal_os.announcements", "personal_os.information_search"}:
            return {"items": []}
        return {"name": name, "ok": True}


class FakeResearchLLM:
    def __init__(self):
        self.prompts = []

    def complete_json(self, system, messages):
        self.prompts.append((system, messages))
        return {
            "status": "complete", "information_completeness": "high",
            "decision": "observe", "summary": "summary", "metrics": [],
            "sources": [], "tags": [],
        }


class SequentialResearchLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def complete_json(self, system, messages):
        self.prompts.append((system, messages))
        return self.responses.pop(0)


def _complete_research_result():
    return {
        "schema_version": 2,
        "type": "investment-research",
        "status": "complete",
        "object_type": "stock",
        "subject": {"code": "000001", "name": "测试公司"},
        "research_date": "2026-08-19",
        "data_date": "2026-08-19",
        "latest_report_period": "2026-06-30",
        "information_completeness": "standard",
        "decision": {"action": "observe"},
        "summary": "summary",
        "highlights": ["h1", "h2", "h3"],
        "thesis": ["t1", "t2", "t3"],
        "antithesis": ["a1", "a2"],
        "risks": ["r1", "r2", "r3"],
        "metrics": [
            {"name": "m1", "value": "1"},
            {"name": "m2", "value": "2"},
            {"name": "m3", "value": "3"},
            {"name": "m4", "value": "4"},
        ],
        "triggers": [
            {"type": "positive", "condition": "c1"},
            {"type": "negative", "condition": "c2"},
        ],
        "sources": [
            {
                "title": "official",
                "url": "https://example.com/official.pdf",
                "date": "2026-08-13",
                "source_type": "official",
            },
            {
                "title": "supplementary",
                "url": "https://example.com/market",
                "date": "2026-08-19",
                "source_type": "supplementary",
            },
        ],
        "tags": ["investment-research", "stock"],
    }


def _minimum_contract_issues(result, evidence, **kwargs):
    minimums = {
        "highlights": 3,
        "thesis": 3,
        "antithesis": 2,
        "risks": 3,
        "metrics": 4,
        "triggers": 2,
        "sources": 2,
    }
    return [
        f"{field} too short"
        for field, minimum in minimums.items()
        if not isinstance(result.get(field), list)
        or len(result[field]) < minimum
    ]


def test_deep_research_workflow_retries_evidence_and_persists_lifecycle():
    store = MemoryOperationStore()
    tools = FakeResearchTools()
    llm = FakeResearchLLM()
    workflow = DeepResearchWorkflow(
        store, llm, tools,
        normalize_result=lambda result, evidence, **kwargs: {
            **result, "normalized": True, "evidence_count": len(evidence),
        },
        preview_json=lambda value: json.dumps(value, ensure_ascii=False),
        select_latest_official_url=lambda sources, year: "",
        extract_official_report_period=lambda evidence: "",
    )
    handle = workflow.start(
        owner_id="owner-a", session_id="research-session",
        question="研究对象：测试公司\n标的代码：000001\n研究日期：2026-08-16",
    )

    events = list(handle.events())
    result = handle.result()
    operation = store.load("owner-a", handle.operation_id)

    assert result.outcome is RunOutcome.COMPLETED
    assert '"evidence_count": 9' in result.final_message
    assert operation.phase is OperationPhase.COMPLETED
    assert len(tools.calls) == 10  # profile failed once and was retried
    assert len(llm.prompts) == 1
    assert any(event.ui_event.get("type") == "tool_call" for event in events)
    assert any(event.runtime_event.event_type is RuntimeEventType.MESSAGE_DELTA for event in events)
    assert events[-1].runtime_event.event_type is RuntimeEventType.RUN_END


def test_deep_research_workflow_preserves_image_attachment_for_synthesis():
    store = MemoryOperationStore()
    tools = FakeResearchTools()
    llm = FakeResearchLLM()
    workflow = DeepResearchWorkflow(
        store, llm, tools,
        normalize_result=lambda result, evidence, **kwargs: result,
        preview_json=lambda value: json.dumps(value, ensure_ascii=False),
        select_latest_official_url=lambda sources, year: "",
        extract_official_report_period=lambda evidence: "",
    )

    handle = workflow.start(
        owner_id="owner-a",
        session_id="research-image",
        question="研究对象：测试公司\n标的代码：000001\n研究日期：2026-08-16",
        attachments=[{
            "type": "image",
            "mime_type": "image/png",
            "base64": "aW1hZ2U=",
        }],
    )

    result = handle.result()

    assert result.outcome is RunOutcome.COMPLETED
    content = llm.prompts[0][1][0]["content"]
    assert isinstance(content, list)
    assert content[0] == {
        "type": "text",
        "text": "研究对象：测试公司\n标的代码：000001\n研究日期：2026-08-16",
    }
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,aW1hZ2U="


def test_deep_research_repairs_once_and_records_safe_diagnostics():
    store = MemoryOperationStore()
    tools = FakeResearchTools()
    llm = SequentialResearchLLM([
        {"summary": "do not persist this prose", "highlights": []},
        _complete_research_result(),
    ])
    workflow = DeepResearchWorkflow(
        store, llm, tools,
        normalize_result=lambda result, evidence, **kwargs: result,
        preview_json=lambda value: json.dumps(value, ensure_ascii=False),
        select_latest_official_url=lambda sources, year: "",
        extract_official_report_period=lambda evidence: "",
        validate_result=_minimum_contract_issues,
    )

    handle = workflow.start(
        owner_id="owner-a",
        session_id="research-repair-success",
        question="研究对象：测试公司\n标的代码：000001\n研究日期：2026-08-19",
        research_date="2026-08-19",
    )
    result = handle.result()
    events = store.event_list(handle.operation_id)

    assert result.outcome is RunOutcome.COMPLETED
    assert len(llm.prompts) == 2
    assert "完整结构" in llm.prompts[1][0]
    repair_event = next(
        event for event in events
        if event.payload.get("stage") == "validation_repair"
    )
    assert repair_event.payload["result_shape"]["fields"]["highlights"]["count"] == 0
    assert "do not persist this prose" not in json.dumps(
        repair_event.payload, ensure_ascii=False,
    )
    assert any(
        event.payload.get("stage") == "validation_repair_passed"
        for event in events
    )


def test_deep_research_fails_after_single_unsuccessful_repair():
    store = MemoryOperationStore()
    tools = FakeResearchTools()
    llm = SequentialResearchLLM([
        {"summary": "first", "highlights": []},
        {"summary": "second", "highlights": []},
    ])
    workflow = DeepResearchWorkflow(
        store, llm, tools,
        normalize_result=lambda result, evidence, **kwargs: result,
        preview_json=lambda value: json.dumps(value, ensure_ascii=False),
        select_latest_official_url=lambda sources, year: "",
        extract_official_report_period=lambda evidence: "",
        validate_result=_minimum_contract_issues,
    )

    handle = workflow.start(
        owner_id="owner-a",
        session_id="research-repair-failure",
        question="研究对象：测试公司\n标的代码：000001\n研究日期：2026-08-19",
        research_date="2026-08-19",
    )
    result = handle.result()
    events = store.event_list(handle.operation_id)

    assert result.outcome is RunOutcome.FAILED
    assert len(llm.prompts) == 2
    assert "已修复 1 次" in result.error
    failed_event = next(
        event for event in events
        if event.payload.get("stage") == "validation_failed"
    )
    assert failed_event.payload["attempt"] == 1
    assert failed_event.payload["result_shape"]["fields"]["sources"]["count"] == 0
    assert events[-1].event_type is RuntimeEventType.RUN_FAILED


def test_deep_research_never_applies_future_period_after_validation():
    store = MemoryOperationStore()
    tools = FakeResearchTools()
    llm = SequentialResearchLLM([_complete_research_result()])
    workflow = DeepResearchWorkflow(
        store, llm, tools,
        normalize_result=lambda result, evidence, **kwargs: result,
        preview_json=lambda value: json.dumps(value, ensure_ascii=False),
        select_latest_official_url=lambda sources, year: "",
        extract_official_report_period=lambda evidence: "2026-12-31",
        validate_result=_minimum_contract_issues,
    )

    handle = workflow.start(
        owner_id="owner-a",
        session_id="research-future-period",
        question="研究对象：测试公司\n标的代码：000001\n研究日期：2026-08-19",
        research_date="2026-08-19",
    )
    result = handle.result()
    payload = json.loads(result.final_message)

    assert result.outcome is RunOutcome.COMPLETED
    assert payload["latest_report_period"] == "2026-06-30"
