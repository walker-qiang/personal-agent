#!/usr/bin/env python3
"""Exercise the stock v3 application protocol with a local fake LLM."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from matrix.llm import DeepSeekClient
from matrix.server.routes import stock_research as route
import uvicorn


class FakeLLM:
    model = "mock-budgeted"

    def __init__(self):
        self.calls = []
        self.metric_plan_calls = []
        self.repair = False
        self.omit_usage = False

    def _usage(self):
        return {} if self.omit_usage else {"input_tokens": 150, "output_tokens": 20}

    def complete_json_budgeted(self, system, messages, maximum, schema=None):
        self.calls.append((system, messages, maximum))
        if self.repair and len(self.calls) == 1:
            return {"episode": {}}, self._usage()
        return {"episode": {"summary": "", "pending_questions": []},
                "thesis_patch": [], "evidence_candidates": []}, self._usage()


class ScopedLLM(FakeLLM):
    """Offline malformed checklist replay; no external model client."""

    def __init__(self, mode: str, *, repairable: bool = True):
        super().__init__()
        self.mode = mode
        self.repairable = repairable

    def complete_json_budgeted(self, system, messages, maximum, schema=None):
        self.calls.append((system, messages, maximum))
        payload = json.loads(messages[0]["content"])
        required = payload["required_claim_ids"]
        assert required == ["claim-1", "claim-2"]
        checklist = schema["properties"]["episode"]["properties"]["claim_checks"]
        assert checklist["minItems"] == checklist["maxItems"] == 2
        assert checklist["items"]["properties"]["claim_id"]["enum"] == required
        assert '"claim_checks":[' in system
        checks = [
            {"claim_id": claim_id, "status": "unresolved",
             "finding": "现有资料不足以核验此判断。", "evidence_refs": []}
            for claim_id in required
        ]
        repairing = "validation_errors" in payload
        if repairing:
            assert "Correct the supplied validation_errors" in system
            assert payload["validation_errors"]
            if self.mode in {"missing", "extra"}:
                assert "claim_checks must cover" in " ".join(payload["validation_errors"])
                assert ("missing=['claim-2']" if self.mode == "missing" else
                        "unexpected=['risk-outside-scope']") in " ".join(payload["validation_errors"])
        if not repairing or not self.repairable:
            if self.mode == "missing":
                checks.pop()
            elif self.mode == "extra":
                checks.append({**checks[0], "claim_id": "risk-outside-scope"})
            elif self.mode == "duplicate":
                checks.append(dict(checks[0]))
            elif self.mode == "forged":
                checks[0] = {**checks[0], "status": "supported", "evidence_refs": ["forged:fact"]}
        return {
            "episode": {
                "summary": "按本次指定范围核对；未获支持的判断保持待核验。",
                "pending_questions": ["核对下一报告期的现金支出"],
                "claim_checks": checks,
                "next_step": {
                    "question": "核对下一报告期的资本开支",
                    "required_fields": ["资本开支", "报告期"],
                    "source_hint": "官方报告现金流量表",
                    "completion_criteria": "取得可定位的同口径披露",
                },
            },
            "thesis_patch": [], "evidence_candidates": [],
        }, self._usage()


class IntegrationLLM(FakeLLM):
    def complete_json_budgeted(self, system, messages, maximum, schema=None):
        payload = json.loads(messages[0]["content"])
        if (isinstance(schema, dict)
                and "metric_ids" in schema.get("properties", {})):
            self.metric_plan_calls.append(payload)
            return {
                "metric_ids": [
                    "revenue_yoy_pct", "net_profit_yoy_pct",
                    "operating_cash_flow_to_net_profit",
                ],
            }, self._usage()
        if "request" in payload:
            payload = payload["request"]
        plan = payload.get("plan", {})
        context = payload.get("context", {})
        documents = payload.get("documents", [])
        if not plan:
            facts_only = "portfolio_context" not in payload
            plan = {
                "plan_id": payload.get("question", "facts-only"),
                "question": payload["question"],
                "facts_only": facts_only,
                "calculated_metrics": payload.get("calculated_metrics", []),
            }
            context = {
                key: value for key, value in payload.items()
                if key in {"question", "subject", "thesis", "portfolio_context",
                           "product_context", "pending_questions", "calculated_metrics"}
            }
            documents = payload.get("documents", [])
        self.calls.append(plan["plan_id"])
        if "required_claim_ids" in payload:
            checklist = schema["properties"]["episode"]["properties"]["claim_checks"]
            assert checklist["minItems"] == checklist["maxItems"] == len(payload["required_claim_ids"])
            if payload["required_claim_ids"]:
                assert checklist["items"]["properties"]["claim_id"]["enum"] == payload["required_claim_ids"]
        if not documents:
            return {"episode": {"summary": "", "pending_questions": []},
                    "thesis_patch": [], "evidence_candidates": []}, {
                        "input_tokens": 150, "output_tokens": 20}
        for container in (plan, context):
            metrics = container.get("calculated_metrics", [])
            assert all(not {
                "source_id", "sources", "url", "raw_response_hash",
            }.intersection(metric) for metric in metrics), metrics
        document = documents[0]
        if plan.get("facts_only"):
            assert set(context) == {"question", "subject", "calculated_metrics", "pending_questions"}
            assert "thesis" not in context and not document.get("excerpt")
            assert "Restricted original" not in messages[0]["content"]
            assert context["calculated_metrics"] == plan["calculated_metrics"]
            metric = next(item for item in plan["calculated_metrics"]
                          if item["name"] == "net_profit_yoy_pct")
            if plan["question"] == "numeric-replay" and getattr(self, "replay_result", None) is not None:
                return self.replay_result, {"input_tokens": 150, "output_tokens": 20}
            return {
                "episode": {
                    "summary": "Cash conversion requires further review.",
                    "pending_questions": ["What explains the change?"],
                    "numeric_claims": [{"metric_id": metric["id"], "value":
                                       "13.01" if plan["question"] == "numeric-bad" else metric["value"]}],
                },
                "thesis_patch": [], "evidence_candidates": [],
            }, {"input_tokens": 150, "output_tokens": 20}
        if "product_context" in context:
            assert context["product_context"]["verification"] == "unverified_registry_context"
            assert context["portfolio_context"] == {"version": 1, "transactions": [], "allocation": []}
            assert "product_quality" in context["thesis"]["decision"]
            assert "company_quality" not in context["thesis"]["decision"]
            segment = payload["evidence_segments"][0]
            return {
                "episode": {
                    "summary": "产品资料已核验，跟踪表现仍待补充。",
                    "pending_questions": ["跟踪误差是否稳定？"],
                    "claim_checks": [{
                        "claim_id": context["thesis"]["claims"][0]["claim_id"], "status": "supported",
                        "finding": "所选判断已有可定位的来源支持。",
                        "evidence_refs": [document["source_id"] + ":fund-fact"],
                    }],
                    "next_step": {
                        "question": "核对最近报告期的跟踪误差口径",
                        "required_fields": ["跟踪误差", "报告期", "年化口径"],
                        "source_hint": "基金定期报告的跟踪情况章节",
                        "completion_criteria": "取得同口径披露或明确记录未披露项",
                    },
                },
                "thesis_patch": [{
                    "entity": "decision", "id": "", "field": "product_quality",
                    "before": context["thesis"]["decision"]["product_quality"],
                    "after": "medium", "change_type": "calibrated",
                    "reason": "官方产品资料支持该项判断。",
                    "evidence_refs": ["forged:fund-fact" if plan["question"] == "fund-partial"
                                      else document["source_id"] + ":fund-fact"],
                }],
                "evidence_candidates": [{
                    "source_id": document["source_id"], "fact_id": "fund-fact",
                    "excerpt_id": segment["excerpt_id"], "statement": "产品官方资料摘录。",
                }],
            }, {"input_tokens": 150, "output_tokens": 20}
        if plan["question"] in {"quality-maintained", "quality-insufficient", "quality-numeric"}:
            claims = context["thesis"]["claims"]
            episode = {
                "summary": "已核对现有来源，本次不调整当前判断。",
                "pending_questions": ["核对下一报告期"],
                "next_step": {
                    "question": "核对下一报告期的现金支付",
                    "required_fields": ["现金支付", "报告期"],
                    "source_hint": "官方报告现金流附注",
                    "completion_criteria": "取得可定位的原文并确认报告期",
                },
                "claim_checks": [{
                    "claim_id": claims[0]["claim_id"], "status": "unresolved",
                    "finding": "现有资料尚不足以核验所选判断。", "evidence_refs": [],
                }],
            }
            candidates = []
            if plan["question"] == "quality-maintained":
                segment = payload["evidence_segments"][0]
                pending = [
                    item["question"] for item in context["pending_questions"]
                    if isinstance(item, dict) and isinstance(item.get("question"), str)
                ]
                if pending:
                    episode["pending_questions"] = pending
                    episode["next_step"]["question"] = pending[0]
                candidates = [{
                    "source_id": document["source_id"], "fact_id": "fact-a",
                    "excerpt_id": segment["excerpt_id"],
                    "statement": "The consideration was paid in cash.",
                }]
                episode["claim_checks"] = [{
                    "claim_id": claims[0]["claim_id"], "status": "supported",
                    "finding": "本次来源支持所选判断，未提出调整。",
                    "evidence_refs": [document["source_id"] + ":fact-a"],
                }]
            elif plan["question"] == "quality-numeric":
                assert any(d.get("numeric_facts") for d in documents) and context.get("calculated_metrics")
            return {"episode": episode, "thesis_patch": [], "evidence_candidates": candidates}, self._usage()
        expected_context_keys = {
            "subject", "question", "thesis", "portfolio_context",
            "pending_questions",
        }
        if plan.get("calculated_metrics"):
            expected_context_keys.add("calculated_metrics")
            assert context["calculated_metrics"] == plan["calculated_metrics"]
        assert set(context) == expected_context_keys
        assert set(context["portfolio_context"]) <= {
            "version", "target_holding", "transactions", "allocation",
        }
        portfolio = context["portfolio_context"]
        assert portfolio["target_holding"]["code"] == "sh600001"
        assert portfolio["target_holding"]["current_value_yuan"] == 1000
        assert portfolio["target_holding"]["cost_basis_yuan"] == 900
        assert len(portfolio["transactions"]) == 1
        assert portfolio["transactions"][0]["amount_yuan"] == 900
        assert any(item["allocation_bucket"] == "growth" for item in portfolio["allocation"])
        assert "permissions" not in document and "content_hash" not in document
        quote = "The consideration was paid in cash."
        segment = next(
            item for item in payload.get("evidence_segments", [])
            if item["source_id"] == document["source_id"] and quote in item["text"]
        )
        claims = context["thesis"]["claims"]
        before_confidence = claims[0]["confidence"] if claims else "low"
        patch = [{
            "entity": "claim", "id": claims[0]["claim_id"], "field": "confidence",
            "before": before_confidence,
            "after": "medium" if before_confidence == "low" else before_confidence,
            "change_type": "strengthened",
            "reason": "The official excerpt was checked against its receipt.",
            "evidence_refs": [document["source_id"] + ":fact-a"],
        }] if claims else []
        if plan["question"] == "partial" and patch:
            patch[0]["evidence_refs"] = ["forged:fact-a"]
        if plan["question"] == "repair" and self.calls.count(plan["plan_id"]) <= 2:
            return {"episode": {}}, {"input_tokens": 150, "output_tokens": 20}
        fact_id = "fact-a"
        if plan["question"] == "repair":
            # A malformed evidence draft has no reusable model-created IDs.
            # Select an exact, program-issued citation for the provided span.
            citation = next(item for item in payload["available_evidence_candidates"]
                            if item["excerpt_id"] == segment["excerpt_id"])
            fact_id = citation["fact_id"]
            for proposed in patch:
                proposed["evidence_refs"] = [citation["evidence_ref"]]
        pending_questions = ["Can the cash payment be reconciled?"]
        next_step_question = "核对现金支付对应的报告期与项目"
        if plan["question"].startswith(("首次校准历史判断：", "核验当前判断：")):
            pending_questions = [
                item["question"] for item in context["pending_questions"]
                if isinstance(item, dict) and isinstance(item.get("question"), str)
            ] or pending_questions
            next_step_question = pending_questions[0]
        return {
            "episode": {
                "summary": "", "pending_questions": pending_questions,
                "next_step": {
                    "question": next_step_question,
                    "required_fields": ["现金支付项目", "报告期"],
                    "source_hint": "官方报告现金流附注",
                    "completion_criteria": "取得可定位的直接披露并核对项目",
                },
                "claim_checks": [{
                    "claim_id": claims[0]["claim_id"], "status": "supported",
                    "finding": "本次来源支持所选判断。", "evidence_refs": [document["source_id"] + ":" + fact_id],
                }] if claims else [],
            },
            "thesis_patch": patch,
            "evidence_candidates": [{"source_id": document["source_id"], "fact_id": fact_id,
                                     "excerpt_id": segment["excerpt_id"], "statement": quote}],
        }, {"input_tokens": 150, "output_tokens": 20}


def serve() -> None:
    for name, value in (
        ("MATRIX_STOCK_V3_INPUT_TOKENS", "25000"),
        ("MATRIX_STOCK_V3_OUTPUT_TOKENS", "1500"),
        ("MATRIX_STOCK_V3_TOTAL_TOKENS", "40000"),
    ):
        os.environ[name] = value
    llm = IntegrationLLM()
    if os.environ.get("STOCK_V3_REPLAY_RESPONSE"):
        llm.replay_result = json.loads(Path(os.environ["STOCK_V3_REPLAY_RESPONSE"]).read_text())["result"]
    app = FastAPI()
    app.state.chat = SimpleNamespace(_get_llm=lambda _: llm)
    app.state.config = SimpleNamespace(llm_available=True)
    app.include_router(route.router)

    @app.get("/acceptance/calls")
    def calls():
        return {"calls": list(llm.calls)}

    @app.get("/acceptance/metric-plans")
    def metric_plans():
        return {"calls": list(llm.metric_plan_calls)}

    @app.post("/acceptance/budget/{enabled}")
    def budget(enabled: int):
        os.environ["MATRIX_STOCK_V3_TOTAL_TOKENS"] = "40000" if enabled else "0"
        return {"enabled": bool(enabled)}

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["STOCK_V3_AGENT_PORT"]),
                log_level="warning")


class Request:
    def __init__(self, llm, payload):
        self.app = SimpleNamespace(state=SimpleNamespace(
            config=SimpleNamespace(llm_available=True),
            chat=SimpleNamespace(_get_llm=lambda _: llm),
        ))
        self.payload = payload
        self.headers = {}

    async def stream(self):
        yield json.dumps(self.payload).encode()

    async def json(self):
        return self.payload


async def verify_reference_repair(base_payload):
    """Real provider adapter + fake HTTP; never contact a model endpoint."""
    for mode in (
        "repaired", "unrepaired", "renamed", "rebound", "missing",
        "partial-numeric", "catalog-ref-missing-candidate", "second-repair",
        "one-shot-segment-reference", "one-shot-invalid-reference-budget",
    ):
        payload = json.loads(json.dumps(base_payload))
        plan, context = payload["plan"], payload["context"]
        plan["scope"]["claim_ids"] = ["legacy-claim-1"]
        context["thesis"] = {"decision": {"confidence": "low"},
                             "claims": [{"claim_id": "legacy-claim-1",
                                         "statement": "核对现金流披露", "confidence": "low"}]}
        if mode == "rebound":
            payload["documents"][0]["excerpt"] = "Official source: cash flow and capital expenditure.\n" * 40
        if mode == "partial-numeric":
            payload["documents"][0]["numeric_facts"] = [{
                "field": "revenue", "value": "1000", "period": "2026-06-30",
                "currency": "CNY", "unit": "yuan",
            }]
        if mode.startswith("one-shot-"):
            plan["scope"] = {
                "claim_ids": ["legacy-claim-1"],
                "max_new_documents": 12, "max_synthesis_calls": 1,
                "max_repair_calls": 2,
            }
        if mode == "one-shot-invalid-reference-budget":
            plan["budget"] = {
                "input_tokens": 25000, "output_tokens": 1500, "total_tokens": 14000,
            }
        plan["documents"] = payload["documents"]
        outgoing = []
        other_excerpt_id = ""
        original_excerpt_id = ""
        catalog_candidate = None

        def fake_http(url, body, headers, timeout):
            nonlocal other_excerpt_id, original_excerpt_id, catalog_candidate
            assert url == "https://api.deepseek.com/chat/completions"
            assert headers["authorization"] == "Bearer fixture-only"
            assert body["max_tokens"] == plan["budget"]["output_tokens"]
            assert body["tool_choice"] == "required" and body["temperature"] == 0
            request = json.loads(body["messages"][-1]["content"])
            outgoing.append(request)
            if mode == "rebound" and len(outgoing) == 1:
                original_excerpt_id = request["evidence_segments"][0]["excerpt_id"]
                other_excerpt_id = request["evidence_segments"][1]["excerpt_id"]
            assert request["required_claim_ids"] == ["legacy-claim-1"]
            source = request["documents"][0]["source_id"]
            segment = request["evidence_segments"][0]
            good_ref, bad_ref = source + ":fact-a", source + ":wrong-fact"
            numeric = mode == "partial-numeric"
            candidate = {"source_id": source, "excerpt_id": segment["excerpt_id"],
                         "fact_id": "fact-a", "statement": "构造的原文事实。"}
            if len(outgoing) == 1 and not numeric:
                schema = body["tools"][0]["function"]["parameters"]
                allowed = schema["properties"]["episode"]["properties"]["claim_checks"]["items"][
                    "properties"]["evidence_refs"]["items"]["enum"]
                assert allowed == request["available_evidence_refs"], (
                    mode, allowed, request["available_evidence_refs"],
                )
                assert request["available_evidence_candidates"]
                if mode == "one-shot-segment-reference":
                    catalog_candidate = next(
                        item for item in request["available_evidence_candidates"]
                        if item["excerpt_id"] == segment["excerpt_id"]
                    )
            if numeric:
                candidate["excerpt_id"] = "forged-excerpt"
            cited_ref = source + ":revenue" if numeric else bad_ref
            patch_ref = source + ":revenue" if numeric else good_ref
            if mode == "one-shot-segment-reference":
                cited_ref = patch_ref = segment["excerpt_id"]
            elif mode == "one-shot-invalid-reference-budget":
                cited_ref = patch_ref = source + ":segment-9999"
            if len(outgoing) >= 2:
                assert patch_ref in request["available_evidence_refs"], (
                    mode, len(outgoing), patch_ref, request["available_evidence_refs"],
                )
                assert bad_ref not in request["available_evidence_refs"], (
                    mode, len(outgoing), bad_ref, request["available_evidence_refs"],
                )
                assert request["available_evidence_candidates"]
            if len(outgoing) >= 3:
                assert request["validation_errors"]
                schema = body["tools"][0]["function"]["parameters"]
                allowed = schema["properties"]["episode"]["properties"]["claim_checks"]["items"][
                    "properties"]["evidence_refs"]["items"]["enum"]
                assert allowed == request["available_evidence_refs"]
                if not numeric and mode == "rebound" and len(outgoing) >= 4:
                    assert any("reference catalog binding" in error
                               for error in request["validation_errors"])
                elif mode == "missing" and len(outgoing) >= 4:
                    errors = " ".join(request["validation_errors"])
                    assert "claim_checks must cover" in errors
                    assert "missing=['legacy-claim-1']" in errors
                elif not numeric:
                    assert any("episode.claim_checks[0].evidence_refs" in error
                               for error in request["validation_errors"]), (
                        mode, len(outgoing), request["validation_errors"],
                    )
                    assert not any("missing=['legacy-claim-1']" in error
                                   for error in request["validation_errors"])
                if (mode == "repaired" and len(outgoing) == 3) or (
                    mode == "second-repair" and len(outgoing) == 4
                ):
                    cited_ref = good_ref
                elif mode == "renamed":
                    # Even a newly grounded candidate cannot rename the
                    # reference catalogue during bounded repair.
                    candidate["fact_id"] = "renamed"
                    cited_ref = patch_ref = source + ":renamed"
                    assert cited_ref not in allowed
                elif mode == "rebound":
                    # The ref is allowed, but it must still identify the same
                    # original excerpt, not another span from that source.
                    cited_ref = good_ref
                    candidate["excerpt_id"] = other_excerpt_id
                    assert good_ref in allowed and other_excerpt_id != original_excerpt_id
                elif mode == "catalog-ref-missing-candidate":
                    catalog_candidate = request["available_evidence_candidates"][0]
                    cited_ref = catalog_candidate["evidence_ref"]
            checks = [{
                "claim_id": "legacy-claim-1", "status": "supported",
                "finding": "构造的核验结果。", "evidence_refs": [cited_ref],
            }]
            if mode == "missing" and len(outgoing) >= 3:
                checks = []
            candidates = [candidate]
            patches = [{
                "entity": "decision", "id": "", "field": "confidence",
                "before": "low", "after": "medium", "change_type": "strengthened",
                "reason": "构造的判断建议。", "evidence_refs": [patch_ref],
            }]
            if mode == "catalog-ref-missing-candidate" and len(outgoing) == 3:
                candidates = []
                patches = []
            if mode == "one-shot-segment-reference":
                candidates = []
            if mode == "one-shot-invalid-reference-budget":
                patch_ref = cited_ref
            result = {
                "episode": {"summary": "仅用于离线验收。", "pending_questions": [],
                            "claim_checks": checks},
                "evidence_candidates": candidates,
                "thesis_patch": patches,
            }
            usage = (
                {"prompt_tokens": 11157, "completion_tokens": 1216, "total_tokens": 12373}
                if mode == "one-shot-invalid-reference-budget"
                else {"prompt_tokens": 150, "completion_tokens": 20, "total_tokens": 170}
            )
            return {
                "choices": [{"finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": "fixture", "type": "function", "function": {
                        "name": "return_json", "arguments": json.dumps(result, ensure_ascii=False),
                    }}],
                }}],
                "usage": usage,
            }

        client = DeepSeekClient(api_key="fixture-only", model="mock-budgeted")
        with patch("matrix.llm.deepseek.post_json", fake_http):
            response = await route.stock_research(Request(client, payload))
        expected_calls = 1 if mode.startswith("one-shot-") else 4 if mode in {
            "second-repair", "unrepaired", "renamed", "rebound", "missing", "partial-numeric",
        } else 3
        assert len(outgoing) == expected_calls, (mode, len(outgoing))
        if mode == "one-shot-segment-reference":
            assert isinstance(response, dict), getattr(response, "body", response)
            check = response["result"]["episode"]["claim_checks"][0]
            assert check["status"] == "supported"
            assert check["evidence_refs"] == [catalog_candidate["evidence_ref"]]
            assert response["result"]["evidence_candidates"][0]["excerpt_id"] == catalog_candidate["excerpt_id"]
            assert not response["result_incomplete"]
        elif mode == "one-shot-invalid-reference-budget":
            assert isinstance(response, dict), getattr(response, "body", response)
            check = response["result"]["episode"]["claim_checks"][0]
            assert check["status"] == "unresolved" and check["evidence_refs"] == []
            next_step = response["result"]["episode"]["next_step"]
            assert "补充可定位的官方原文" in next_step["question"]
            assert next_step["required_fields"] == ["来源链接或文件标识", "可定位的原文段落"]
            assert not response["result"]["thesis_patch"]
            assert response["result_incomplete"]
            assert response["validation_diagnostics"]["filter_counts"] == {
                "dropped_candidates": 0, "dropped_patches": 1,
                "downgraded_claim_checks": 1,
            }
        elif mode in {"unrepaired", "renamed"}:
            assert isinstance(response, dict), getattr(response, "body", response)
            check = response["result"]["episode"]["claim_checks"][0]
            assert check["status"] == "unresolved" and check["evidence_refs"] == []
            assert not response["result"]["thesis_patch"] and response["result_incomplete"]
        elif mode in {"repaired", "partial-numeric", "catalog-ref-missing-candidate", "second-repair"}:
            assert isinstance(response, dict), getattr(response, "body", response)
            assert response["result"]["episode"]["claim_checks"][0]["status"] == "supported"
            assert response["model_calls"] == expected_calls and response["usage_complete"]
            if mode == "partial-numeric":
                diagnostic = response["validation_diagnostics"]
                assert response["result_incomplete"] and not diagnostic["post_filter_errors"]
                assert diagnostic["filter_counts"] == {
                    "dropped_candidates": 1, "dropped_patches": 0,
                    "downgraded_claim_checks": 0,
                }
                assert any("unknown excerpt_id" in error for error in diagnostic["validation_errors"])
            elif mode == "catalog-ref-missing-candidate":
                assert catalog_candidate is not None
                materialized = response["result"]["evidence_candidates"]
                assert len(materialized) == 1
                assert materialized[0]["source_id"] == catalog_candidate["source_id"]
                assert materialized[0]["excerpt_id"] == catalog_candidate["excerpt_id"]
                assert materialized[0]["fact_id"] == catalog_candidate["fact_id"]
                assert materialized[0]["limited_quote"] == materialized[0]["statement"]
                assert not response["result_incomplete"]
            else:
                assert not response["result_incomplete"]
        elif mode == "rebound":
            assert isinstance(response, dict), getattr(response, "body", response)
            assert response["result_incomplete"]
            materialized = response["result"]["evidence_candidates"]
            assert len(materialized) == 1
            assert materialized[0]["excerpt_id"] == original_excerpt_id
        else:
            assert response.status_code == 422, response
            failure = json.loads(response.body)
            details = failure["details"]
            assert failure["error_code"] == "result_contract_invalid"
            assert details["model_calls"] == expected_calls and details["usage_complete"]
            assert details["post_filter_errors"] and details["validation_errors"]
            assert details["filter_counts"] == {
                "dropped_candidates": 0, "dropped_patches": int(mode == "missing"),
                "downgraded_claim_checks": 0,
            }
            errors = " ".join(details["validation_errors"])
            assert ("missing=['legacy-claim-1']" in errors) == (mode == "missing"), (mode, errors)
            assert len(json.dumps(details)) < 2000


async def main():
    for name, value in (
        ("MATRIX_STOCK_V3_INPUT_TOKENS", "25000"),
        ("MATRIX_STOCK_V3_OUTPUT_TOKENS", "1500"),
        ("MATRIX_STOCK_V3_TOTAL_TOKENS", "40000"),
    ):
        os.environ[name] = value
    budget = {"input_tokens": 25000, "output_tokens": 1500, "total_tokens": 40000}
    metric_payload = {
        "plan_id": "metric-fixture", "object_type": "stock", "model": "mock-budgeted",
        "question": "核验收入与利润增长", "claims": [], "pending_questions": [],
    }

    class MetricPlannerLLM(FakeLLM):
        def __init__(self, metric_ids):
            super().__init__()
            self.metric_ids = metric_ids

        def complete_json_budgeted(self, system, messages, maximum, schema=None):
            self.calls.append((system, messages, maximum))
            assert schema["properties"]["metric_ids"]["items"]["enum"] == sorted(
                route._FINANCIAL_METRIC_IDS
            )
            return {"metric_ids": self.metric_ids}, self._usage()

    metric_llm = MetricPlannerLLM(["revenue_yoy_pct", "net_profit_yoy_pct"])
    metric_result = await route.plan_financial_metrics(Request(metric_llm, metric_payload))
    assert metric_result["metric_ids"] == ["revenue_yoy_pct", "net_profit_yoy_pct"]
    assert metric_result["planner_version"] == route._FINANCIAL_METRIC_PLANNER_VERSION
    assert metric_result["usage_complete"] and len(metric_llm.calls) == 1
    assert set(json.loads(metric_llm.calls[0][1][0]["content"])) == {
        "question", "claims", "pending_questions",
    }
    assert "公式" in metric_llm.calls[0][0] and len(metric_llm.calls[0][0]) < 500
    invalid_metric_llm = MetricPlannerLLM(["unregistered_metric"])
    invalid_metric_result = await route.plan_financial_metrics(
        Request(invalid_metric_llm, metric_payload)
    )
    assert invalid_metric_result.status_code == 422 and len(invalid_metric_llm.calls) == 1
    empty_metric_llm = MetricPlannerLLM([])
    empty_metric_result = await route.plan_financial_metrics(
        Request(empty_metric_llm, metric_payload)
    )
    assert empty_metric_result["metric_ids"] == [] and len(empty_metric_llm.calls) == 1

    documents = [{"source_id": "fixture-source",
                  "excerpt": "Official source: operating cash flow, interest and capital expenditure."}]
    plan = {"plan_id": "fixture", "object_type": "stock", "model": "mock-budgeted",
            "confirmed_at": "2026-10-06T00:00:00Z", "input_hash": "fixture-hash",
            "question": "What changed?", "budget": budget, "documents": documents,
            "facts_only": False, "pending_questions": [],
            "portfolio_context": {"version": 1, "transactions": [], "allocation": []}}
    context = {
        "question": plan["question"],
        "subject": {"code": "sh600001", "name": "Fixture"},
        "thesis": {"decision": {}, "claims": []},
        "portfolio_context": {"version": 1, "transactions": [], "allocation": []},
        "pending_questions": [],
    }
    payload = {"plan": plan, "context": context, "documents": documents}
    llm = FakeLLM()
    result = await route.stock_research(Request(llm, payload))
    assert result["model_calls"] == 1 and result["usage_complete"] and len(llm.calls) == 1
    private_plan = json.loads(json.dumps(plan))
    private_context = json.loads(json.dumps(context))
    private_plan["portfolio_context"]["target_holding"] = {
        "code": "sh600001", "name": "Fixture", "currency": "CNY",
        "allocation_bucket": "growth", "position_status": "held", "current_value_yuan": 1000,
    }
    private_context["portfolio_context"] = private_plan["portfolio_context"]
    private_payload = {"plan": private_plan, "context": private_context,
                       "documents": private_plan["documents"]}
    consent_llm = FakeLLM()
    consent_request = Request(consent_llm, private_payload)
    consent_request.url = SimpleNamespace(path="/api/research/investment.v3")
    denied = await route.stock_research(consent_request)
    assert denied.status_code == 422 and not consent_llm.calls
    assert "consent" in json.loads(denied.body)["detail"]
    private_plan["include_portfolio"] = True
    allowed = await route.stock_research(consent_request)
    assert allowed["protocol"] == "investment-research-v3" and consent_llm.calls
    two_stage_plan = dict(plan)
    two_stage_plan["scope"] = {
        "max_new_documents": 12,
        "max_synthesis_calls": 2,
        "max_repair_calls": 2,
    }
    two_stage_payload = dict(payload)
    two_stage_payload["plan"] = two_stage_plan
    llm = FakeLLM()
    result = await route.stock_research(Request(llm, two_stage_payload))
    assert result["model_calls"] == 2 and result["usage_complete"] and len(llm.calls) == 2
    llm = FakeLLM()
    llm.repair = True
    result = await route.stock_research(Request(llm, payload))
    assert result["model_calls"] == 2 and len(llm.calls) == 2
    llm = FakeLLM()
    llm.repair = True
    llm.omit_usage = True
    result = await route.stock_research(Request(llm, payload))
    assert result["model_calls"] == 2 and not result["usage_complete"] and len(llm.calls) == 2
    llm = FakeLLM()
    payload["documents"] = [{"source_id": "unsanctioned", "permissions": {
        "external_model_excerpt": True}, "excerpt": "text", "content_hash": "sha256"}]
    result = await route.stock_research(Request(llm, payload))
    assert result.status_code == 422 and not llm.calls
    payload["documents"] = plan["documents"] = []
    plan["facts_only"] = True
    plan["calculated_metrics"] = [{"id": "fixture:net_profit_yoy_pct", "value": "13.02"}]
    plan.pop("pending_questions", None)
    plan.pop("portfolio_context", None)
    payload["context"] = {"question": plan["question"], "subject": {"name": "Fixture"},
                          "calculated_metrics": plan["calculated_metrics"],
                          "pending_questions": []}
    llm = FakeLLM()
    result = await route.stock_research(Request(llm, payload))
    assert result["model_calls"] == 1 and len(llm.calls) == 1
    llm = FakeLLM()
    payload["context"]["calculated_metrics"] = [{"id": "fixture:net_profit_yoy_pct", "value": "13.01"}]
    result = await route.stock_research(Request(llm, payload))
    assert result.status_code == 422 and not llm.calls
    payload["context"]["calculated_metrics"] = plan["calculated_metrics"]
    payload["context"]["thesis"] = {"private": "must not be transmitted"}
    result = await route.stock_research(Request(llm, payload))
    assert result.status_code == 422 and not llm.calls
    del payload["context"]["thesis"]
    forbidden = {"source_id": "fixture", "content_hash": "fixture-hash",
                 "excerpt": "restricted original", "numeric_facts": [
                     {"field": "net_profit", "value": "1130.15", "period": "2026-06-30",
                      "currency": "CNY", "unit": "yuan"}],
                 "permissions": {"external_model_facts": True, "external_model_excerpt": False}}
    payload["documents"] = plan["documents"] = [forbidden]
    result = await route.stock_research(Request(llm, payload))
    assert result.status_code == 422 and not llm.calls
    payload["documents"] = plan["documents"] = []
    short_lines = "其他综合收益\n税后净额\n归属项目\n" * 100 + (
        "合并现金流量表\n经营现金流\n1000\n利息支付\n100\n资本开支\n300\n"
    )
    source_docs = [{"source_id": "short-lines", "excerpt": short_lines}]
    evidence_context = {"question": "核验经营现金流、利息和资本开支", "thesis": {"claims": []}}
    segments = route._evidence_segments(source_docs, evidence_context)
    assert segments and "资本开支" in segments[0]["text"]
    assert sum(len(s["text"]) for s in segments) > 500
    for segment in segments:
        assert short_lines[segment["excerpt_start"]:segment["excerpt_end"]] == segment["text"]
        assert len(segment["text"]) <= 240
    missing_plan = {**plan, "facts_only": False, "documents": []}
    missing_plan.pop("calculated_metrics", None)
    missing_context = {
        "question": missing_plan["question"], "subject": {"code": "sh600001", "name": "Fixture"},
        "thesis": {"claims": [], "decision": {}},
        "portfolio_context": {"version": 1, "transactions": [], "allocation": []},
        "pending_questions": [],
    }
    missing_plan["pending_questions"] = []
    missing_plan["portfolio_context"] = missing_context["portfolio_context"]
    no_source_llm = FakeLLM()
    no_source = await route.stock_research(Request(no_source_llm, {
        "plan": missing_plan, "context": missing_context, "documents": [],
    }))
    assert no_source.status_code == 422 and not no_source_llm.calls
    assert json.loads(no_source.body)["code"] == "evidence_extraction_empty"
    large_documents = [
        {"source_id": f"large-{i}", "excerpt": "官方报告经营现金流与资本支出披露。\n" * 2000}
        for i in range(2)
    ]
    trim_plan = {**missing_plan, "documents": large_documents,
                 "budget": {"input_tokens": 7000, "output_tokens": 1500, "total_tokens": 20000}}
    trim_llm = FakeLLM()
    trimmed = await route.stock_research(Request(trim_llm, {
        "plan": trim_plan, "context": missing_context, "documents": large_documents,
    }))
    assert isinstance(trimmed, dict), getattr(trimmed, "body", trimmed)
    assert trimmed["evidence_selection"]["budget_trimmed"] and trim_llm.calls
    assert 0 < trimmed["evidence_selection"]["selected_char_count"] < 11520
    numeric_text_documents = [{**large_documents[0], "numeric_facts": [
        {"field": "revenue", "value": "1000", "period": "2026-06-30",
         "currency": "CNY", "unit": "yuan"},
    ]}]
    exhausted_plan = {**missing_plan, "documents": numeric_text_documents,
                      "budget": {"input_tokens": 1, "output_tokens": 1500, "total_tokens": 20000}}
    exhausted_llm = FakeLLM()
    exhausted = await route.stock_research(Request(exhausted_llm, {
        "plan": exhausted_plan, "context": missing_context, "documents": numeric_text_documents,
    }))
    assert exhausted.status_code == 422 and not exhausted_llm.calls
    assert json.loads(exhausted.body)["code"] == "input_context_too_large"
    numeric_doc = {"source_id": "numeric-ref", "numeric_facts": [
        {"field": "revenue", "value": "1000", "period": "2026-06-30", "currency": "CNY", "unit": "yuan"},
    ]}
    checked = {
        "episode": {"summary": "来源数据已核对。", "pending_questions": [],
                    "claim_checks": [{"claim_id": "claim-1", "status": "supported",
                                      "finding": "已核对披露数值。", "evidence_refs": ["numeric-ref:revenue"]}],
                    "next_step": {"question": "核对资本开支", "required_fields": ["购建资产现金支出"],
                                  "source_hint": "现金流量表", "completion_criteria": "取得同口径报告期数值"}},
        "thesis_patch": [], "evidence_candidates": [],
    }
    assert not route._v3_result_errors(checked, [numeric_doc], [], facts_only=False)
    assert not route._v3_result_errors(
        checked, [numeric_doc], [], facts_only=False, required_claim_ids={"claim-1"},
    )
    checked["episode"]["claim_checks"][0]["evidence_refs"] = ["forged:revenue"]
    assert route._v3_result_errors(checked, [numeric_doc], [], facts_only=False)
    scoped_plan = {
        **missing_plan, "documents": documents, "evidence_selection_version": "paragraph-v2",
        "scope": {"claim_ids": ["claim-2", "claim-1"], "max_new_documents": 12,
                  "max_synthesis_calls": 2, "max_repair_calls": 2},
    }
    scoped_context = {
        **missing_context,
        "thesis": {"decision": {}, "claims": [
            {"claim_id": "claim-1", "statement": "核验经营现金流"},
            {"claim_id": "claim-2", "statement": "核验资本支出"},
        ]},
    }
    scoped_payload = {"plan": scoped_plan, "context": scoped_context, "documents": documents}
    for mode in ("missing", "extra", "duplicate", "forged"):
        scoped_llm = ScopedLLM(mode)
        scoped_result = await route.stock_research(Request(scoped_llm, scoped_payload))
        assert isinstance(scoped_result, dict), getattr(scoped_result, "body", scoped_result)
        assert scoped_result["model_calls"] == 3 and len(scoped_llm.calls) == 3
        assert not route._v3_result_errors(
            scoped_result["result"], documents, [], facts_only=False,
            required_claim_ids={"claim-1", "claim-2"},
        )
        assert all(c["status"] == "unresolved" for c in scoped_result["result"]["episode"]["claim_checks"])
        assert not scoped_result["result"]["thesis_patch"]
    failed_llm = ScopedLLM("missing", repairable=False)
    failed_result = await route.stock_research(Request(failed_llm, scoped_payload))
    assert failed_result.status_code == 422 and len(failed_llm.calls) == 4
    failure = json.loads(failed_result.body)
    assert failure["error_code"] == "result_contract_invalid"
    assert failure["details"]["model_calls"] == 4 and failure["details"]["usage_complete"]
    assert "input_tokens" in json.dumps(failure["details"]["usage"])
    assert "missing=['claim-2']" in " ".join(failure["details"]["validation_errors"])
    empty_schema = route._v3_result_schema(facts_only=False, required_claim_ids=set())
    assert empty_schema["properties"]["episode"]["properties"]["claim_checks"]["maxItems"] == 0
    await verify_reference_repair(scoped_payload)
    empty_refs = route._v3_result_schema(
        facts_only=False, required_claim_ids={"claim-1"}, available_evidence_refs=[],
    )
    assert empty_refs["properties"]["thesis_patch"]["maxItems"] == 0
    assert empty_refs["properties"]["episode"]["properties"]["claim_checks"]["items"][
        "properties"]["evidence_refs"]["maxItems"] == 0
    assert not route._v3_result_errors(
        {"episode": {"summary": "", "pending_questions": [], "claim_checks": [{
            "claim_id": "claim-1", "status": "unresolved",
            "finding": "无可用引用。", "evidence_refs": [],
        }], "next_step": checked["episode"]["next_step"]},
         "thesis_patch": [], "evidence_candidates": []},
        [], [], facts_only=False, required_claim_ids={"claim-1"}, available_evidence_refs=set(),
    )
    diagnostic = route._validation_diagnostics(
        ["test sk-secretfixture Bearer secretfixture " + "x" * 1000] * 20,
        ["y" * 1000] * 20, {"dropped_candidates": 1, "dropped_patches": 0},
    )
    assert "secretfixture" not in json.dumps(diagnostic)
    assert len(diagnostic["validation_errors"]) == 3
    assert all(len(error) <= 200 for error in diagnostic["validation_errors"])
    assert diagnostic["error_counts"] == {"before_filter": 20, "after_filter": 20}
    os.environ["MATRIX_STOCK_V3_TOTAL_TOKENS"] = "0"
    result = await route.stock_research(Request(llm, payload))
    assert result.status_code == 503 and not llm.calls
    print(json.dumps({"status": "passed", "checks": [
        "two-stage synthesis", "up to two bounded repairs", "repair without provider usage",
        "immutable documents",
        "immutable API calculations", "private context/excerpt blocked", "explicit V3 portfolio consent",
        "missing budget fails closed", "short PDF lines and table context",
        "exact original span mapping", "empty evidence skips model",
        "numeric citations and executable next step", "evidence trimmed within confirmed budget",
        "exact scoped IDs in all stages and provider schema",
        "financial metric planner whitelist and bounded response",
        "missing/extra/duplicate/forged checklist bounded repair",
        "unrepairable checklist fails closed with usage, not a budget error",
        "real DeepSeek adapter with fake HTTP: precise reference repair/failure",
        "invalid checks preserved, missing distinct from invalid",
        "repair reference catalogue and schema agree, renaming/rebinding rejected",
        "safe numeric partial result retains filter diagnostics",
        "bounded redacted diagnostics, zero real model requests"]}))


def replay_recorded_evidence(run_id: str) -> None:
    """Inspect a saved local input without invoking a model or writing assets."""
    db_path = Path(__file__).resolve().parents[3] / "personal-os/var/automation/reviews.sqlite"
    with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as db:
        row = db.execute("SELECT context_json FROM review_runs WHERE id=?", (run_id,)).fetchone()
    assert row, "Recorded run not found"
    plan = json.loads(row[0])
    documents = plan["documents"]
    context = {"question": plan["question"], "pending_questions": plan.get("pending_questions", []),
               "thesis": {"claims": []}}
    selected = route._evidence_segments(documents, context)
    by_id = {d["source_id"]: d["excerpt"] for d in documents}
    for segment in selected:
        raw = by_id[segment["source_id"]]
        assert raw[segment["excerpt_start"]:segment["excerpt_end"]] == segment["text"]
    old_characters = 0
    for document in documents:
        pieces = []
        for line in document["excerpt"].splitlines():
            for start in range(0, len(line), 240):
                text = line[start:start + 240].strip()
                if text and len(pieces) < 6:
                    pieces.append(text)
        old_characters += sum(len(p) for p in pieces)
    print(json.dumps({"recorded_run": run_id, "model_calls": 0,
                      "source_characters": sum(len(d["excerpt"]) for d in documents),
                      "old_selected_characters": old_characters,
                      "new_selected_characters": sum(len(s["text"]) for s in selected),
                      "segments": len(selected), "exact_source_spans": True}))


if __name__ == "__main__":
    if sys.argv[1:] == ["--serve"]:
        serve()
    elif sys.argv[1:2] == ["--recorded-run"]:
        replay_recorded_evidence(sys.argv[2])
    else:
        asyncio.run(main())
