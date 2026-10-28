#!/usr/bin/env python3
"""Exercise the stock v3 application protocol with a local fake LLM."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from matrix.llm import DeepSeekClient
from matrix.llm.errors import LLMResponseFormatError, LLMTransientError
from matrix.server.routes import stock_research as route
import uvicorn


def adapt_v2_fixture(advice, policy, mode=""):
    if not route.is_deterministic_decision(policy):
        return advice
    methods = policy["valuation_methods"]
    method = next((item for item in methods if item["status"] == "ready"), methods[0])
    valuation = advice["valuation"]
    rationale = valuation.get("rationale", valuation.get("basis", "离线方法解释，不代表真实投资建议。"))
    invalid_refs = valuation.get("calculation_refs") == ["invalid:metric"]
    advice.pop("new_money", None)
    advice.pop("review_trigger", None)
    advice["review_trigger_ids"] = [policy["review_triggers"][0]["id"]]
    advice["valuation"] = {
        "status": valuation["status"], "method_id": method["id"],
        "uncertainty": (
            "data_missing" if method["status"] == "blocked" else
            "business_uncertainty" if valuation["status"] == "unknown" else "none"
        ),
        "rationale": rationale, "evidence_refs": valuation["evidence_refs"],
        "calculation_refs": ["invalid:metric"] if invalid_refs else (
            method["metric_ids"] if method["status"] == "ready" else []
        ),
    }
    if mode == "valuation-false-gap":
        advice["valuation"].update({"status": "unknown", "uncertainty": "data_missing"})
    if mode == "valuation-stale-trigger":
        advice["review_trigger_ids"] = ["already_published_q3"]
    if mode == "valuation-derived-action":
        advice["new_money"] = "consider_buy"
    if mode == "valuation-method-ref":
        advice["valuation"]["calculation_refs"] = []
    if mode == "valuation-narrative-override":
        advice["valuation"]["basis"] = "模型改写的数值结论"
    if "decision_tools" in policy:
        tools = policy["decision_tools"]
        selected, seen = [], set()
        for case in tools["cases"]:
            if case["tool_id"] not in seen and len(selected) < 2:
                selected.append({"case_id": case["id"],
                                 "implication": "离线情景用于检查价格承受能力，不是经营预测或买入规则。"})
                seen.add(case["tool_id"])
        selected_ids = {item["case_id"] for item in selected}
        conditions = [item["id"] for item in tools["conditions"] if item["case_id"] in selected_ids][:2]
        advice.pop("conditions", None)
        advice["decision_analysis"] = {
            "case_assessments": selected,
            "current_price_reasoning": "离线验收结合条件测算解释当前价格下的行动，不证明投资建议正确。",
            "condition_ids": conditions,
            "unresolved_premise": "后续盈利是否持续低于所选压力情景仍没有足够证据判断。" if advice["valuation"]["status"] == "unknown" else "",
        }
        if mode == "valuation-no-price-reasoning":
            advice["decision_analysis"]["current_price_reasoning"] = ""
        elif mode == "valuation-forged-case":
            advice["decision_analysis"]["case_assessments"][0]["case_id"] = "invented:stress"
        elif mode == "valuation-forged-condition":
            advice["decision_analysis"]["condition_ids"] = ["invented:price"]
        elif mode == "valuation-generic-unknown":
            advice["valuation"].update({"status": "unknown", "uncertainty": "business_uncertainty"})
            advice["decision_analysis"]["unresolved_premise"] = ""
        elif mode == "valuation-conflicting-condition":
            advice["conditions"] = "程序未计算的价格会自动触发买入。"
    return advice


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
                "conclusion": {
                    "supported": "无直接证据。",
                    "unverified": "两项判断仍待核验。",
                    "impact": "投资判断与置信度保持不变。",
                },
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
                           "product_context", "pending_questions", "calculated_metrics",
                           "market_context", "background_claims", "decision_context", "valuation_context"}
            }
            documents = payload.get("documents", [])
        self.calls.append(plan["plan_id"])
        policy = context.get("decision_context")
        if policy and policy.get("decision_tools") is not None:
            begin = system.index('{"episode":', system.index("The investment-research-v3 result contract is strict."))
            example = json.loads(system[begin:system.index(". episode.summary", begin)])
            example_advice = example["episode"]["investment_advice"]
            advice_schema = schema["properties"]["episode"]["properties"]["investment_advice"]
            assert set(example_advice) == set(advice_schema["required"])
            assert "conditions" not in example_advice
            assert "decision_analysis" in example_advice
        if plan["question"] == "response-failed":
            raise LLMResponseFormatError(
                "response root must be a JSON object",
                draft='["PRIVATE_RESPONSE_DRAFT_MARKER"]',
                usage=self._usage(), finish_reason="stop",
            )
        advice_required = "investment_advice" in (
            schema.get("properties", {}).get("episode", {}).get("required", [])
            if isinstance(schema, dict) else []
        )

        def add_advice(episode, evidence_refs=None):
            if advice_required:
                episode["investment_advice"] = {
                    "new_money": "wait", "existing_position": "not_assessed",
                    "reason": "现有证据不足以支持新增，暂不新增。",
                    "conditions": "核验当前估值和现金流后再评估。",
                    "review_trigger": "取得新的官方报告或近期行情估值时。",
                    "evidence_refs": evidence_refs or [],
                }
                if context.get("decision_context"):
                    episode["investment_advice"].update({
                        "valuation": {
                            "status": "unknown", "basis": "缺少可核验的估值比较依据，不能只按 PE 判断低估。",
                            "evidence_refs": [],
                        },
                        "recommendation": "watch",
                        "metric_assessments": [
                            {"metric_id": metric_id, "impact": "negative",
                             "explanation": "该项经营指标显著恶化，削弱当前买入依据。"}
                            for metric_id in context["decision_context"]["required_metric_ids"]
                        ],
                    })
                    if plan["question"] == "decision-invalid":
                        episode["investment_advice"]["recommendation"] = "buy"
                    if context.get("valuation_context"):
                        episode["investment_advice"]["valuation"]["calculation_refs"] = []
                        if isinstance(episode.get("next_step"), dict):
                            episode["next_step"]["kind"] = "fetch_now"
            if context.get("decision_context"):
                adapt_v2_fixture(episode["investment_advice"], context["decision_context"])
            return episode

        if plan["question"].startswith("valuation-"):
            data = context["valuation_context"]
            assert data["version"] == "valuation-data-v2"
            assert data["foundation"]["version"] == "fundamental-data-v1"
            assert len(data["foundation"]["fields"]) == 8
            assert "attempts" not in data["foundation"]
            assert all(
                not {"raw_hash", "source_url"}.intersection(field.get("selected", {}))
                for field in data["foundation"]["fields"]
            )
            assert context["portfolio_context"] == {"version": 1, "transactions": [], "allocation": []}
            assert all("sources" not in metric for metric in data["metrics"])
            episode = add_advice({
                "summary": "隔离验收仅验证计算输入和建议契约。",
                "pending_questions": [],
                "claim_checks": [{"claim_id": item, "status": "unresolved",
                                 "finding": "不把参考估值当作竞争优势证据。", "evidence_refs": []}
                                for item in payload.get("required_claim_ids", [])],
                "conclusion": {"supported": "计算输入齐备。", "unverified": "商业假设待核验。",
                               "impact": "保留方法适用性限制。"},
                "next_step": {"kind": "monitor", "question": "下一报告披露后复查",
                              "required_fields": ["未来报告"], "source_hint": "下一期官方报告",
                              "completion_criteria": "未来披露后核对变化"},
            })
            episode["investment_advice"]["valuation"] = {
                "status": "fair", "basis": "离线测试的估值参考输入，不证明真实投资建议有效。",
                "evidence_refs": [],
                "calculation_refs": ["invalid:metric"] if plan["question"] == "valuation-invalid" else [
                    "pe_reported_ttm", "dividend_yield_12m_pct", "peer_pe_median",
                ],
            }
            episode["investment_advice"]["recommendation"] = "hold"
            episode["next_step"]["kind"] = "monitor"
            if plan["question"] == "valuation-missing":
                episode["investment_advice"]["recommendation"] = "watch"
                episode["investment_advice"]["valuation"]["status"] = "unknown"
                episode["investment_advice"]["valuation"]["calculation_refs"] = ["dividend_yield_12m_pct"]
            if plan["question"] == "valuation-basic-query":
                episode["investment_advice"]["recommendation"] = "watch"
                episode["investment_advice"]["valuation"]["status"] = "unknown"
                episode["investment_advice"]["valuation"]["calculation_refs"] = ["pb_market_reference"]
                episode["next_step"].update({
                    "kind": "fetch_now", "question": "补齐报告PB与每股净资产",
                    "required_fields": ["pb_reported", "reported_book_value_per_share"],
                })
            adapt_v2_fixture(episode["investment_advice"], context["decision_context"], plan["question"])
            return {"episode": episode, "thesis_patch": [], "evidence_candidates": []}, self._usage()

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
        if context.get("decision_context") and any(
            fact.get("field") == "operating_cash_flow"
            for doc in documents for fact in doc.get("numeric_facts", [])
        ):
            assert any(metric["name"] == "operating_cash_flow_yoy_pct"
                       for metric in context["calculated_metrics"])
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
                    "conclusion": {
                        "supported": "产品资料中的目标信息已找到。",
                        "unverified": "跟踪表现仍待核验。",
                        "impact": "仅按已核实范围更新判断。",
                    },
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
                "conclusion": {
                    "supported": "已检查现有来源。",
                    "unverified": "下一报告期仍待披露。",
                    "impact": "投资判断与置信度保持不变。",
                },
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
            return {"episode": add_advice(episode, [
                        document["source_id"] + ":fact-a",
                    ] if candidates else []), "thesis_patch": [],
                    "evidence_candidates": candidates}, self._usage()
        expected_context_keys = {
            "subject", "question", "thesis", "portfolio_context",
            "pending_questions",
        }
        if advice_required:
            expected_context_keys.update({"market_context", "background_claims", "decision_context"})
        if context.get("valuation_context"):
            expected_context_keys.add("valuation_context")
        if plan.get("calculated_metrics"):
            expected_context_keys.add("calculated_metrics")
            assert context["calculated_metrics"] == plan["calculated_metrics"]
        assert set(context) == expected_context_keys, (
            sorted(context), sorted(expected_context_keys),
        )
        assert set(context["portfolio_context"]) <= {
            "version", "target_holding", "transactions", "allocation",
        }
        portfolio = context["portfolio_context"]
        assert portfolio.get("target_holding", {}).get("code") == "sh600001", portfolio
        assert portfolio["target_holding"]["current_value_yuan"] == 1000
        assert portfolio["target_holding"]["cost_basis_yuan"] == 900
        assert len(portfolio["transactions"]) == 1
        assert portfolio["transactions"][0]["amount_yuan"] == 900
        assert any(item["allocation_bucket"] == "growth" for item in portfolio["allocation"])
        assert "permissions" not in document and "content_hash" not in document
        if context.get("decision_context") and (
            "高估、正常还是低估" in plan["question"] or plan["question"] in {"decision-invalid", "response-success"}
        ):
            return {
                "episode": add_advice({
                    "summary": "隔离场景缺少估值依据，不编造行动前提。",
                    "pending_questions": [],
                    "claim_checks": [
                        {"claim_id": item, "status": "unresolved", "finding": "现有资料不足。",
                         "evidence_refs": []}
                        for item in payload.get("required_claim_ids", [])
                    ],
                    "conclusion": {"supported": "无直接证据。", "unverified": "估值依据仍缺失。",
                                   "impact": "不据此买入。"},
                }),
                "thesis_patch": [], "evidence_candidates": [],
            }, self._usage()
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
        episode = {
                "summary": "", "pending_questions": pending_questions,
                "conclusion": {
                    "supported": "官方资料记录了本次核对的事实。",
                    "unverified": "现金支付的后续情况仍待观察。",
                    "impact": "仅依据已核实事实提出变更。",
                },
                "next_step": {
                    "question": next_step_question,
                    "required_fields": ["现金支付项目", "报告期"],
                    "source_hint": "官方报告现金流附注",
                    "completion_criteria": "取得可定位的直接披露并核对项目",
                },
                "claim_checks": [{
                    "claim_id": claims[0]["claim_id"],
                    "status": ("partially_supported" if plan["question"] == "partial-support"
                               else "supported"),
                    "finding": "本次仅找到机制的支持性证据，收益仍待核验。",
                    "evidence_refs": [document["source_id"] + ":" + fact_id],
                }] if claims else [],
        }
        if plan["question"] == "partial-support":
            episode["summary"] = "只有机制有来源支持，收益传导仍待证实。"
            episode["conclusion"] = {
                "supported": "官方资料描述了相关机制。",
                "unverified": "收益传导尚无充分证据。",
                "impact": "当前判断与置信度不调整。",
            }
            episode.pop("next_step")
        return {
            "episode": add_advice(episode, [document["source_id"] + ":" + fact_id]),
            "thesis_patch": [] if plan["question"] == "partial-support" else patch,
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
            if "validation_errors" in request:
                repair_number = sum(
                    "validation_errors" in item
                    for item in outgoing
                )
                assert request["validation_errors"]
                schema = body["tools"][0]["function"]["parameters"]
                allowed = schema["properties"]["episode"]["properties"]["claim_checks"]["items"][
                    "properties"]["evidence_refs"]["items"]["enum"]
                assert allowed == request["available_evidence_refs"]
                if not numeric and mode == "rebound" and repair_number >= 2:
                    assert any("reference catalog binding" in error
                               for error in request["validation_errors"])
                elif mode == "missing" and repair_number >= 2:
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
                if (mode == "repaired" and repair_number == 1) or (
                    mode == "second-repair" and repair_number == 2
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
            if mode == "missing" and "validation_errors" in request:
                checks = []
            candidates = [candidate]
            patches = [{
                "entity": "decision", "id": "", "field": "confidence",
                "before": "low", "after": "medium", "change_type": "strengthened",
                "reason": "构造的判断建议。", "evidence_refs": [patch_ref],
            }]
            if mode == "catalog-ref-missing-candidate" and "validation_errors" in request:
                candidates = []
                patches = []
            if mode == "one-shot-segment-reference":
                candidates = []
            if mode == "one-shot-invalid-reference-budget":
                patch_ref = cited_ref
            result = {
                "episode": {"summary": "仅用于离线验收。", "pending_questions": [],
                            "conclusion": {
                                "supported": "已核对来源。",
                                "unverified": "仍需观察。",
                                "impact": "仅依据有效证据调整判断。",
                            },
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
        expected_calls = 1 if mode.startswith("one-shot-") else 3 if mode in {
            "second-repair", "unrepaired", "renamed", "rebound", "missing", "partial-numeric",
        } else 2
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
            assert response["result"]["episode"].get("next_step") is None
            assert "未形成可追溯" in response["result"]["episode"]["conclusion"]["supported"]
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


async def verify_response_recovery(base_payload):
    marker = "PRIVATE_RESPONSE_DRAFT_MARKER"
    broken = '{"episode":{"summary":"' + marker + '","pending_questions":[]}, BROKEN'

    def provider_response(arguments, *, finish="tool_calls", tool=True, usage=True):
        message = (
            {"content": None, "tool_calls": [{
                "id": "offline", "function": {"name": "return_json", "arguments": arguments},
            }]} if tool else {"content": arguments}
        )
        result = {"choices": [{"finish_reason": finish, "message": message}]}
        if usage:
            result["usage"] = {"prompt_tokens": 100, "completion_tokens": 20}
        return result

    client = DeepSeekClient(api_key="fixture-only", model="mock-budgeted")
    for text in (
        broken, '["' + marker + '"]', "null", '"text"', '{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}',
        '{"a":1}{"b":2}', '{"a":1} trailing', '{"a":1,"a":2}}', '{"a":NaN}}', '{"a":1}}}}',
    ):
        with patch("matrix.llm.deepseek.post_json", return_value=provider_response(text)) as http:
            try:
                client.complete_json_budgeted("Return JSON", [], 1500, schema={"type": "object"})
            except LLMResponseFormatError as error:
                assert error.draft == text and error.usage == {"input_tokens": 100, "output_tokens": 20}
                assert error.recovered_object is None
            else:
                raise AssertionError("invalid object must not be accepted")
            assert http.call_count == 1, "the adapter must never make a hidden fallback request"
    fenced = '```json\n{"ok":true}\n```'
    with patch("matrix.llm.deepseek.post_json", return_value=provider_response(fenced)) as http:
        assert client.complete_json_budgeted("Return JSON", [], 1500, schema={"type": "object"})[0] == {"ok": True}
        assert http.call_count == 1
    good_syntax = {
        "episode": {"summary": "仅报告来源支持的结论。", "pending_questions": [],
                    "claim_checks": [{"claim_id": item, "status": "unresolved",
                                     "finding": "现有资料不足。", "evidence_refs": []}
                                    for item in base_payload["plan"]["scope"]["claim_ids"]],
                    "conclusion": {"supported": "无直接证据。", "unverified": "仍有缺口。",
                                   "impact": "不调整判断。"}},
        "thesis_patch": [], "evidence_candidates": [],
    }
    for tail in ("}", "} \n }"):
        syntax_payload = json.loads(json.dumps(base_payload))
        syntax_payload["plan"]["scope"].update({"max_synthesis_calls": 1, "max_repair_calls": 0})
        raw = json.dumps(good_syntax, ensure_ascii=False) + tail
        with patch("matrix.llm.deepseek.post_json", return_value=provider_response(raw)) as http:
            response = await route.stock_research(Request(client, syntax_payload))
        assert isinstance(response, dict), getattr(response, "body", response)
        assert http.call_count == response["model_calls"] == 1
        assert response["usage"] == {"input_tokens": 100, "output_tokens": 20}
        assert "response_syntax_normalized" in response["budget_adjustments"]
        assert response["private_diagnostics"]["response_drafts"][0]["draft"] == raw
        assert response["private_diagnostics"]["attempts"][0]["syntax_normalization"] == "redundant_trailing_object_closers"
    for finish, wrong_name in (("length", False), ("stop", False), ("tool_calls", True)):
        response = provider_response(json.dumps(good_syntax) + "}", finish=finish)
        if wrong_name:
            response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "wrong_tool"
        with patch("matrix.llm.deepseek.post_json", return_value=response) as http:
            try:
                client.complete_json_budgeted("Return JSON", [], 1500, schema={"type": "object"})
            except LLMResponseFormatError as error:
                assert error.recovered_object is None
            else:
                raise AssertionError("an incomplete or wrong tool call must not be recovered")
            assert http.call_count == 1
    metric_payload = {
        "plan_id": "failed-metric-plan", "object_type": "stock", "model": client.model,
        "question": "核验经营现金流", "claims": [], "pending_questions": [],
    }
    with patch("matrix.llm.deepseek.post_json", return_value=provider_response(broken)) as http:
        failed_metric = await route.plan_financial_metrics(Request(client, metric_payload))
        metric_error = json.loads(failed_metric.body)
        assert failed_metric.status_code == 422 and http.call_count == 1
        assert metric_error["details"]["model_calls"] == 1 and metric_error["details"]["usage_complete"]
        assert marker in metric_error["private_diagnostics"]["response_drafts"][0]["draft"]

    for mode in ("repaired", "array", "exhausted", "missing-usage", "unknown-budget",
                 "truncated", "http-after-draft", "synthesis-format"):
        payload = json.loads(json.dumps(base_payload))
        if mode == "unknown-budget":
            payload["plan"]["budget"]["total_tokens"] = 14000
        expected_claims = payload["plan"]["scope"]["claim_ids"]
        good = {
            "episode": {
                "summary": "仅报告现有来源允许的结论。", "pending_questions": [],
                "claim_checks": [{"claim_id": item, "status": "unresolved",
                                 "finding": "现有资料不足。", "evidence_refs": []}
                                for item in expected_claims],
                "conclusion": {"supported": "无直接证据。", "unverified": "仍有缺口。",
                               "impact": "不调整原判断。"},
            },
            "thesis_patch": [], "evidence_candidates": [],
        }
        outgoing = []

        def fake_http(url, body, headers, timeout):
            request = json.loads(body["messages"][-1]["content"])
            outgoing.append(body)
            assert body["max_tokens"] == payload["plan"]["budget"]["output_tokens"]
            if mode == "http-after-draft" and len(outgoing) == 2:
                raise LLMTransientError("offline provider unavailable")
            malformed = (
                len(outgoing) == 1 if mode != "synthesis-format"
                else len(outgoing) == 2
            )
            malformed = malformed or mode in {"array", "unknown-budget", "exhausted"}
            if len(outgoing) > 1 and mode != "synthesis-format" or mode == "synthesis-format" and len(outgoing) == 3:
                assert "validation_errors" in request and "untrusted_response_draft" in request
                assert marker in request["untrusted_response_draft"] or mode == "truncated"
                assert body.get("response_format") == {"type": "json_object"}
                assert "tools" not in body
            text = ('["' + marker + '"]' if mode == "array" else broken) if malformed else json.dumps(good)
            finish = "length" if mode == "truncated" and len(outgoing) == 1 else (
                "tool_calls" if body.get("tools") else "stop"
            )
            response = provider_response(
                text, finish=finish, tool=bool(body.get("tools")),
                usage=mode not in {"missing-usage", "unknown-budget"},
            )
            if mode == "exhausted":
                response["usage"] = {"prompt_tokens": payload["plan"]["budget"]["total_tokens"],
                                     "completion_tokens": 20}
            return response

        with patch("matrix.llm.deepseek.post_json", fake_http):
            response = await route.stock_research(Request(client, payload))
        failed = mode in {"array", "exhausted", "unknown-budget", "http-after-draft"}
        envelope = json.loads(response.body) if failed else response
        if failed:
            assert response.status_code in (422, 502), envelope
            details = envelope["details"]
            assert marker not in json.dumps(details), "private drafts must not enter public diagnostics"
            expected = 3 if mode == "array" else 2 if mode == "http-after-draft" else None
            if expected is not None:
                assert len(outgoing) == expected
            if mode == "exhausted":
                assert len(outgoing) == 1
            if mode == "unknown-budget":
                assert len(outgoing) <= 2
        else:
            assert isinstance(response, dict), getattr(response, "body", response)
            assert response["result"]["episode"]["summary"] == good["episode"]["summary"]
            assert marker not in json.dumps(response["result"])
            assert len(outgoing) == (3 if mode == "synthesis-format" else 2)
            details = response
        assert details["model_calls"] == len(outgoing)
        assert details["usage_complete"] == (mode not in {"missing-usage", "unknown-budget", "http-after-draft"})
        private = envelope["private_diagnostics"]
        assert len(private["request_id"]) == 32
        assert len(private["attempts"]) == len(outgoing)
        assert any(marker in item["draft"] for item in private["response_drafts"])
        if details["usage_complete"] and mode != "exhausted":
            assert sum(item["input_tokens"] + item["output_tokens"]
                       for item in route._usage_parts(details["usage"])) == 120 * len(outgoing)
    return "real adapter response failures: one HTTP call, strict object parsing, bounded repair, complete usage and private drafts"


async def verify_decision_contract(base_payload):
    """Replay the real route with fake models; this is not investment-quality proof."""
    for code, valuation, action, fault in (
        ("sz000651", "undervalued", "buy", ""),
        ("sh600887", "fair", "hold", ""),
        ("sh601899", "overvalued", "sell", ""),
        ("sz000651", "unknown", "watch", ""),
        ("sh600887", "fair", "hold", "repair-metric"),
        ("sh601899", "fair", "hold", "missing-metric"),
        ("sz000651", "undervalued", "buy", "forged-valuation"),
        ("sh600887", "unknown", "buy", "unknown-buy"),
        ("sh601899", "fair", "hold", "missing-quote"),
    ):
        payload = json.loads(json.dumps(base_payload))
        metric_id = code + ":cash-yoy"
        decision = {
            "version": "stock-decision-v1",
            "objective": "评估当前估值是高估、正常还是低估，并给出买入、持有或卖出建议",
            "required_metric_ids": [metric_id],
        }
        market = {"status": "unavailable"} if fault == "missing-quote" else {
            "status": "available", "price": 30.0, "pe_approx": 8.0,
            "quote_datetime": "2026-10-26", "basis": "isolated fixture",
        }
        metrics = [{
            "id": metric_id, "name": "operating_cash_flow_yoy_pct", "value": "-33.60",
            "unit": "percent", "status": "computed", "period": "2026-06-30",
        }]
        payload["plan"].update({
            "plan_id": code + ":" + fault, "advice_required": True,
            "decision_context": decision, "market_context": market,
            "background_claims": [], "calculated_metrics": metrics,
        })
        payload["plan"]["scope"]["claim_ids"] = []
        payload["context"].update({
            "subject": {"code": code, "name": "离线场景"},
            "decision_context": decision, "market_context": market,
            "background_claims": [], "calculated_metrics": metrics,
        })

        class DecisionLLM(FakeLLM):
            def complete_json_budgeted(self, system, messages, maximum, schema=None):
                self.calls.append((system, messages, maximum))
                request = json.loads(messages[0]["content"])
                assert request["decision_context"] == decision
                assert request["calculated_metrics"] == metrics
                assert "stock-decision-v1" in system
                assert '"valuation":{"status":"unknown"' in system
                advice_schema = schema["properties"]["episode"]["properties"]["investment_advice"]
                assert {"valuation", "recommendation", "metric_assessments"} <= set(advice_schema["required"])
                ref = request["available_evidence_refs"][0]
                assessment = [{"metric_id": metric_id, "impact": "negative",
                               "explanation": "经营现金流同比降约33.6%，现金创造能力转弱，削弱新增依据。"}]
                if fault == "missing-metric" or fault == "repair-metric" and "validation_errors" not in request:
                    assessment = []
                return {
                    "episode": {
                        "summary": "离线决策场景。", "pending_questions": [], "claim_checks": [],
                        "investment_advice": {
                            "new_money": {"buy": "consider_buy", "sell": "avoid"}.get(action, "wait"),
                            "existing_position": "not_assessed",
                            "reason": "已结合资料、估值与现金流评估当前行动。",
                            "conditions": "盈利能力或估值依据发生实质变化时重新评估。",
                            "review_trigger": "新的官方报告披露。",
                            "evidence_refs": [ref],
                            "valuation": {
                                "status": valuation,
                                "basis": "离线提供的经营与估值对照依据；不是用低 PE 自动判断低估。",
                                "evidence_refs": ["forged:valuation"] if fault == "forged-valuation" else [ref],
                            },
                            "recommendation": action, "metric_assessments": assessment,
                        },
                    },
                    "thesis_patch": [], "evidence_candidates": [],
                }, self._usage()

        llm = DecisionLLM()
        response = await route.stock_research(Request(llm, payload))
        if fault in {"missing-metric", "forged-valuation", "unknown-buy", "missing-quote"}:
            assert response.status_code == 422, (code, fault, response)
            error = json.loads(response.body)
            assert error["code"] == "result_contract_invalid"
            assert error["details"]["model_calls"] == 3
            assert error["details"]["repair_attempts"] == 2
        else:
            assert isinstance(response, dict), getattr(response, "body", response)
            advice = response["result"]["episode"]["investment_advice"]
            assert advice["recommendation"] == action and advice["valuation"]["status"] == valuation
            assert advice["metric_assessments"][0]["metric_id"] == metric_id
            assert response["model_calls"] == 2
            if fault == "repair-metric":
                assert "validation_errors" in json.loads(llm.calls[1][1][0]["content"])
    return "decision-v1 cross-target valuation/actions, same evidence across synthesis/repair, no prose false positive"


async def verify_deterministic_decisions(base_payload):
    for scenario, code, status, action in (
        ("optional-bps-missing", "sz000651", "undervalued", "buy"),
        ("optional-history-missing", "sh600887", "fair", "hold"),
        ("risk-outweighs-price", "sh601899", "overvalued", "sell"),
        ("business-uncertain", "sz000651", "unknown", "watch"),
        ("known-with-business-risk", "sz000651", "undervalued", "watch"),
        ("all-blocked", "sh600887", "unknown", "watch"),
        ("false-data-gap", "sz000651", "unknown", "watch"),
        ("blocked-choice", "sh600887", "unknown", "watch"),
        ("missing-method-ref", "sh601899", "fair", "hold"),
        ("forged-ref", "sz000651", "fair", "hold"),
        ("unknown-buy", "sh600887", "unknown", "buy"),
        ("invented-trigger", "sh601899", "fair", "hold"),
        ("conflicting-derived-action", "sz000651", "fair", "hold"),
        ("conflicting-derived-basis", "sh600887", "fair", "hold"),
        ("invalid-format", "sh601899", "fair", "hold"),
        ("invalid-action-type", "sz000651", "fair", "hold"),
    ):
        payload = json.loads(json.dumps(base_payload))
        blocked = scenario == "all-blocked"
        metric_ids = ["peer_target_pe", "peer_pe_median", "pe_to_peer_median"]
        valuation_data = {
            "version": "valuation-data-v1", "as_of": "2026-10-27", "currency": "CNY",
            "data_hash": "b" * 64, "peers": [], "gaps": [],
            "metrics": [
                {"id": item, "value": value, "unit": "ratio", "period": "2026-09-30",
                 "status": "computed", "basis": "isolated same-period reference",
                 "verification": "provider_unverified"}
                for item, value in zip(metric_ids, ("8", "12", "0.6667"))
            ] if not blocked else [],
        }
        policy = {
            "version": "stock-decision-v2", "as_of": "2026-10-27",
            "latest_report_period": "2026-09-30",
            "objective": "评估当前估值是高估、正常还是低估，并给出买入、持有或卖出建议",
            "required_metric_ids": [],
            "valuation_methods": [
                {"id": "peer_relative", "name": "同业相对估值",
                 "status": "blocked" if blocked else "ready", "metric_ids": metric_ids,
                 "missing_inputs": metric_ids if blocked else [],
                 "summary": "同业相对估值：同口径比较参考，不能单独证明内在价值。"},
                {"id": "reported_assets", "name": "账面资产参考", "status": "blocked",
                 "metric_ids": ["pb_reported"], "missing_inputs": ["pb_reported"],
                 "summary": "账面资产参考：缺少报告口径市净率。"},
            ],
            "review_triggers": [
                {"id": "next_report", "period": "2026-12-31",
                 "label": "财报期为2026-12-31的后续定期报告披露时"},
                {"id": "material_event", "label": "影响经营或估值的重大事项披露时"},
            ],
        }
        # No report-derived PE/PB is available: a supported relative method must
        # not be blocked by that legacy catch-all prerequisite.
        market = {"status": "available", "price": 36.58, "quote_datetime": "2026-10-27",
                  "pe_approx": None, "pb_approx": None, "basis": "isolated quote"}
        payload["plan"].update({
            "plan_id": "v2:" + scenario, "advice_required": True,
            "decision_context": policy, "valuation_context": valuation_data,
            "market_context": market, "background_claims": [], "calculated_metrics": [],
        })
        payload["plan"]["scope"].update({"claim_ids": [], "max_synthesis_calls": 1, "max_repair_calls": 2})
        payload["context"].update({
            "subject": {"code": code, "name": "确定性边界场景"},
            "decision_context": policy, "valuation_context": valuation_data,
            "market_context": market, "background_claims": [], "calculated_metrics": [],
        })

        class PolicyLLM(FakeLLM):
            def complete_json_budgeted(self, system, messages, maximum, schema=None):
                self.calls.append((system, messages, maximum))
                request = json.loads(messages[0]["content"])
                assert all("summary" not in item for item in request["decision_context"]["valuation_methods"])
                advice_schema = schema["properties"]["episode"]["properties"]["investment_advice"]
                assert not {"new_money", "review_trigger"}.intersection(advice_schema["properties"])
                assert "basis" not in advice_schema["properties"]["valuation"]["properties"]
                assert "stock-decision-v2" in system
                if scenario == "invalid-format":
                    raise LLMResponseFormatError(
                        "invalid JSON", draft="[invalid output]", usage=self._usage(), finish_reason="stop",
                    )
                ref = request["available_evidence_refs"][0]
                advice = {
                    "recommendation": action, "existing_position": "not_assessed",
                    "reason": "结合盈利与现金转换的不确定性权衡当前行动。",
                    "conditions": "经营证据或估值依据发生实质变化。",
                    "review_trigger_ids": ["next_report"], "evidence_refs": [ref],
                    "valuation": {
                        "status": status, "method_id": "peer_relative",
                        "uncertainty": ("data_missing" if blocked else
                                        "business_uncertainty" if status == "unknown" else "none"),
                        "rationale": "估值比较的局限需要结合业务前景判断，不推断绝对公允价格。",
                        "evidence_refs": [ref], "calculation_refs": [] if blocked else list(metric_ids),
                    },
                    "metric_assessments": [],
                }
                if scenario == "false-data-gap":
                    advice["valuation"]["uncertainty"] = "data_missing"
                if scenario == "known-with-business-risk":
                    advice["valuation"]["uncertainty"] = "business_uncertainty"
                if scenario == "blocked-choice":
                    advice["valuation"].update({"method_id": "reported_assets", "uncertainty": "data_missing"})
                if scenario == "missing-method-ref":
                    advice["valuation"]["calculation_refs"].pop()
                if scenario == "forged-ref":
                    advice["valuation"]["evidence_refs"] = ["forged:evidence"]
                if scenario == "invented-trigger":
                    advice["review_trigger_ids"] = ["already_published_q3"]
                if scenario == "conflicting-derived-action":
                    advice["new_money"] = "consider_buy"
                if scenario == "conflicting-derived-basis":
                    advice["valuation"]["basis"] = "模型自行覆盖计算摘要"
                if scenario == "invalid-action-type":
                    advice["recommendation"] = ["hold"]
                return {
                    "episode": {"summary": "离线契约验收。", "pending_questions": [],
                                "claim_checks": [], "investment_advice": advice},
                    "thesis_patch": [], "evidence_candidates": [],
                }, self._usage()

        llm = PolicyLLM()
        response = await route.stock_research(Request(llm, payload))
        assert len(llm.calls) == 1, "program projection and invalid DTOs must not start model repairs"
        if scenario in {
            "optional-bps-missing", "optional-history-missing", "risk-outweighs-price",
            "business-uncertain", "all-blocked",
            "known-with-business-risk",
        }:
            assert isinstance(response, dict), getattr(response, "body", response)
            advice = response["result"]["episode"]["investment_advice"]
            assert advice["new_money"] == {"buy": "consider_buy", "sell": "avoid"}.get(action, "wait")
            assert advice["valuation"]["basis"] == policy["valuation_methods"][0]["summary"]
            assert "2026-12-31" in advice["review_trigger"] and "2026-09-30" not in advice["review_trigger"]
            assert advice["recommendation"] == action and advice["valuation"]["status"] == status
        else:
            assert response.status_code == 422, (scenario, getattr(response, "body", response))
            error = json.loads(response.body)
            assert error["code"] == "result_contract_invalid"
            assert error["details"]["model_calls"] == 1 and error["details"]["repair_attempts"] == 0
        if scenario == "optional-bps-missing":
            invalid = json.loads(json.dumps(payload))
            for container in (invalid["plan"], invalid["context"]):
                container["decision_context"]["review_triggers"][0]["period"] = "2026-09-30"
            prior_calls = len(llm.calls)
            bad_request = await route.stock_research(Request(llm, invalid))
            assert bad_request.status_code in (400, 422) and len(llm.calls) == prior_calls
    return "decision-v2 deterministic methods/actions/triggers, optional gaps isolated, invalid results fail without LLM repairs"


async def verify_capital_return_ordering(base_payload):
    from matrix.server.routes.stock_decision_policy import validate_capital_return_facts
    code, scheme = "sz000651", "2026-019"
    old_quote = "截至2026年7月31日，公司暂未实\n施本次回购"
    current_quote = "截至2026年8月31日，公司累计回购股份数量为1,740,000\n股"
    header = "《回购股份报告书》（公告编号：2026-019）。\n"
    documents = [
        {"source_id": "progress-old", "report_period": "2026-07-31", "excerpt": header + old_quote + "。"},
        {"source_id": "progress-new", "report_period": "2026-08-31", "excerpt": header + current_quote + "。"},
    ]
    fact = {
        "event_id": "repurchase-" + hashlib.sha256((code + "|" + scheme).encode()).hexdigest()[:20],
        "scheme_ref": scheme, "as_of": "2026-08-31", "status": "in_progress",
        "executed_shares": "1740000", "source_id": "progress-new", "quote": current_quote,
        "superseded": [{"source_id": "progress-old", "quote": old_quote}],
        "summary": "截至2026-08-31，该方案已累计回购1740000股，不代表方案完成。",
    }
    policy = {
        "version": "stock-decision-v2", "objective": "判断当前估值与行动。",
        "as_of": "2026-10-27", "required_metric_ids": [],
        "valuation_methods": [{"id": "reported_assets", "name": "账面资产参考", "status": "blocked",
                               "metric_ids": ["pb_reported"], "missing_inputs": ["pb_reported"],
                               "summary": "报告口径市净率缺失，此方法暂不可用。"}],
        "review_triggers": [{"id": "capital_return", "label": "新的回购进度披露时"}],
        "capital_return_facts": [fact],
    }
    validate_capital_return_facts(policy, documents, code)
    selected = route._evidence_segments(documents, {"decision_context": policy})
    assert any(segment["source_id"] == "progress-new" and "1,740,000" in segment["text"] for segment in selected)
    assert not any("暂未实" in segment["text"] for segment in selected)
    assert old_quote in documents[0]["excerpt"], "raw historical source must not be rewritten"
    for field, value in (
        ("scheme_ref", "2026-020"), ("as_of", "2026-08-30"), ("executed_shares", "999"),
        ("status", "not_started"), ("source_id", "missing-source"),
    ):
        bad = json.loads(json.dumps(policy))
        bad["capital_return_facts"][0][field] = value
        try:
            validate_capital_return_facts(bad, documents, code)
        except ValueError:
            pass
        else:
            raise AssertionError("tampered capital return metadata must fail: " + field)
    payload = json.loads(json.dumps(base_payload))
    market = {"status": "available", "price": 36.58, "quote_datetime": "2026-10-27",
              "pe_approx": None, "pb_approx": None, "basis": "isolated reference"}
    payload["documents"] = documents
    payload["plan"].update({
        "documents": documents, "advice_required": True, "decision_context": policy,
        "market_context": market, "background_claims": [], "calculated_metrics": [],
    })
    payload["plan"]["scope"].update({"claim_ids": [], "max_synthesis_calls": 1, "max_repair_calls": 0})
    payload["context"].update({
        "subject": {"code": code, "name": "事件进度夹具"}, "decision_context": policy,
        "market_context": market, "background_claims": [], "calculated_metrics": [],
    })
    client = DeepSeekClient(api_key="fixture-only", model=payload["plan"]["model"])
    for stale in (False, True):
        outgoing = []

        def fake_http(url, body, headers, timeout):
            outgoing.append(body)
            request = json.loads(body["messages"][-1]["content"])
            assert "quote" not in request["decision_context"]["capital_return_facts"][0]
            assert not any("暂未实" in item["text"] for item in request["evidence_segments"])
            refs = request["available_evidence_refs"]
            current_ref = next(ref for ref in refs if ref.startswith("progress-new:"))
            advice = {
                "recommendation": "watch", "existing_position": "not_assessed",
                "reason": "回购尚未实施。" if stale else "回购已经开始，但当前估值证据仍不足。",
                "conditions": "取得适用估值方法的输入后复查。",
                "review_trigger_ids": ["capital_return"], "evidence_refs": [current_ref],
                "valuation": {"status": "unknown", "method_id": "reported_assets",
                              "uncertainty": "data_missing", "rationale": "缺少报告口径市净率。",
                              "calculation_refs": [], "evidence_refs": []},
                "metric_assessments": [],
            }
            result = {"episode": {"summary": "只读事件核验。", "pending_questions": [],
                                  "claim_checks": [], "investment_advice": advice},
                      "thesis_patch": [], "evidence_candidates": []}
            return {
                "choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [{
                    "function": {"name": "return_json", "arguments": json.dumps(result, ensure_ascii=False) + "}"},
                }]}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }

        with patch("matrix.llm.deepseek.post_json", side_effect=fake_http):
            response = await route.stock_research(Request(client, payload))
        assert len(outgoing) == 1
        if stale:
            assert response.status_code == 422
            assert "repurchase" in json.dumps(json.loads(response.body)["details"]["validation_errors"])
        else:
            assert isinstance(response, dict), getattr(response, "body", response)
            assert response["model_calls"] == 1 and "response_syntax_normalized" in response["budget_adjustments"]
            assert response["result"]["episode"]["investment_advice"]["reason"] == "回购已经开始，但当前估值证据仍不足。"
    return "capital returns: same-scheme source/date verification, current evidence priority, obsolete spans excluded and stale claims rejected"


async def verify_prepared_evidence_and_comparisons(base_payload):
    from matrix.server.routes.stock_decision_policy import validate_comparisons, validate_prepared_evidence
    for profile, fault in (
        ("general", ""), ("utility", ""), ("banking", ""), ("general", "compound-valid"),
        ("general", "direction"), ("general", "missing-reference"), ("general", "historical-as-current"),
        ("banking", "inapplicable"),
    ):
        payload = json.loads(json.dumps(base_payload))
        quote = ("2026年前三季度经营活动产生的现金流量净额同比下降，剔除受限资金影响后下降24.71%，主要系销售商品收到的现金减少。"
                 if profile != "banking" else "2026年前三季度净利润同比下降，主要系信用减值变化，不良贷款率和资本充足率已披露。")
        documents = [{"source_id": "current-report", "report_period": "2026-09-30",
                      "excerpt": "本报告原因说明。\n" + quote + "\n其他资料。"}]
        explanation = {
            "id": "reported-" + hashlib.sha256(("current-report|" + quote).encode()).hexdigest()[:20],
            "source_id": "current-report", "period": "2026-09-30", "quote": quote,
            "kind": "issuer_explanation" if profile == "banking" else "issuer_reported_adjustment",
            "metric_ids": ["net_profit_yoy_pct" if profile == "banking" else "operating_cash_flow_yoy_pct"],
        }
        def number(code, value):
            return {"code": code, "value": value, "unit": "ratio", "currency": "CNY",
                    "period": "2025-12-31", "period_type": "FY", "verification": "provider_unverified"}
        comparison = {
            "id": "peer_cash_conversion_ttm", "metric_id": "ttm_cash_to_profit",
            "status": "comparable", "basis": "same_period_currency_reported_ttm", "period": "2025-12-31",
            "target": number("sz000651", "1.6000"), "peers": [number("sz000333", "1.2000"), number("sh600690", "1.4000")],
            "median": "1.3000", "position": "higher",
        }
        if profile == "banking":
            comparison = {key: value for key, value in comparison.items() if key not in {"target", "median", "position"}}
            comparison.update({"status": "not_applicable", "period": "", "peers": [], "reason": "industry_not_applicable"})
        policy = {
            "version": "stock-decision-v2", "objective": "本次为离线契约验收。",
            "as_of": "2026-10-27", "latest_report_period": "2026-09-30", "required_metric_ids": [],
            "analysis_profile": {"kind": profile, "source_id": "current-report"},
            "reported_explanations": [explanation], "comparisons": [comparison],
            "valuation_methods": [{"id": "reported_assets", "name": "账面资产参考", "status": "blocked",
                                   "metric_ids": ["pb_reported"], "missing_inputs": ["pb_reported"],
                                   "summary": "缺少报告口径市净率，此方法暂不可用。"}],
            "review_triggers": [{"id": "next_report", "period": "2026-12-31", "label": "下一份定期报告披露时"}],
        }
        validate_prepared_evidence(policy, documents)
        validate_comparisons(policy, "sz000651")
        if profile == "general" and not fault:
            for field, value in (("currency", "USD"), ("period", "2026-09-30"), ("unit", "percent")):
                bad = json.loads(json.dumps(policy))
                bad["comparisons"][0]["peers"][0][field] = value
                try:
                    validate_comparisons(bad, "sz000651")
                except ValueError:
                    pass
                else:
                    raise AssertionError("mixed comparison input must fail before model: " + field)
            altered = json.loads(json.dumps(policy))
            altered["reported_explanations"][0]["quote"] += "模型补写的原因"
            try:
                validate_prepared_evidence(altered, documents)
            except ValueError:
                pass
            else:
                raise AssertionError("rewritten issuer disclosure must fail")
        market = {"status": "available", "price": 36.6, "quote_datetime": "2026-10-27",
                  "pe_approx": None, "pb_approx": None, "basis": "isolated quote"}
        payload["documents"] = documents
        payload["plan"].update({"documents": documents, "advice_required": True, "decision_context": policy,
                                "market_context": market, "background_claims": [], "calculated_metrics": []})
        payload["plan"]["scope"].update({"claim_ids": [], "max_synthesis_calls": 1, "max_repair_calls": 0})
        payload["context"].update({"subject": {"code": "sz000651", "name": "离线行业样例"},
                                   "decision_context": policy, "market_context": market,
                                   "background_claims": [], "calculated_metrics": []})
        class PreparedLLM(FakeLLM):
            def complete_json_budgeted(self, system, messages, maximum, schema=None):
                self.calls.append((system, messages, maximum))
                request = json.loads(messages[0]["content"])
                assert "quote" not in request["decision_context"]["reported_explanations"][0]
                assert any(quote in item["text"] for item in request["evidence_segments"])
                ref = request["available_evidence_refs"][0]
                reason = "仅使用已披露资本与信用质量资料，不把现金利润比用于银行质量判断。"
                refs = []
                if profile != "banking" or fault == "inapplicable":
                    reason = "共同报告期的现金转换" + ("弱于同业。" if fault == "direction" else "强于同业。")
                    if fault == "historical-as-current":
                        reason = "当前现金转换强于同业。"
                    if fault == "compound-valid":
                        reason = "净利率有所下降，但现金转换强于同业。"
                    refs = [] if fault == "missing-reference" else ["peer_cash_conversion_ttm"]
                return {
                    "episode": {"summary": "本次为离线验收，不验证真实投资意见。", "pending_questions": [],
                                "claim_checks": [], "investment_advice": {
                                    "recommendation": "watch", "existing_position": "not_assessed",
                                    "reason": reason, "conditions": "取得适用估值输入后复查。",
                                    "review_trigger_ids": ["next_report"], "evidence_refs": [ref],
                                    "valuation": {"status": "unknown", "method_id": "reported_assets",
                                                  "uncertainty": "data_missing", "rationale": "估值输入仍缺失。",
                                                  "evidence_refs": [], "calculation_refs": refs},
                                    "metric_assessments": [],
                                }},
                    "thesis_patch": [], "evidence_candidates": [],
                }, self._usage()
        llm = PreparedLLM()
        response = await route.stock_research(Request(llm, payload))
        assert len(llm.calls) == 1
        if fault and fault != "compound-valid":
            assert response.status_code == 422, (profile, fault, getattr(response, "body", response))
        else:
            assert isinstance(response, dict), getattr(response, "body", response)
    return "prepared issuer disclosures and same-basis comparisons: manufacturing/banking/utility, no invented adjustments or mixed-period rankings"


async def main():
    foundation = {
        "version": "fundamental-data-v1", "code": "sz000651", "as_of": "2026-10-27",
        "data_hash": "a" * 64,
        "fields": [{"id": key, "name": key, "status": "missing"} for key in route._FUNDAMENTAL_UNITS],
    }
    route._validate_fundamental_context(foundation, "sz000651")
    for invalid in (
        {**foundation, "code": "sh600036"},
        {**foundation, "attempts": []},
        {**foundation, "as_of": "2026-99-99"},
    ):
        try:
            route._validate_fundamental_context(invalid, "sz000651")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid foundation identity/private metadata/date must fail")
    shape = {
        "episode": {"summary": "", "pending_questions": [],
                    "thesis_patch": [], "evidence_candidates": []},
    }
    fixed, moved = route._normalize_result_envelope(shape)
    assert moved == ["evidence_candidates", "thesis_patch"]
    assert fixed == {"episode": {"summary": "", "pending_questions": []},
                     "thesis_patch": [], "evidence_candidates": []}
    assert "thesis_patch" in shape["episode"], "normalization must not mutate the raw draft"
    for ambiguous in (
        {**shape, "thesis_patch": []},
        {"episode": {**shape["episode"], "unsupported": "not an authorized field"}},
        {"episode": {**shape["episode"], "evidence_candidates": {}}},
    ):
        assert route._normalize_result_envelope(ambiguous)[1] == []
    assert route._followup_output_limit(23236, 29180, {
        "input_tokens": 30000, "output_tokens": 10000, "total_tokens": 60000,
    }) == 7584
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
                    "conclusion": {
                        "supported": "已核对披露数值。",
                        "unverified": "资本开支仍待核验。",
                        "impact": "判断保持不变。",
                    },
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
    advised = json.loads(json.dumps(checked))
    advised["episode"]["investment_advice"] = {
        "new_money": "wait", "existing_position": "not_assessed",
        "reason": "回购尚未实施，现金转换仍良好，暂不新增。",
        "conditions": "收入企稳再评估。", "review_trigger": "下一份财报。",
        "evidence_refs": ["numeric-ref:revenue"],
    }
    official_progress = {
        "source_id": "repurchase-progress",
        "excerpt": "截至2026年8月31日，公司以集中竞价方式累计回购股份数量为 1,740,000 股。",
    }
    adverse_metrics = [{
        "name": "operating_cash_flow_yoy_pct", "value": "-33.60",
        "status": "computed", "period": "2026-06-30",
    }]
    advice_errors = route._v3_result_errors(
        advised, [numeric_doc, official_progress], [], facts_only=False,
        advice_required=True, calculated_metrics=adverse_metrics,
    )
    assert any("repurchase progress" in issue for issue in advice_errors), advice_errors
    assert any("operating_cash_flow_yoy_pct" in issue for issue in advice_errors), advice_errors
    advised["episode"]["investment_advice"]["reason"] = (
        "截至8月末已实施回购；经营现金流同比下降33.60%，暂不新增。"
    )
    assert not route._v3_result_errors(
        advised, [numeric_doc, official_progress], [], facts_only=False,
        advice_required=True, calculated_metrics=adverse_metrics,
    )
    partial = json.loads(json.dumps(checked))
    partial["episode"]["claim_checks"][0]["status"] = "partially_supported"
    assert not route._v3_result_errors(
        partial, [numeric_doc], [], facts_only=False, required_claim_ids={"claim-1"},
    )
    partial["thesis_patch"] = [{
        "entity": "decision", "id": "", "field": "confidence",
        "before": "low", "after": "medium", "change_type": "strengthened",
        "reason": "单项事实不足以升级判断。", "evidence_refs": ["numeric-ref:revenue"],
    }]
    assert "judgment or calibration changes require a cited checked claim" in (
        route._v3_result_errors(
            partial, [numeric_doc], [], facts_only=False, required_claim_ids={"claim-1"},
        )
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
    response_check = await verify_response_recovery(scoped_payload)
    decision_check = await verify_decision_contract(scoped_payload)
    deterministic_check = await verify_deterministic_decisions(scoped_payload)
    capital_return_check = await verify_capital_return_ordering(scoped_payload)
    prepared_check = await verify_prepared_evidence_and_comparisons(scoped_payload)
    for mode in ("missing", "extra", "duplicate", "forged"):
        scoped_llm = ScopedLLM(mode)
        scoped_result = await route.stock_research(Request(scoped_llm, scoped_payload))
        assert isinstance(scoped_result, dict), getattr(scoped_result, "body", scoped_result)
        assert scoped_result["model_calls"] == 2 and len(scoped_llm.calls) == 2
        assert not route._v3_result_errors(
            scoped_result["result"], documents, [], facts_only=False,
            required_claim_ids={"claim-1", "claim-2"},
        )
        assert all(c["status"] == "unresolved" for c in scoped_result["result"]["episode"]["claim_checks"])
        assert not scoped_result["result"]["thesis_patch"]
    failed_llm = ScopedLLM("missing", repairable=False)
    failed_result = await route.stock_research(Request(failed_llm, scoped_payload))
    assert failed_result.status_code == 422 and len(failed_llm.calls) == 3
    failure = json.loads(failed_result.body)
    assert failure["error_code"] == "result_contract_invalid"
    assert failure["details"]["model_calls"] == 3 and failure["details"]["usage_complete"]
    assert "input_tokens" in json.dumps(failure["details"]["usage"])
    assert "missing=['claim-2']" in " ".join(failure["details"]["validation_errors"])
    empty_schema = route._v3_result_schema(facts_only=False, required_claim_ids=set())
    assert empty_schema["properties"]["episode"]["properties"]["claim_checks"]["maxItems"] == 0
    assert "conclusion" in route._v3_result_schema(
        facts_only=False, required_claim_ids={"claim-1"},
    )["properties"]["episode"]["required"]
    await verify_reference_repair(scoped_payload)
    empty_refs = route._v3_result_schema(
        facts_only=False, required_claim_ids={"claim-1"}, available_evidence_refs=[],
    )
    assert empty_refs["properties"]["thesis_patch"]["maxItems"] == 0
    assert empty_refs["properties"]["episode"]["properties"]["claim_checks"]["items"][
        "properties"]["evidence_refs"]["maxItems"] == 0
    assert not route._v3_result_errors(
        {"episode": {"summary": "", "pending_questions": [],
                     "conclusion": {"supported": "无直接证据。", "unverified": "仍待核验。",
                                    "impact": "判断不调整。"}, "claim_checks": [{
            "claim_id": "claim-1", "status": "unresolved",
            "finding": "无可用引用。", "evidence_refs": [],
        }]},
         "thesis_patch": [], "evidence_candidates": []},
        [], [], facts_only=False, required_claim_ids={"claim-1"}, available_evidence_refs=set(),
    )
    assert not route._v3_result_errors(
        {"episode": {"summary": "现有资料不足，不调整判断。",
                     "conclusion": {"supported": "无直接证据。", "unverified": "仍待核验。",
                                    "impact": "判断不调整。"},
                     "pending_questions": ["等待下期官方披露"], "claim_checks": [{
                         "claim_id": "claim-1", "status": "unresolved",
                         "finding": "尚无证据，判断保持不变。", "evidence_refs": [],
                     }]}, "thesis_patch": [], "evidence_candidates": []},
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
        "foundation identity, availability, date and private metadata boundaries",
        "unambiguous envelope normalization preserves raw drafts and rejects conflicts",
        "follow-up output cap fits the unchanged total budget",
        "two-stage synthesis", "up to two bounded repairs", "repair without provider usage",
        "immutable documents",
        "immutable API calculations", "private context/excerpt blocked", "explicit V3 portfolio consent",
        "missing budget fails closed", "short PDF lines and table context",
        "exact original span mapping", "empty evidence skips model",
        "numeric citations and executable next step", "evidence trimmed within confirmed budget",
        "exact scoped IDs in all stages and provider schema",
        "financial metric planner whitelist and bounded response",
        decision_check,
        deterministic_check,
        capital_return_check,
        prepared_check,
        response_check,
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

async def replay_recorded_response(path: str) -> None:
    """Use the saved response as a fake provider, preserving the real public DTO."""
    recorded = json.loads(Path(path).read_text())
    payload = recorded["payload"]
    original = recorded["result"]
    untouched = json.dumps(original, sort_keys=True)
    normalized, moved = route._normalize_result_envelope(original)
    assert moved == ["evidence_candidates", "thesis_patch"]
    assert set(normalized) == route._V3_RESULT_KEYS
    assert json.dumps(original, sort_keys=True) == untouched
    collision = json.loads(json.dumps(original))
    collision["thesis_patch"] = []
    assert route._normalize_result_envelope(collision)[1] == []
    malformed = json.loads(json.dumps(original))
    malformed["episode"]["evidence_candidates"] = {}
    assert route._normalize_result_envelope(malformed)[1] == []
    budget = payload["plan"]["budget"]

    class RecordedLLM(FakeLLM):
        def __init__(self, mode):
            super().__init__()
            self.model = payload["plan"]["model"]
            self.mode = mode

        def complete_json_budgeted(self, system, messages, maximum, schema=None):
            self.calls.append((system, messages, maximum))
            result = json.loads(json.dumps(original))
            if self.mode != "structure" and len(self.calls) == 1:
                result["episode"]["summary"] = None
            used = recorded["recorded_usage"] if len(self.calls) == 1 else self._usage()
            if self.mode == "exhausted":
                used = {"input_tokens": budget["total_tokens"], "output_tokens": 0}
            return result, used

    env = {name: str(budget[field]) for name, field in (
        ("MATRIX_STOCK_V3_INPUT_TOKENS", "input_tokens"),
        ("MATRIX_STOCK_V3_OUTPUT_TOKENS", "output_tokens"),
        ("MATRIX_STOCK_V3_TOTAL_TOKENS", "total_tokens"),
    )}
    results = {}
    with patch.dict(os.environ, env):
        for mode in ("structure", "repair", "exhausted"):
            llm = RecordedLLM(mode)
            request = Request(llm, payload)
            request.url = SimpleNamespace(path="/api/research/investment.v3")
            response = await route.stock_research(request)
            if mode == "exhausted":
                assert response.status_code == 422 and len(llm.calls) == 1
                details = json.loads(response.body)["details"]
                assert details["repair_status"] == "skipped_budget"
                assert details["repair_attempts"] == 0
                continue
            assert isinstance(response, dict), getattr(response, "status_code", type(response).__name__)
            assert response["model_calls"] == len(llm.calls) == (1 if mode == "structure" else 2)
            assert "result_envelope_normalized" in response["budget_adjustments"]
            assert response["result"]["episode"]["investment_advice"]["valuation"]["status"] == "unknown"
            assert "thesis_patch" not in response["result"]["episode"]
            assert not response["result_incomplete"], response.get("validation_diagnostics")
            if mode == "repair":
                system, messages, maximum = llm.calls[-1]
                repair = json.loads(messages[0]["content"])
                assert 0 < maximum < budget["output_tokens"]
                used = recorded["recorded_usage"]
                bound = route.measure_budget(
                    [system, messages[0]["content"]], overhead_tokens=2048,
                ).input_tokens
                assert bound <= budget["input_tokens"]
                assert used["input_tokens"] + used["output_tokens"] + bound + maximum <= budget["total_tokens"]
                assert "repair_output_reduced_to_remaining_budget" in response["budget_adjustments"]
                assert "repair_evidence_trimmed_to_budget" in response["budget_adjustments"]
                for field in ("thesis", "portfolio_context", "calculated_metrics", "decision_context", "valuation_context"):
                    if field in payload["context"]:
                        assert repair[field] == payload["context"][field], field
                assert repair["documents"] == route._model_document_index(payload["documents"])
                kept = {item["excerpt_id"] for item in repair["evidence_segments"]}
                assert {item["excerpt_id"] for item in normalized["evidence_candidates"]} <= kept
                originals = {item["excerpt_id"]: item["text"]
                             for item in route._evidence_segments(payload["documents"], payload["context"])}
                assert all(originals[item["excerpt_id"]] == item["text"] for item in repair["evidence_segments"])
            results[mode] = {
                "fake_provider_calls": len(llm.calls),
                "last_output_limit": llm.calls[-1][2], "contract_valid": True,
            }
    print(json.dumps({
        "real_model_calls": 0, "recorded_response": results,
        "ambiguous_shapes_rejected": True, "budget_exhaustion_reported_honestly": True,
        "investment_quality_verified": False,
    }))


async def replay_current_response(path: str) -> None:
    """Replay real malformed output without changing its investment opinion."""
    recorded = json.loads(Path(path).read_text())
    payload = recorded["updated_payload"]
    policy = payload["plan"]["decision_context"]
    raw = recorded["raw_result_draft"]
    facts = policy["capital_return_facts"]
    assert len(facts) == 1 and facts[0]["as_of"] == "2026-08-31"
    assert facts[0]["executed_shares"] == "1740000"
    assert facts[0]["scheme_ref"] == "2026-019" and facts[0]["superseded"]
    from matrix.llm.protocol import recover_json_object_response
    recovered = recover_json_object_response(raw)
    assert recovered is not None, "actual redundant closing brace must be recoverable"
    projected = route.project_decision_result(recovered, policy)
    advice = projected["episode"]["investment_advice"]
    assert advice["valuation"]["status"] == "undervalued" and advice["valuation"]["uncertainty"] == "business_uncertainty"
    assert not route.decision_policy_errors(advice, policy)
    assert route._advice_quality_errors(advice, payload["documents"], [], check_metric_prose=False), (
        "recovery must not turn the original stale repurchase claim into a valid result"
    )
    budget = payload["plan"]["budget"]
    env = {name: str(budget[field]) for name, field in (
        ("MATRIX_STOCK_V3_INPUT_TOKENS", "input_tokens"),
        ("MATRIX_STOCK_V3_OUTPUT_TOKENS", "output_tokens"),
        ("MATRIX_STOCK_V3_TOTAL_TOKENS", "total_tokens"),
    )}
    outgoing = []
    client = DeepSeekClient(api_key="fixture-only", model=payload["plan"]["model"])

    def fake_http(url, body, headers, timeout):
        outgoing.append(body)
        request = json.loads(body["messages"][-1]["content"])
        old_source = facts[0]["superseded"][0]["source_id"]
        old_quote = facts[0]["superseded"][0]["quote"]
        source = next(doc["excerpt"] for doc in payload["documents"] if doc["source_id"] == old_source)
        begin, end = source.index(old_quote), source.index(old_quote) + len(old_quote)
        assert not any(
            segment["source_id"] == old_source and segment["excerpt_start"] < end and begin < segment["excerpt_end"]
            for segment in request["evidence_segments"]
        ), "obsolete execution status must not be exposed as current evidence"
        assert any(segment["source_id"] == facts[0]["source_id"] and "1,740,000" in segment["text"]
                   for segment in request["evidence_segments"]), "current execution evidence must survive the input budget"
        used = recorded["recorded_usage"]
        return {
            "choices": [{"finish_reason": recorded["finish_reason"], "message": {"tool_calls": [{
                "function": {"name": "return_json", "arguments": raw},
            }]}}],
            "usage": {"prompt_tokens": used["input_tokens"], "completion_tokens": used["output_tokens"],
                      "total_tokens": used["input_tokens"] + used["output_tokens"]},
        }

    request = Request(client, payload)
    request.url = SimpleNamespace(path="/api/research/investment.v3")
    with patch.dict(os.environ, env), patch("matrix.llm.deepseek.post_json", side_effect=fake_http):
        response = await route.stock_research(request)
    assert response.status_code == 422, "the unchanged stale opinion must still fail"
    failure = json.loads(response.body)
    assert len(outgoing) == failure["details"]["model_calls"] == 1
    assert failure["details"]["repair_attempts"] == 0
    assert "response_syntax_normalized" in failure["details"]["budget_adjustments"]
    assert failure["private_diagnostics"]["response_drafts"][0]["draft"] == raw
    print(json.dumps({
        "real_model_calls": 0, "fake_provider_http_calls": 1,
        "syntax_recovered": True, "known_valuation_with_business_risk_accepted": True,
        "current_event_evidence_prepared": True, "unchanged_stale_opinion_rejected": True,
        "investment_quality_verified": False,
    }))


async def replay_prepared_response(path: str) -> None:
    """Verify first-call inputs from a real saved run, without rewriting that run."""
    recorded = json.loads(Path(path).read_text())
    payload = recorded["updated_payload"]
    policy = payload["plan"]["decision_context"]
    explanations = policy["reported_explanations"]
    assert any(item["kind"] == "issuer_reported_adjustment" and "24.71%" in item["quote"] for item in explanations)
    comparison = next(item for item in policy["comparisons"] if item["id"] == "peer_cash_conversion_ttm")
    assert comparison["status"] == "comparable" and comparison["period"] == "2025-12-31"
    assert comparison["position"] == "higher"
    assert all(
        all(metric["id"] == "pe_reported_ttm" for metric in peer["metrics"])
        for peer in payload["context"]["valuation_context"]["peers"]
    ), "unaligned raw peer cash metrics must not be exposed"
    original_advice = recorded["result"]["episode"]["investment_advice"]
    assert route.comparison_claim_errors(original_advice, policy, "")
    dishonest_ref = json.loads(json.dumps(original_advice))
    dishonest_ref["valuation"].setdefault("calculation_refs", []).append(comparison["id"])
    assert any("direction" in error for error in route.comparison_claim_errors(dishonest_ref, policy, ""))
    untouched = json.dumps(recorded["result"], sort_keys=True)
    budget = payload["plan"]["budget"]
    env = {name: str(budget[field]) for name, field in (
        ("MATRIX_STOCK_V3_INPUT_TOKENS", "input_tokens"),
        ("MATRIX_STOCK_V3_OUTPUT_TOKENS", "output_tokens"),
        ("MATRIX_STOCK_V3_TOTAL_TOKENS", "total_tokens"),
    )}
    class PreparedRecordedLLM(FakeLLM):
        model = payload["plan"]["model"]
        def complete_json_budgeted(self, system, messages, maximum, schema=None):
            self.calls.append((system, messages, maximum))
            request = json.loads(messages[0]["content"])
            evidence = next(item for item in request["evidence_segments"] if "24.71%" in item["text"])
            ref = next(item["evidence_ref"] for item in request["available_evidence_candidates"]
                       if item["excerpt_id"] == evidence["excerpt_id"])
            method = next(item for item in policy["valuation_methods"] if item["status"] == "ready")
            return {
                "episode": {
                    "summary": "离线输入验收，不验证真实投资建议。",
                    "pending_questions": [],
                    "claim_checks": [{"claim_id": claim_id, "status": "unresolved", "finding": "离线验收不校准判断。",
                                     "evidence_refs": []} for claim_id in request["required_claim_ids"]],
                    "conclusion": {"supported": "已取得原文原因。", "unverified": "投资价值仍需评估。", "impact": "不修改真实判断。"},
                    "investment_advice": {
                        "recommendation": "watch", "existing_position": "not_assessed",
                        "reason": "应区分公司披露的调整口径与程序计算，不能跨期推断同业排名。",
                        "conditions": "后续经营证据发生变化时复查。", "review_trigger_ids": ["next_report"],
                        "evidence_refs": [ref],
                        "valuation": {"status": "fair", "method_id": method["id"], "uncertainty": "business_uncertainty",
                                      "rationale": "仅验证协议与输入，不证明投资意见正确。",
                                      "evidence_refs": [ref], "calculation_refs": method["metric_ids"]},
                        "metric_assessments": [{"metric_id": metric_id, "impact": "mixed",
                                               "explanation": "需要结合已披露原因判断现金流变化。"}
                                              for metric_id in policy["required_metric_ids"]],
                    },
                }, "thesis_patch": [], "evidence_candidates": [],
            }, self._usage()
    llm = PreparedRecordedLLM()
    request = Request(llm, payload)
    request.url = SimpleNamespace(path="/api/research/investment.v3")
    with patch.dict(os.environ, env):
        response = await route.stock_research(request)
    assert isinstance(response, dict), getattr(response, "body", response)
    assert len(llm.calls) == response["model_calls"] == 1
    assert json.dumps(recorded["result"], sort_keys=True) == untouched
    print(json.dumps({
        "real_model_calls": 0, "first_call_has_issuer_adjustment": True,
        "common_period_cash_comparison": comparison["period"], "wrong_comparison_rejected": True,
        "fake_provider_calls": 1, "saved_opinion_unchanged": True, "investment_quality_verified": False,
    }))


async def replay_decision_tools(path: str) -> None:
    """Use the actual saved inputs with a fake response, never update the opinion."""
    from matrix.server.routes.stock_decision_tools import decision_analysis_errors
    recorded = json.loads(Path(path).read_text())
    payload = recorded["updated_payload"]
    policy = payload["plan"]["decision_context"]
    original = recorded["result"]["episode"]["investment_advice"]
    assert decision_analysis_errors(original, policy)
    untouched = json.dumps(recorded["result"], sort_keys=True)
    budget = payload["plan"]["budget"]
    env = {name: str(budget[field]) for name, field in (
        ("MATRIX_STOCK_V3_INPUT_TOKENS", "input_tokens"),
        ("MATRIX_STOCK_V3_OUTPUT_TOKENS", "output_tokens"),
        ("MATRIX_STOCK_V3_TOTAL_TOKENS", "total_tokens"),
    )}

    class DecisionReplayLLM(FakeLLM):
        model = payload["plan"]["model"]

        def complete_json_budgeted(self, system, messages, maximum, schema=None):
            self.calls.append((system, messages, maximum))
            value = json.loads(messages[0]["content"])
            assert value["decision_context"]["decision_tools"]["cases"]
            assert any("24.71%" in segment["text"] for segment in value["evidence_segments"])
            assert all("summary" not in item for item in value["decision_context"]["decision_tools"]["cases"])
            assert route.measure_budget([system, messages[0]["content"]], overhead_tokens=2048).input_tokens <= budget["input_tokens"]
            assert "conditions" not in schema["properties"]["episode"]["properties"]["investment_advice"]["properties"]
            evidence = value["available_evidence_candidates"][0]["evidence_ref"]
            advice = json.loads(json.dumps(original))
            advice["evidence_refs"] = [evidence]
            advice["valuation"]["evidence_refs"] = [evidence]
            adapt_v2_fixture(advice, policy)
            return {
                "episode": {
                    "summary": "真实输入的离线协议回放，不代表真实模型生成了新的建议。",
                    "pending_questions": [],
                    "conclusion": {"supported": "压力测算输入可追溯。", "unverified": "未来盈利仍有不确定性。",
                                   "impact": "离线验收不改变真实投资意见。"},
                    "claim_checks": [
                        {"claim_id": item, "status": "unresolved", "finding": "离线验收不改变真实判断。",
                         "evidence_refs": []} for item in value["required_claim_ids"]
                    ],
                    "investment_advice": advice,
                },
                "thesis_patch": [], "evidence_candidates": [],
            }, self._usage()

    llm = DecisionReplayLLM()
    request = Request(llm, payload)
    request.url = SimpleNamespace(path="/api/research/investment.v3")
    with patch.dict(os.environ, env):
        response = await route.stock_research(request)
    if not isinstance(response, dict):
        raise AssertionError(json.loads(response.body).get("details", {}).get("validation_errors"))
    assert len(llm.calls) == response["model_calls"] == 1
    assert json.dumps(recorded["result"], sort_keys=True) == untouched
    print(json.dumps({
        "real_model_calls": 0, "fake_calls": 1, "actual_saved_inputs": True,
        "old_generic_advice_requires_decision_analysis": True,
        "decision_cases": len(policy["decision_tools"]["cases"]),
        "confirmed_input_budget_preserved": True, "saved_opinion_unchanged": True,
        "investment_quality_verified": False,
    }))


async def prepare_quarantined_recovery(path: str, output: str) -> None:
    """Revalidate the exact paid response in-process; no provider or asset writes."""
    recorded = json.loads(Path(path).read_text())
    payload = recorded["payload"]
    from matrix.llm.protocol import parse_json_object_response
    original = parse_json_object_response(recorded["raw_result_draft"])
    budget = payload["plan"]["budget"]
    used = recorded["recorded_usage"]

    class QuarantinedLLM(FakeLLM):
        model = payload["plan"]["model"]

        def complete_json_budgeted(self, system, messages, maximum, schema=None):
            self.calls.append((system, messages, maximum))
            begin = system.index('{"episode":', system.index("The investment-research-v3 result contract is strict."))
            example = json.loads(system[begin:system.index(". episode.summary", begin)])["episode"]["investment_advice"]
            advice_schema = schema["properties"]["episode"]["properties"]["investment_advice"]
            assert set(example) == set(advice_schema["required"])
            assert "conditions" not in example and "decision_analysis" in example
            return json.loads(json.dumps(original)), used

    env = {name: str(budget[field]) for name, field in (
        ("MATRIX_STOCK_V3_INPUT_TOKENS", "input_tokens"),
        ("MATRIX_STOCK_V3_OUTPUT_TOKENS", "output_tokens"),
        ("MATRIX_STOCK_V3_TOTAL_TOKENS", "total_tokens"),
    )}
    llm = QuarantinedLLM()
    request = Request(llm, payload)
    request.url = SimpleNamespace(path="/api/research/investment.v3")
    with patch.dict(os.environ, env):
        response = await route.stock_research(request)
    if not isinstance(response, dict):
        raise AssertionError(json.loads(response.body).get("details", {}).get("validation_errors"))
    assert len(llm.calls) == response["model_calls"] == 1
    response.pop("private_diagnostics", None)
    response["usage"] = {"call_1": used}
    target = Path(output)
    target.touch(mode=0o600, exist_ok=True)
    target.chmod(0o600)
    target.write_text(json.dumps({
        "version": "stock-response-recovery-v1",
        "run_id": recorded["run_id"], "diagnostics_hash": recorded["diagnostics_hash"],
        "draft_hash": recorded["draft_hash"], "response": response,
    }, ensure_ascii=False))
    print(json.dumps({
        "additional_real_model_calls": 0, "preserved_model_calls": 1,
        "full_agent_validation_passed": True, "candidate": str(target),
    }))


if __name__ == "__main__":
    if sys.argv[1:] == ["--serve"]:
        serve()
    elif sys.argv[1:2] == ["--recorded-run"]:
        replay_recorded_evidence(sys.argv[2])
    elif sys.argv[1:2] == ["--recorded-response"]:
        asyncio.run(replay_recorded_response(sys.argv[2]))
    elif sys.argv[1:2] == ["--recorded-current-response"]:
        asyncio.run(replay_current_response(sys.argv[2]))
    elif sys.argv[1:2] == ["--recorded-prepared-response"]:
        asyncio.run(replay_prepared_response(sys.argv[2]))
    elif sys.argv[1:2] == ["--recorded-decision-tools"]:
        asyncio.run(replay_decision_tools(sys.argv[2]))
    elif sys.argv[1:2] == ["--prepare-quarantined-recovery"]:
        asyncio.run(prepare_quarantined_recovery(sys.argv[2], sys.argv[3]))
    else:
        asyncio.run(main())
