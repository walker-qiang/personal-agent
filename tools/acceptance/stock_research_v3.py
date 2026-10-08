#!/usr/bin/env python3
"""Exercise the stock v3 application protocol with a local fake LLM."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi import FastAPI
from matrix.server.routes import stock_research as route
import uvicorn


class FakeLLM:
    model = "mock-budgeted"

    def __init__(self):
        self.calls = []
        self.repair = False

    def complete_json_budgeted(self, system, messages, maximum):
        self.calls.append((system, messages, maximum))
        if self.repair and len(self.calls) == 1:
            return {"episode": {}}, {"input_tokens": 150, "output_tokens": 20}
        return {"episode": {"pending_questions": []},
                "thesis_patch": [], "evidence_candidates": []}, {
                    "input_tokens": 150, "output_tokens": 20,
                }


class IntegrationLLM(FakeLLM):
    def complete_json_budgeted(self, system, messages, maximum):
        payload = json.loads(messages[0]["content"])
        if "request" in payload:
            payload = payload["request"]
        plan = payload.get("plan", {})
        context = payload.get("context", {})
        documents = payload.get("documents", [])
        if not plan:
            plan = {
                "plan_id": payload.get("question", "facts-only"),
                "question": payload["question"],
                "facts_only": True,
                "calculated_metrics": payload.get("calculated_metrics", []),
            }
            context = {
                "question": payload["question"],
                "subject": payload.get("subject", {}),
                "calculated_metrics": payload.get("calculated_metrics", []),
                "pending_questions": payload.get("pending_questions", []),
            }
        self.calls.append(plan["plan_id"])
        if not documents:
            return {"episode": {"pending_questions": []},
                    "thesis_patch": [], "evidence_candidates": []}, {
                        "input_tokens": 150, "output_tokens": 20}
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
        quote = "The consideration was paid in cash."
        assert quote in document["excerpt"]
        claims = context["thesis"]["claims"]
        patch = [{
            "entity": "claim", "id": claims[0]["claim_id"], "field": "confidence",
            "before": "low", "after": "medium", "change_type": "strengthened",
            "reason": "The official excerpt was checked against its receipt.",
            "evidence_refs": [document["source_id"] + ":fact-a"],
        }] if claims else []
        if plan["question"] == "partial" and patch:
            patch[0]["evidence_refs"] = ["forged:fact-a"]
        if plan["question"] == "repair" and self.calls.count(plan["plan_id"]) == 1:
            return {"episode": {}}, {"input_tokens": 150, "output_tokens": 20}
        return {
            "episode": {"pending_questions": ["Can the cash payment be reconciled?"]},
            "thesis_patch": patch,
            "evidence_candidates": [{"source_id": document["source_id"], "fact_id": "fact-a",
                                     "limited_quote": quote, "statement": quote}],
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

    async def json(self):
        return self.payload


async def main():
    for name, value in (
        ("MATRIX_STOCK_V3_INPUT_TOKENS", "25000"),
        ("MATRIX_STOCK_V3_OUTPUT_TOKENS", "1500"),
        ("MATRIX_STOCK_V3_TOTAL_TOKENS", "40000"),
    ):
        os.environ[name] = value
    budget = {"input_tokens": 25000, "output_tokens": 1500, "total_tokens": 40000}
    plan = {"plan_id": "fixture", "object_type": "stock", "model": "mock-budgeted",
            "confirmed_at": "2026-10-06T00:00:00Z", "input_hash": "fixture-hash",
            "question": "What changed?", "budget": budget, "documents": []}
    context = {"question": plan["question"], "thesis": {"revision": 1}}
    payload = {"plan": plan, "context": context, "documents": []}
    llm = FakeLLM()
    result = await route.stock_research(Request(llm, payload))
    assert result["model_calls"] == 1 and result["usage_complete"] and len(llm.calls) == 1
    llm = FakeLLM()
    llm.repair = True
    result = await route.stock_research(Request(llm, payload))
    assert result["model_calls"] == 2 and len(llm.calls) == 2
    llm = FakeLLM()
    payload["documents"] = [{"source_id": "unsanctioned", "permissions": {
        "external_model_excerpt": True}, "excerpt": "text", "content_hash": "sha256"}]
    result = await route.stock_research(Request(llm, payload))
    assert result.status_code == 422 and not llm.calls
    payload["documents"] = []
    plan["facts_only"] = True
    plan["calculated_metrics"] = [{"id": "fixture:net_profit_yoy_pct", "value": "13.02"}]
    payload["context"] = {"question": plan["question"], "subject": {"name": "Fixture"},
                          "calculated_metrics": plan["calculated_metrics"]}
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
    os.environ["MATRIX_STOCK_V3_TOTAL_TOKENS"] = "0"
    result = await route.stock_research(Request(llm, payload))
    assert result.status_code == 503 and not llm.calls
    print(json.dumps({"status": "passed", "checks": [
        "one synthesis", "one bounded repair", "immutable documents",
        "immutable API calculations", "private context/excerpt blocked",
        "missing budget fails closed"]}))


if __name__ == "__main__":
    if sys.argv[1:] == ["--serve"]:
        serve()
    else:
        asyncio.run(main())
