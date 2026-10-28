"""Read prepared OS inputs without re-running retrieval or financial formulas."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .contracts import ResearchContract


@dataclass(frozen=True)
class ResearchCapabilities:
    contract: ResearchContract
    payload_json: str

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], *, expected_contract_digest: str,
    ) -> ResearchCapabilities:
        if not isinstance(payload, dict) or not isinstance(payload.get("engine_version"), str) or "contract" not in payload:
            raise ValueError("capabilities require a trusted contract bundle and engine version")
        contract = ResearchContract.from_bundle(
            payload["contract"], expected_digest=expected_contract_digest,
        )
        methods, tools = payload.get("methods"), payload.get("tool_index")
        if not isinstance(methods, list) or not isinstance(tools, list):
            raise ValueError("capabilities require method and compact tool registries")
        method_ids, tool_ids = set(), set()
        for method in methods:
            contract.validate("MethodSpec", method)
            if method["method_id"] in method_ids:
                raise ValueError("capabilities contain a duplicate method identity")
            method_ids.add(method["method_id"])
        for tool in tools:
            contract.validate("ToolSpec", tool)
            if tool["tool_id"] in tool_ids or not set(tool["method_ids"]) <= method_ids:
                raise ValueError("tool registry contains duplicate IDs or unknown methods")
            tool_ids.add(tool["tool_id"])
        return cls(contract, json.dumps(payload, ensure_ascii=False))

    def tool_index(self, method_id: str) -> list[dict[str, str]]:
        """Only this small index belongs in an eventual model planning prompt."""
        payload = json.loads(self.payload_json)
        return [
            {"tool_id": item["tool_id"], "version": item["version"], "summary": item["summary"]}
            for item in payload["tool_index"] if method_id in item["method_ids"]
        ]


@dataclass(frozen=True)
class PreparedResearchInputs:
    capabilities: ResearchCapabilities
    payload_json: str

    @classmethod
    def from_payload(
        cls, capabilities: ResearchCapabilities, payload: dict[str, Any],
        *, expected_preparation_id: str, expected_instrument_id: str,
    ) -> PreparedResearchInputs:
        capabilities.contract.validate("Preparation", payload)
        if payload["contract_digest"] != capabilities.contract.digest:
            raise ValueError("preparation contract digest differs from trusted capabilities")
        if payload["engine_version"] != json.loads(capabilities.payload_json)["engine_version"]:
            raise ValueError("preparation engine version differs from trusted capabilities")
        if payload["preparation_id"] != expected_preparation_id or payload["instrument_id"] != expected_instrument_id:
            raise ValueError("preparation result belongs to another operation or instrument")
        methods = {
            item["method_id"]: item
            for item in json.loads(capabilities.payload_json)["methods"]
        }
        method_id = payload["method_id"]
        if method_id and (method_id not in methods or methods[method_id]["version"] != payload["method_version"]):
            raise ValueError("preparation method version differs from trusted capabilities")
        if payload["status"] == "ready" and not method_id:
            raise ValueError("ready preparation requires a selected supported method")
        return cls(capabilities, json.dumps(payload, ensure_ascii=False))

    def snapshot(self) -> dict[str, Any]:
        return json.loads(self.payload_json)

    @property
    def can_start_analysis(self) -> bool:
        payload = self.snapshot()
        methods = {
            item["method_id"]: item
            for item in json.loads(self.capabilities.payload_json)["methods"]
        }
        return (
            payload["source_mode"] == "public_retrieval"
            and payload["status"] == "ready"
            and payload["analysis_ready"] is True
            and methods.get(payload["method_id"], {}).get("analysis_ready") is True
        )

    def require_analysis_enabled(self) -> None:
        if not self.can_start_analysis:
            raise ValueError(
                "preparation is not an enabled investment analysis: inspect blocking gaps, "
                "source mode and the method release gate; do not call a model or fabricate advice"
            )
