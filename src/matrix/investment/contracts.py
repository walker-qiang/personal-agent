"""Consume the personal-os-owned contract; never duplicate financial formulas."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

CONTRACT_VERSION = "investment-research-core-v1"
SCHEMA_ID = "urn:personal-os:investment-research:core:v1"
RECORD_KINDS = frozenset({
    "Instrument", "Evidence", "Observation", "ToolSpec", "Calculation",
    "Permission", "Plan", "Result", "Event",
    "Publisher", "Candidate", "MethodSpec", "PreparationRequest", "Attempt", "Preparation",
    "Answer", "AnalysisDraft", "ResearchReplayRequest", "AnalysisContext",
    "MetadataClaim",
    "ResearchRequest", "ResearchPlanSnapshot", "ResearchRun", "ResearchTimePolicy", "ResearchWindow",
    "ValuationSubject", "ValuationSeries", "ValuationPolicy", "ScenarioAssumption", "ValuationContext",
})


@dataclass(frozen=True)
class ResearchContract:
    """Construct only from a trusted API bundle pinned by its capability digest."""

    schema_json: str
    digest: str

    @classmethod
    def from_bundle(cls, bundle: dict[str, Any], *, expected_digest: str) -> ResearchContract:
        if not isinstance(bundle, dict) or set(bundle) != {"version", "sha256", "schema_json"}:
            raise ValueError("contract bundle must contain version, sha256 and schema_json")
        if bundle["version"] != CONTRACT_VERSION or not isinstance(bundle["schema_json"], str):
            raise ValueError("unsupported investment research contract version or schema encoding")
        digest = hashlib.sha256(bundle["schema_json"].encode("utf-8")).hexdigest()
        if digest != bundle["sha256"] or digest != expected_digest:
            raise ValueError("contract digest differs from the trusted capability digest")
        schema = json.loads(bundle["schema_json"])
        if not isinstance(schema, dict) or schema.get("$id") != SCHEMA_ID or schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            raise ValueError("contract identity or JSON Schema dialect is unsupported")
        _local_references_only(schema, root=True)
        if not {"date-time", "date", "uri"} <= FormatChecker().checkers.keys():
            raise ValueError("required date/URI validators are unavailable; install the core contract dependencies")
        Draft202012Validator.check_schema(schema)
        if not RECORD_KINDS <= schema.get("$defs", {}).keys():
            raise ValueError("contract omits required research record definitions")
        return cls(bundle["schema_json"], digest)

    def validate(self, kind: str, value: Any) -> None:
        if kind not in RECORD_KINDS:
            raise ValueError(f"unknown research contract kind: {kind}")
        schema = json.loads(self.schema_json)
        schema["$ref"] = f"#/$defs/{kind}"
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        errors = sorted(validator.iter_errors(value), key=lambda error: str(list(error.path)))
        if errors:
            first = errors[0]
            location = ".".join(str(part) for part in first.path) or "<record>"
            raise ValueError(f"{kind}.{location}: {first.message}")


def _local_references_only(value: Any, *, root: bool = False) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "$id" and not root:
                raise ValueError("nested schema identities are disabled")
            if key in ("$ref", "$dynamicRef") and (not isinstance(item, str) or not item.startswith("#/$defs/")):
                raise ValueError("external schema references are disabled")
            _local_references_only(item)
    elif isinstance(value, list):
        for item in value:
            _local_references_only(item)
