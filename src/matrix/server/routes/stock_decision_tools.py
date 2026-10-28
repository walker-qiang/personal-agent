"""Validate program-calculated sensitivities and evidence-linked decision coverage."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

TOOLS_VERSION = "decision-tools-v1"
_TOOLS = {"earnings_stress", "dividend_stress", "peer_parity"}


def decision_analysis_text(advice: dict[str, Any]) -> str:
    analysis = advice.get("decision_analysis")
    if not isinstance(analysis, dict):
        return ""
    parts = [analysis.get("current_price_reasoning"), analysis.get("unresolved_premise")]
    assessments = analysis.get("case_assessments")
    if isinstance(assessments, list):
        parts.extend(item.get("implication") for item in assessments if isinstance(item, dict))
    return " ".join(part for part in parts if isinstance(part, str))


def _number(value: Any) -> Decimal:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ValueError("decision tool numbers must be finite decimals")
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("decision tool numbers must be finite decimals") from None
    if not number.is_finite() or abs(number) > Decimal("1e30"):
        raise ValueError("decision tool number is outside supported bounds")
    return number


def validate_decision_tools(
    policy: dict[str, Any], valuation: dict[str, Any] | None, market: dict[str, Any] | None,
) -> None:
    tools = policy.get("decision_tools")
    if tools is None:
        return
    if (not isinstance(tools, dict) or set(tools) != {"version", "cases", "conditions", "unavailable"}
            or tools.get("version") != TOOLS_VERSION
            or not isinstance(tools.get("cases"), list) or len(tools["cases"]) > 7
            or not isinstance(tools.get("conditions"), list) or len(tools["conditions"]) > 10
            or not isinstance(tools.get("unavailable"), dict)
            or set(tools["unavailable"]) - _TOOLS
            or any(not isinstance(reason, str) or not reason or len(reason) > 80
                   for reason in tools["unavailable"].values())):
        raise ValueError("decision tools require a bounded supported versioned index")
    metrics = {item["id"]: item for item in (valuation or {}).get("metrics", [])}
    currency = (valuation or {}).get("currency")

    def metric_value(key: str, unit: str) -> Decimal | None:
        item = metrics.get(key, {})
        if item.get("status") != "computed" or item.get("unit") != unit:
            return None
        if unit == "currency_per_share" and item.get("currency") != currency:
            return None
        number = _number(item.get("value"))
        return number if number > 0 else None

    expected: dict[str, tuple[str, str, list[str], dict[str, str], dict[str, Decimal]]] = {}
    condition_values: dict[str, tuple[str, str, str, Decimal, Decimal]] = {}
    with localcontext() as context:
        context.prec = 100
        quote = _number((market or {}).get("price", "0")) if (market or {}).get("status") == "available" else Decimal(0)
        if (market or {}).get("status") == "available" and quote > 0 and currency in ("CNY", "HKD", "USD"):
            eps = metric_value("reported_eps_ttm", "currency_per_share")
            dividend = metric_value("cash_dividend_per_share_12m", "currency_per_share")
            for cut in (10, 20, 30):
                retained = Decimal(100 - cut) / 100
                if eps is not None:
                    key, stressed = f"earnings_down_{cut}", eps * retained
                    expected[key] = (
                        "earnings_stress", metrics["reported_eps_ttm"]["period"],
                        ["quote_price", "reported_eps_ttm"], {"earnings_change_pct": str(-cut)},
                        {"eps": stressed, "pe": quote / stressed, "earnings_yield_pct": stressed / quote * 100},
                    )
                    condition_values[key + ":eps"] = (key, "reported_eps_ttm", "lte", stressed, eps)
                if dividend is not None:
                    key, stressed = f"dividend_down_{cut}", dividend * retained
                    price = quote * retained
                    expected[key] = (
                        "dividend_stress", metrics["cash_dividend_per_share_12m"]["period"],
                        ["quote_price", "cash_dividend_per_share_12m"], {"dividend_change_pct": str(-cut)},
                        {"dividend_per_share": stressed, "dividend_yield_pct": stressed / quote * 100,
                         "price_at_baseline_yield": price},
                    )
                    condition_values[key + ":dividend"] = (
                        key, "cash_dividend_per_share_12m", "lte", stressed, dividend,
                    )
                    condition_values[key + ":price"] = (key, "quote_price", "lte", price, quote)
            target = metric_value("peer_target_pe", "ratio")
            median = metric_value("peer_pe_median", "ratio")
            if (target is not None and median is not None
                    and metrics["peer_target_pe"].get("period")
                    and metrics["peer_target_pe"]["period"] == metrics["peer_pe_median"].get("period")):
                retained = target / median
                key = "peer_earnings_parity"
                expected[key] = (
                    "peer_parity", metrics["peer_target_pe"]["period"],
                    ["quote_price", "peer_target_pe", "peer_pe_median"],
                    {"peer_multiple": "fixed", "earnings_basis": "common_report_period"},
                    {"earnings_change_to_parity_pct": (retained - 1) * 100,
                     "price_at_peer_multiple": quote / retained},
                )
                if target != median:
                    condition_values[key + ":multiple"] = (
                        key, "pe_to_peer_median", "gte" if target < median else "lte", Decimal(1), retained,
                    )
        seen: set[str] = set()
        for case in tools["cases"]:
            if (not isinstance(case, dict) or set(case) != {
                "id", "tool_id", "period", "input_ids", "assumptions", "values", "summary",
            } or not isinstance(case.get("id"), str) or case["id"] not in expected or case["id"] in seen):
                raise ValueError("decision cases must identify supported sensitivity formulas")
            tool, period, inputs, assumptions, values = expected[case["id"]]
            if (case["tool_id"] != tool or case["period"] != period or case["input_ids"] != inputs
                    or case["assumptions"] != assumptions or not isinstance(case["values"], dict)
                    or set(case["values"]) != set(values)
                    or not isinstance(case["summary"], str) or not 1 <= len(case["summary"]) <= 180):
                raise ValueError("decision case basis, inputs or assumptions differ from its formula")
            for key, value in values.items():
                tolerance = Decimal("0.00005") if key in {
                    "pe", "earnings_yield_pct", "dividend_yield_pct", "earnings_change_to_parity_pct",
                } else Decimal("0.000000005")
                if abs(_number(case["values"][key]) - value) > tolerance:
                    raise ValueError("decision case result differs from supplied inputs: " + key)
            seen.add(case["id"])
        if seen != set(expected):
            raise ValueError("decision cases omit available calculated sensitivities")
        available = {case[0] for case in expected.values()}
        if set(tools["unavailable"]) != _TOOLS - available:
            raise ValueError("decision tool availability differs from supplied inputs")
        seen = set()
        for condition in tools["conditions"]:
            if (not isinstance(condition, dict) or set(condition) != {
                "id", "case_id", "metric_id", "operator", "threshold", "baseline", "label",
            } or not isinstance(condition.get("id"), str)
                    or condition["id"] not in condition_values or condition["id"] in seen
                    or not isinstance(condition["label"], str) or not 1 <= len(condition["label"]) <= 100):
                raise ValueError("decision conditions must use calculated case thresholds")
            case_id, metric, operator, threshold, baseline = condition_values[condition["id"]]
            if (condition["case_id"] != case_id or condition["metric_id"] != metric or condition["operator"] != operator
                    or abs(_number(condition["threshold"]) - threshold) > Decimal("0.000000005")
                    or abs(_number(condition["baseline"]) - baseline) > Decimal("0.000000005")):
                raise ValueError("decision condition is inconsistent with its source case")
            seen.add(condition["id"])
        if seen != set(condition_values):
            raise ValueError("decision conditions omit supplied case thresholds")


def decision_condition_text(policy: dict[str, Any], ids: Any) -> str:
    tools = policy.get("decision_tools", {})
    if (not isinstance(ids, list) or len(ids) > 2 or any(not isinstance(item, str) for item in ids)
            or len(ids) != len(set(ids))):
        return ""
    known = {item["id"]: item["label"] for item in tools.get("conditions", [])}
    if any(item not in known for item in ids):
        return ""
    if not ids:
        return "" if known else "补齐可核验的估值输入后重新评估，不据缺失资料触发交易。"
    return "；".join(known[item] for item in ids) + "。"


def decision_analysis_errors(advice: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    tools = policy.get("decision_tools")
    if tools is None:
        return []
    analysis = advice.get("decision_analysis")
    if (not isinstance(analysis, dict) or set(analysis) != {
        "case_assessments", "current_price_reasoning", "condition_ids", "unresolved_premise",
    } or not isinstance(analysis.get("current_price_reasoning"), str)
            or len(analysis["current_price_reasoning"].strip()) < 12
            or len(analysis["current_price_reasoning"]) > 180
            or not isinstance(analysis.get("unresolved_premise"), str)
            or len(analysis["unresolved_premise"]) > 180
            or not isinstance(analysis.get("case_assessments"), list)
            or len(analysis["case_assessments"]) > 2):
        return ["decision output requires current-price reasoning and bounded scenario assessments"]
    known = {item["id"]: item for item in tools["cases"]}
    available = {item["tool_id"] for item in tools["cases"]}
    seen: set[str] = set()
    selected: set[str] = set()
    for item in analysis["case_assessments"]:
        if (not isinstance(item, dict) or set(item) != {"case_id", "implication"}
                or not isinstance(item.get("case_id"), str) or item["case_id"] not in known
                or item["case_id"] in seen or not isinstance(item.get("implication"), str)
                or not 8 <= len(item["implication"].strip()) <= 120):
            return ["decision assessments must explain distinct supplied cases without invented calculations"]
        seen.add(item["case_id"])
        selected.add(known[item["case_id"]]["tool_id"])
    errors = []
    if len(selected) < min(2, len(available)):
        errors.append("decision must weigh two available atomic tools, or all tools when fewer are available")
    conditions = decision_condition_text(policy, analysis["condition_ids"])
    if not conditions or advice.get("conditions") != conditions:
        errors.append("decision conditions are program-owned calculated thresholds")
    if isinstance(analysis["condition_ids"], list):
        for condition in tools["conditions"]:
            if condition["id"] in analysis["condition_ids"] and condition["case_id"] not in seen:
                errors.append("decision conditions must belong to an assessed scenario")
    if (advice.get("valuation", {}).get("status") == "unknown"
            and len(analysis["unresolved_premise"].strip()) < 12):
        errors.append("unknown valuation requires a specific unresolved investment premise, not a generic risk label")
    return errors


def apply_decision_analysis_schema(advice: dict[str, Any], policy: dict[str, Any]) -> None:
    tools = policy.get("decision_tools")
    if tools is None:
        return
    advice["properties"].pop("conditions")
    advice["required"].remove("conditions")
    cases = [item["id"] for item in tools["cases"]]
    conditions = [item["id"] for item in tools["conditions"]]
    advice["required"].append("decision_analysis")
    advice["properties"]["decision_analysis"] = {
        "type": "object", "additionalProperties": False,
        "required": ["case_assessments", "current_price_reasoning", "condition_ids", "unresolved_premise"],
        "properties": {
            "case_assessments": {
                "type": "array", "minItems": min(2, len({item["tool_id"] for item in tools["cases"]})),
                "maxItems": min(2, len(cases)),
                "items": {"type": "object", "additionalProperties": False,
                          "required": ["case_id", "implication"], "properties": {
                              "case_id": {"type": "string", **({"enum": cases} if cases else {})},
                              "implication": {"type": "string", "minLength": 8, "maxLength": 120},
                          }},
            },
            "current_price_reasoning": {"type": "string", "minLength": 12, "maxLength": 180},
            "condition_ids": {"type": "array", "minItems": min(1, len(conditions)),
                              "maxItems": min(2, len(conditions)), "uniqueItems": True,
                              "items": {"type": "string", **({"enum": conditions} if conditions else {})}},
            "unresolved_premise": {"type": "string", "maxLength": 180},
        },
    }
