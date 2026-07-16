"""weather tool — query current weather via wttr.in (free, no API key)."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from ..base import ToolDefinition

tool_definition = ToolDefinition(
    name="weather",
    description="查询指定城市的实时天气状况和未来几天预报。用于回答天气相关问题。",
    input_schema={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称，英文（如 Shenzhen、Beijing）或中文（如 深圳、北京）均可",
            },
            "days": {
                "type": "integer",
                "description": "预报天数，默认 1（仅今天），最大 3",
                "default": 1,
            },
        },
        "required": ["city"],
    },
    handler=None,
)

_BASE_URL = "https://wttr.in"
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def weather(city: str, days: int = 1) -> dict[str, Any]:
    """Query weather for a city using wttr.in."""
    days = min(max(days, 1), 3)

    # Try JSON API first for structured data
    encoded_city = urllib.parse.quote(city, safe="")
    json_url = f"{_BASE_URL}/{encoded_city}?format=j1"
    req = urllib.request.Request(json_url, headers={"User-Agent": _UA, "Accept-Language": "zh-CN,zh"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {"error": f"获取天气失败，请检查城市名称是否正确", "city": city}

    current = (data.get("current_condition") or [{}])[0]
    forecast_days = data.get("weather", [])[:days]

    result = {
        "city": city,
        "current": {
            "temp_c": current.get("temp_C", "?"),
            "weather": (current.get("weatherDesc") or [{}])[0].get("value", "?"),
            "humidity": current.get("humidity", "?"),
            "wind_speed_kmh": current.get("windspeedKmph", "?"),
            "feels_like_c": current.get("FeelsLikeC", "?"),
        },
        "forecast": [],
    }

    for day in forecast_days:
        result["forecast"].append({
            "date": day.get("date", "?"),
            "max_temp_c": day.get("maxtempC", "?"),
            "min_temp_c": day.get("mintempC", "?"),
            "weather": (day.get("hourly", [{}])[4].get("weatherDesc") or [{}])[0].get("value", "?"),
        })

    return result