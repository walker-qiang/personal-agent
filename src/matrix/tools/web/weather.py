"""Weather tool backed by Open-Meteo (free, no API key)."""

from __future__ import annotations

import urllib.parse
import urllib.request
from typing import Any
import json

from ..base import ToolDefinition, tool_error

tool_definition = ToolDefinition(
    name="weather",
    description="查询指定城市天气（实时 + 未来几天预报）。用于：用户问「今天天气」「明天会下雨吗」「某地多少度」等。城市名支持中英文。",
    capabilities=["weather"],
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

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def weather(city: str, days: int = 1) -> dict[str, Any]:
    """Query current weather and daily forecasts for a city."""
    days = min(max(days, 1), 3)

    try:
        latitude, longitude = _resolve_city(city)
        data = _fetch_json(
            _FORECAST_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": ",".join(
                    [
                        "temperature_2m",
                        "relative_humidity_2m",
                        "apparent_temperature",
                        "weather_code",
                        "wind_speed_10m",
                    ]
                ),
                "daily": "weather_code,temperature_2m_max,temperature_2m_min",
                "timezone": "auto",
                "forecast_days": days,
            },
        )
    except Exception:
        return tool_error(
            "weather_get_current",
            "获取天气",
            "获取天气数据失败",
            "请检查城市名称是否正确，或稍后重试。",
            {"city": city},
        )

    current = data.get("current") or {}
    daily = data.get("daily") or {}
    daily_dates = daily.get("time") or []
    daily_codes = daily.get("weather_code") or []
    daily_max = daily.get("temperature_2m_max") or []
    daily_min = daily.get("temperature_2m_min") or []

    result = {
        "city": city,
        "current": {
            "temp_c": _format_number(current.get("temperature_2m")),
            "weather": _weather_label(current.get("weather_code")),
            "humidity": _format_number(current.get("relative_humidity_2m")),
            "wind_speed_kmh": _format_number(current.get("wind_speed_10m")),
            "feels_like_c": _format_number(current.get("apparent_temperature")),
        },
        "forecast": [],
    }

    for index, date in enumerate(daily_dates[:days]):
        result["forecast"].append({
            "date": date,
            "max_temp_c": _format_number(_at(daily_max, index)),
            "min_temp_c": _format_number(_at(daily_min, index)),
            "weather": _weather_label(_at(daily_codes, index)),
        })

    return result


def _fetch_json(base_url: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{base_url}?{query}",
        headers={"User-Agent": _UA, "Accept-Language": "zh-CN,zh"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _resolve_city(city: str) -> tuple[float, float]:
    aliases = {
        "深圳": (22.5431, 114.0579),
        "shenzhen": (22.5431, 114.0579),
        "北京": (39.9042, 116.4074),
        "beijing": (39.9042, 116.4074),
        "上海": (31.2304, 121.4737),
        "shanghai": (31.2304, 121.4737),
        "广州": (23.1291, 113.2644),
        "guangzhou": (23.1291, 113.2644),
    }
    normalized = city.strip().lower()
    if normalized in aliases:
        return aliases[normalized]

    data = _fetch_json(
        _GEOCODING_URL,
        {"name": city, "count": 1, "language": "zh", "format": "json"},
    )
    locations = data.get("results") or []
    if not locations:
        raise ValueError(f"city not found: {city}")
    return float(locations[0]["latitude"]), float(locations[0]["longitude"])


def _at(values: list[Any], index: int) -> Any:
    return values[index] if index < len(values) else None


def _format_number(value: Any) -> str:
    try:
        return str(round(float(value)))
    except (TypeError, ValueError):
        return "?"


def _weather_label(code: Any) -> str:
    try:
        weather_code = int(code)
    except (TypeError, ValueError):
        return "未知"

    if weather_code == 0:
        return "晴"
    if weather_code == 1:
        return "晴"
    if weather_code == 2:
        return "多云"
    if weather_code == 3:
        return "阴天"
    if weather_code in (45, 48):
        return "雾"
    if weather_code in (51, 53, 55, 56, 57, 61, 66, 80):
        return "小雨"
    if weather_code in (63, 81):
        return "中雨"
    if weather_code in (65, 67, 82):
        return "大雨"
    if weather_code in (71, 73, 75, 77, 85, 86):
        return "下雪"
    if weather_code in (95, 96, 99):
        return "雷阵雨"
    return "未知"
