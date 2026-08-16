"""实用 MCP 服务器：时间查询、数学计算、字符串工具。

启动方式：python utility_tools.py
"""
from mcp.server.fastmcp import FastMCP
from datetime import datetime, timezone, timedelta
import math

mcp = FastMCP("utility-tools")


@mcp.tool()
def current_time(timezone_offset: int = 8) -> dict:
    """获取当前时间，支持时区偏移。

    Args:
        timezone_offset: UTC 时区偏移小时数（默认 8 表示北京时间）

    Returns:
        包含日期、时间、星期、时区信息的字典
    """
    now = datetime.utcnow()
    local = now.replace(tzinfo=timezone.utc) + timedelta(hours=timezone_offset)
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return {
        "date": local.strftime("%Y-%m-%d"),
        "time": local.strftime("%H:%M:%S"),
        "weekday": weekdays[local.weekday()],
        "timezone": f"UTC+{timezone_offset}",
        "timestamp": int(local.timestamp()),
    }


@mcp.tool()
def calculate(expression: str) -> dict:
    """安全计算数学表达式。

    支持运算符：+ - * / ** %
    支持函数：sqrt, sin, cos, tan, log, log10, abs, round, pow, max, min
    支持常量：pi, e

    Args:
        expression: 数学表达式，如 "2 + 3 * 4" 或 "sqrt(16)" 或 "sin(pi/2)"

    Returns:
        包含表达式和计算结果的字典
    """
    allowed_names = {
        k: v for k, v in math.__dict__.items() if not k.startswith("_")
    }
    allowed_names.update({
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e, "log": math.log, "log10": math.log10,
        "abs": abs, "round": round, "pow": pow, "max": max, "min": min,
    })
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"expression": expression, "error": str(exc)}


@mcp.tool()
def string_reverse(text: str) -> dict:
    """反转字符串。

    Args:
        text: 要反转的字符串

    Returns:
        包含原始字符串和反转结果的字典
    """
    return {"original": text, "reversed": text[::-1]}


@mcp.tool()
def word_count(text: str) -> dict:
    """统计文本的字数、行数、字符数。

    Args:
        text: 要统计的文本

    Returns:
        包含字符数、单词数、行数的字典
    """
    lines = text.split("\n") if text else []
    words = text.split() if text else []
    return {
        "chars": len(text),
        "chars_no_spaces": len(text.replace(" ", "")),
        "words": len(words),
        "lines": len(lines),
    }


if __name__ == "__main__":
    mcp.run()
