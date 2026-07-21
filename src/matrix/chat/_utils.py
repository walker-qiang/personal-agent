"""Chat service utilities: SSE helpers, JSON preview, memory prompt."""

from __future__ import annotations

import json
import queue
import time
from typing import Any, Iterator

MEMORY_EXTRACTION_PROMPT = """从以下对话中提取用户的关键信息，以 JSON 格式返回。
-只提取明确陈述的事实，不要推测。
-每个记忆条目需要标注类型：
  * "policy"：硬性规则/约束，不可被单次指令覆盖（如"投资组合中单只股票占比不超过 30%"、"禁止投资加密货币"）
  * "preference"：用户偏好，可以被覆盖（如"喜欢简洁回答"、"使用中文"、"每天早上查看持仓"）
-返回格式：{"memories": [{"key": "简短键名", "value": "事实描述", "type": "policy|preference"}]}

可提取的信息类型：
- 用户偏好（如"喜欢简洁回答"、"使用中文"）→ type: preference
- 关键实体（如"我持有腾讯股票"、"我的投资目标是XX"）→ type: preference
- 常用指令（如"每天早上查看持仓"）→ type: preference
- 个人信息（如"我是软件工程师"、"我在北京"）→ type: preference
- 硬性约束（如"不买亏损股票"、"最大回撤不超过 10%"）→ type: policy

如果没有新信息，返回 {"memories": []}。

对话：
用户：{question}
助手：{answer}"""


def _drain_queue(q: queue.Queue, tracked: set[tuple[str, str]] | None = None) -> Iterator[dict[str, Any]]:
    """Drain all pending events from the queue and yield SSE events.

    If tracked is provided, tool_call keys are added to prevent double emission
    from the state-based path.
    """
    import json as _json
    while True:
        try:
            evt_type, evt_data = q.get_nowait()
            if evt_type == "tool_call":
                if tracked is not None:
                    args_key = _json.dumps(evt_data.get("args", {}), sort_keys=True)
                    tracked.add((evt_data["name"], args_key))
                yield {
                    "type": "tool_call",
                    "name": evt_data["name"],
                    "args": evt_data.get("args", {}),
                }
            elif evt_type == "tool_result":
                yield {
                    "type": "tool_result",
                    "name": evt_data["name"],
                    "preview": preview_json(
                        evt_data.get("result", evt_data.get("error", "")),
                        limit=2000,
                    ),
                }
            elif evt_type == "thinking":
                yield {
                    "type": "thinking",
                    "content": evt_data.get("content", ""),
                }
            elif evt_type == "progress":
                yield {
                    "type": "progress",
                    "message": evt_data.get("message", ""),
                }
        except queue.Empty:
            break


def preview_json(value: Any, limit: int = 1200) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def result_count(result: Any) -> int:
    """Count results from a tool return value."""
    if isinstance(result, list):
        return len(result)
    if isinstance(result, dict):
        for key in ("count", "holding_count", "bucket_count"):
            value = result.get(key)
            if isinstance(value, int):
                return value
        for key in ("assets", "snapshots", "holdings", "buckets"):
            value = result.get(key)
            if isinstance(value, list):
                return len(value)
        return 0
    return 0


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())