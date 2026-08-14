"""Scriptable model fake; it does not call a provider."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Iterator

from ..domain.messages import ToolCall
from ..ports.model import ModelEvent, ModelPort, ModelRequest, ModelResponse


class FakeModel(ModelPort):
    """Return scripted responses in order and record requests."""

    def __init__(self, responses: Iterable[ModelResponse] = ()) -> None:
        self._responses = deque(responses)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            return ModelResponse(content="fake model has no scripted response", finish_reason="stop")
        return self._responses.popleft()

    def stream(self, request: ModelRequest) -> Iterator[ModelEvent]:
        response = self.complete(request)
        if response.content:
            yield ModelEvent(kind="message_delta", content=response.content)
        if response.tool_calls:
            yield ModelEvent(kind="tool_calls", tool_calls=response.tool_calls)
        yield ModelEvent(kind="message_end", metadata={"finish_reason": response.finish_reason})


def tool_call(call_id: str, name: str, arguments: dict | None = None) -> ToolCall:
    """Small helper for readable fake-model scenarios."""

    return ToolCall(call_id=call_id, name=name, arguments=arguments or {})
