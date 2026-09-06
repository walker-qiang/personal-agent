"""Adapter from Matrix's existing LLM protocol to Runtime model values."""

from __future__ import annotations

from typing import Any, Iterator

from ...llm.protocol import LLMClient
from ..domain.messages import Message, ToolCall
from ..domain.tools import ToolSpec
from ..ports.model import ModelEvent, ModelPort, ModelRequest, ModelResponse


class MatrixModelAdapter(ModelPort):
    """Keep provider-specific request/response shapes outside Runtime Core."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def complete(self, request: ModelRequest) -> ModelResponse:
        result = self.client.function_call(
            request.system_prompt,
            [_message_to_dict(message) for message in request.messages],
            [_tool_to_dict(tool) for tool in request.tools],
        )
        return ModelResponse(
            content=result.content,
            tool_calls=tuple(
                ToolCall(
                    call_id=tool_call.id,
                    name=tool_call.name,
                    arguments=tool_call.arguments,
                )
                for tool_call in result.tool_calls
            ),
            finish_reason=result.finish_reason,
        )

    def stream(self, request: ModelRequest) -> Iterator[ModelEvent]:
        native_tool_stream = getattr(self.client, "stream_function_call", None)
        if request.tools and callable(native_tool_stream):
            for event in native_tool_stream(
                request.system_prompt,
                [_message_to_dict(message) for message in request.messages],
                [_tool_to_dict(tool) for tool in request.tools],
            ):
                yield ModelEvent(
                    kind=event.kind,
                    content=event.content,
                    tool_calls=tuple(
                        ToolCall(
                            call_id=tool.id,
                            name=tool.name,
                            arguments=tool.arguments,
                        )
                        for tool in event.tool_calls
                    ),
                    metadata=dict(event.metadata),
                )
            return
        if request.tools:
            # Providers without native tool streaming still enter Runtime
            # through the same event contract; only the provider call itself
            # is buffered until it returns a complete tool decision.
            result = self.client.function_call(
                request.system_prompt,
                [_message_to_dict(message) for message in request.messages],
                [_tool_to_dict(tool) for tool in request.tools],
            )
            if result.content:
                yield ModelEvent(kind="message_delta", content=result.content)
            if result.tool_calls:
                yield ModelEvent(kind="tool_calls", tool_calls=tuple(
                    ToolCall(
                        call_id=tool.id,
                        name=tool.name,
                        arguments=tool.arguments,
                    )
                    for tool in result.tool_calls
                ))
            yield ModelEvent(
                kind="message_end",
                metadata={"finish_reason": result.finish_reason},
            )
            return
        for content in self.client.stream_complete(
            request.system_prompt,
            [_message_to_dict(message) for message in request.messages],
        ):
            yield ModelEvent(kind="message_delta", content=content)
        yield ModelEvent(kind="message_end")


def _message_to_dict(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.call_id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": _json_arguments(tool_call.arguments),
                },
            }
            for tool_call in message.tool_calls
        ]
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    return payload


def _tool_to_dict(tool: ToolSpec) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def _json_arguments(arguments: dict[str, Any]) -> str:
    import json

    return json.dumps(arguments, ensure_ascii=False)
