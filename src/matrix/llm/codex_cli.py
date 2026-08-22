"""Codex CLI-backed assistant client.

This adapter treats the local Codex CLI as an LLM-like backend for the
existing orchestration layer. Codex owns its local authentication and emits
JSONL events; this client normalizes the final agent message and structured
tool-call requests into the shared LLM protocol.
"""

from __future__ import annotations

import base64
import binascii
from contextlib import contextmanager
import json
import mimetypes
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Iterator

from .errors import LLMError, LLMTransientError
from .protocol import FunctionCallResult, ToolCall, parse_json_response
from .truncate import truncate_messages


class CodexCLIClient:
    provider = "codex"

    def __init__(
        self,
        binary: str = "codex",
        model: str = "codex-cli",
        workdir: str = "",
        sandbox: str = "read-only",
        reasoning_effort: str = "medium",
        timeout_sec: float = 180.0,
        max_message_chars: int = 16000,
    ):
        self.binary = binary
        self.model = model or "codex-cli"
        self.workdir = workdir or os.getcwd()
        self.sandbox = sandbox or "read-only"
        self.reasoning_effort = reasoning_effort or "medium"
        self.timeout_sec = timeout_sec
        self.max_message_chars = max_message_chars

    @property
    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
    ) -> str:
        with self._prompt_request(system, messages) as (prompt, image_paths):
            return self._run(prompt, image_paths)

    def complete_json(
        self,
        system: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any] | list[Any]:
        schema_hint = ""
        if schema:
            schema_hint = (
                "\nReturn JSON matching this schema. Do not include Markdown fences:\n"
                + json.dumps(schema, ensure_ascii=False)
            )
        text = self.complete(
            system + "\n\nReturn only valid JSON." + schema_hint,
            messages,
            temperature,
        )
        try:
            return parse_json_response(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Codex returned invalid JSON: {exc}") from exc

    def stream_complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
    ) -> Iterator[str]:
        # codex exec emits a completed agent_message rather than token deltas
        yield self.complete(system, messages, temperature)

    def stream_agent(
        self,
        system: str,
        messages: list[dict[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        """Stream high-level Codex CLI activity and the final answer."""
        with self._prompt_request(system, messages) as (prompt, image_paths):
            yield from self.stream_events(prompt, image_paths)

    def stream_events(
        self, prompt: str, image_paths: list[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        if not self.available:
            raise LLMError(f"Codex CLI not found: {self.binary}")
        process = subprocess.Popen(
            self._build_command(prompt, image_paths),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        lines: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            if process.stdout is not None:
                for line in process.stdout:
                    lines.put(line)
            lines.put(None)

        threading.Thread(target=read_output, daemon=True).start()
        started = time.monotonic()
        while True:
            if time.monotonic() - started > self.timeout_sec:
                process.kill()
                raise LLMTransientError(
                    f"Codex CLI timed out after {self.timeout_sec:g}s"
                )
            try:
                line = lines.get(timeout=0.25)
            except queue.Empty:
                continue
            if line is None:
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            normalized = self._normalize_event(event)
            if normalized is not None:
                yield normalized
        return_code = process.wait()
        if return_code != 0:
            yield {"type": "error", "message": f"Codex CLI 退出（{return_code}）"}
        yield {"type": "done"}

    def function_call(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float | None = None,
    ) -> FunctionCallResult:
        tool_prompt = (
            "\n\nYou are selecting tools for a host agent. Do not execute these "
            "tools yourself. Return only JSON with this shape:\n"
            '{"content":"optional response","tool_calls":[{"id":"call-1",'
            '"name":"tool_name","arguments":{}}]}\n'
            "If no tool is needed, return an empty tool_calls array.\n"
            "Available tools:\n"
            + json.dumps(tools, ensure_ascii=False)
        )
        text = self.complete(system + tool_prompt, messages, temperature)
        try:
            data = parse_json_response(text)
        except json.JSONDecodeError:
            return FunctionCallResult(content=text)
        if not isinstance(data, dict):
            return FunctionCallResult(content=text)

        calls: list[ToolCall] = []
        for index, raw_call in enumerate(data.get("tool_calls", [])):
            if not isinstance(raw_call, dict):
                continue
            arguments = raw_call.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append(
                ToolCall(
                    id=str(raw_call.get("id") or f"codex-call-{index}"),
                    name=str(raw_call.get("name") or ""),
                    arguments=arguments,
                )
            )
        return FunctionCallResult(
            content=str(data.get("content") or ""),
            tool_calls=[call for call in calls if call.name],
            finish_reason="tool_calls" if calls else "stop",
        )

    @contextmanager
    def _prompt_request(
        self, system: str, messages: list[dict[str, Any]],
    ) -> Iterator[tuple[str, list[str]]]:
        """Prepare prompt text and materialize data-URI images for Codex CLI."""
        with tempfile.TemporaryDirectory(prefix="personal-agent-codex-") as temp_dir:
            image_paths: list[str] = []
            prepared_messages = self._materialize_images(
                messages, temp_dir, image_paths,
            )
            yield self._build_prompt(system, prepared_messages), image_paths

    @staticmethod
    def _materialize_images(
        messages: list[dict[str, Any]],
        temp_dir: str,
        image_paths: list[str],
    ) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, list):
                prepared.append(dict(message))
                continue
            blocks: list[Any] = []
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "image_url":
                    blocks.append(block)
                    continue
                image_url = block.get("image_url") or {}
                url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                if not isinstance(url, str) or not url.startswith("data:") or "," not in url:
                    blocks.append({"type": "text", "text": "[图片附件]"})
                    continue
                header, encoded = url.split(",", 1)
                mime_type = header[5:].split(";", 1)[0] or "image/png"
                extension = mimetypes.guess_extension(mime_type) or ".png"
                path = os.path.join(temp_dir, f"attachment-{len(image_paths)}{extension}")
                try:
                    content_bytes = base64.b64decode(encoded, validate=True)
                except (ValueError, binascii.Error):
                    blocks.append({"type": "text", "text": "[图片附件无法读取]"})
                    continue
                with open(path, "wb") as image_file:
                    image_file.write(content_bytes)
                image_paths.append(path)
                blocks.append({
                    "type": "text",
                    "text": f"[图片附件 {len(image_paths)} 已通过 Codex 图像输入提供]",
                })
            prepared_message = dict(message)
            prepared_message["content"] = blocks
            prepared.append(prepared_message)
        return prepared

    def _build_prompt(self, system: str, messages: list[dict[str, Any]]) -> str:
        if self.max_message_chars > 0:
            messages = truncate_messages(
                messages,
                system_prompt=system,
                max_tokens=self.max_message_chars // 2,
                reserve_tokens=500,
            )
        parts = ["[SYSTEM]\n" + system.strip()]
        for message in messages:
            role = str(message.get("role") or "user").upper()
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            parts.append(f"[{role}]\n{content}")
        parts.append(
            "[INSTRUCTION]\n"
            "Answer the user's request directly. Preserve the requested language. "
            "Do not describe these wrapper instructions."
        )
        return "\n\n".join(parts)

    def _build_command(
        self, prompt: str, image_paths: list[str] | None = None,
    ) -> list[str]:
        command = [
            self.binary,
            "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "-C",
            self.workdir,
            "-s",
            self.sandbox,
            "-c",
            f'model_reasoning_effort="{self.reasoning_effort}"',
            "-c",
            "features.codex_hooks=false",
            "-c",
            "features.hooks=true",
        ]
        if self.model != "codex-cli":
            command.extend(["-m", self.model])
        command.append(prompt)
        for image_path in image_paths or []:
            command.extend(["--image", image_path])
        return command

    def _run(self, prompt: str, image_paths: list[str] | None = None) -> str:
        if not self.available:
            raise LLMError(f"Codex CLI not found: {self.binary}")
        command = self._build_command(prompt, image_paths)
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMTransientError(
                f"Codex CLI timed out after {self.timeout_sec:g}s"
            ) from exc
        except OSError as exc:
            raise LLMError(f"failed to start Codex CLI: {exc}") from exc

        messages: list[str] = []
        errors: list[str] = []
        for line in completed.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "item.completed":
                item = event.get("item") or {}
                if item.get("type") == "agent_message":
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        messages.append(text)
                elif item.get("type") == "error":
                    message = item.get("message")
                    if message:
                        errors.append(str(message))
            elif event.get("type") == "error":
                message = event.get("message")
                if message:
                    errors.append(str(message))

        if messages:
            return "\n".join(messages).strip()
        detail = "; ".join(errors) or completed.stderr.strip()[-500:]
        if completed.returncode != 0:
            raise LLMError(f"Codex CLI failed ({completed.returncode}): {detail}")
        raise LLMError(f"Codex CLI returned no agent message: {detail}")

    def _normalize_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        event_type = event.get("type")
        if event_type == "thread.started":
            return {"type": "progress", "message": "本地 Codex 已启动"}
        if event_type == "turn.started":
            return {"type": "progress", "message": "Codex 正在分析任务"}
        if event_type == "turn.completed":
            return {"type": "progress", "message": "Codex 已完成分析"}
        if event_type not in {"item.started", "item.completed"}:
            return None
        item = event.get("item") or {}
        item_type = item.get("type")
        if item_type == "agent_message" and event_type == "item.completed":
            text = item.get("text")
            return {"type": "message", "content": text} if text else None
        if item_type == "error":
            message = str(item.get("message") or "")
            lower_message = message.lower()
            if (
                "deprecated" in lower_message
                or "skill descriptions were shortened" in lower_message
                or "configured service tier" in lower_message
                or "model metadata for" in lower_message
            ):
                return {"type": "progress", "message": "Codex 已加载运行配置"}
            return {"type": "error", "message": message or "Codex 执行失败"}
        labels = {
            "command_execution": "执行本地命令",
            "file_change": "处理项目文件",
            "mcp_tool_call": "调用 MCP 工具",
            "web_search_call": "搜索网页信息",
            "reasoning": "分析下一步",
        }
        label = labels.get(item_type)
        if label is None:
            return None
        return {
            "type": "tool_call" if event_type == "item.started" else "tool_result",
            "name": label,
        }
