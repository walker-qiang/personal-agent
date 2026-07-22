"""MCP Client Manager — manages MCP server connections in a background asyncio loop.

The MCP Python SDK is async-first, but personal-agent's server runs synchronously.
This module bridges the gap by running a dedicated asyncio event loop in a daemon
thread. All MCP operations are submitted to this loop via run_coroutine_threadsafe.

Supports dynamic add/remove of individual servers without restarting the manager.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time as _time
from concurrent.futures import Future
from typing import Any

from .config import MCPServerConfig

logger = logging.getLogger(__name__)

# MCP SDK types (lazy import to keep mcp optional)
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.types import Tool as MCPTool, TextContent, ImageContent, EmbeddedResource

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    logger.debug("mcp package not installed; MCP client support disabled")


class MCPClientManager:
    """Manages connections to multiple MCP servers in a background event loop.

    Supports dynamic add/remove of individual servers without full restart.

    Usage:
        manager = MCPClientManager()
        manager.start(servers)          # connect to initial servers
        manager.add_server(cfg)         # dynamically add a new server
        manager.remove_server("name")   # disconnect a specific server
        tools = manager.get_all_tool_defs()
        result = manager.call_tool_sync("server", "tool", {"arg": "val"})
        manager.stop()                  # disconnect all and shut down
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._global_stop: asyncio.Event | None = None
        # server_name -> ClientSession
        self._sessions: dict[str, ClientSession] = {}
        # server_name -> list of MCPTool
        self._tools_cache: dict[str, list[Any]] = {}
        # server_name -> asyncio.Event (per-server stop signal)
        self._server_stops: dict[str, asyncio.Event] = {}
        # server_name -> MCPServerConfig
        self._configs: dict[str, MCPServerConfig] = {}
        # server_name -> asyncio.Task (background manage coroutine)
        self._tasks: dict[str, asyncio.Task] = {}
        self._started = False

    # ---- Lifecycle ----

    def start(self, servers: list[MCPServerConfig]) -> None:
        """Start the background loop and connect to all configured servers."""
        if not _MCP_AVAILABLE:
            logger.warning("mcp package not installed; skipping MCP client startup")
            return
        if self._started:
            logger.warning("mcp client already started")
            return

        self._loop = asyncio.new_event_loop()
        self._global_stop = asyncio.Event()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="mcp-client-loop"
        )
        self._thread.start()
        self._started = True

        if not servers:
            logger.info("mcp: no servers configured, skipping initial connections")
            return

        # Connect to all servers (blocking, with timeout)
        future = asyncio.run_coroutine_threadsafe(
            self._connect_servers(servers), self._loop
        )
        try:
            future.result(timeout=30.0)
        except Exception as exc:
            logger.error("mcp: startup error: %s", exc)

        total_tools = sum(len(t) for t in self._tools_cache.values())
        connected = len(self._sessions)
        logger.info(
            "mcp: started — %d/%d servers connected, %d tools discovered",
            connected, len(servers), total_tools,
        )

    def stop(self) -> None:
        """Disconnect from all servers and shut down the background loop."""
        if not self._started:
            return

        # Signal all per-server stop events + global stop
        if self._loop and self._global_stop:
            for name in list(self._server_stops.keys()):
                self._loop.call_soon_threadsafe(self._server_stops[name].set)
            self._loop.call_soon_threadsafe(self._global_stop.set)
            _time.sleep(0.5)

        # Stop the event loop
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=10.0)

        self._sessions.clear()
        self._tools_cache.clear()
        self._server_stops.clear()
        self._configs.clear()
        self._tasks.clear()
        self._started = False
        logger.info("mcp: stopped")

    # ---- Dynamic server management ----

    def add_server(self, cfg: MCPServerConfig) -> dict[str, Any]:
        """Dynamically connect to a new MCP server.

        Returns a dict with connection status and discovered tools.
        If a server with the same name exists but is disconnected, it will be reconnected.
        """
        if not self._started or not self._loop:
            return {"error": True, "message": "MCP client not started"}
        if cfg.name in self._sessions:
            return {"error": True, "message": f"server '{cfg.name}' already connected"}

        future: Future = asyncio.run_coroutine_threadsafe(
            self._connect_single(cfg), self._loop
        )
        try:
            result = future.result(timeout=20.0)
            return result
        except TimeoutError:
            return {"error": True, "message": f"connection timeout for '{cfg.name}'"}
        except Exception as exc:
            logger.error("mcp: add_server failed ('%s'): %s", cfg.name, exc)
            return {"error": True, "message": str(exc)}

    def remove_server(self, name: str) -> dict[str, Any]:
        """Disconnect a specific MCP server by name."""
        if not self._started or not self._loop:
            return {"error": True, "message": "MCP client not started"}
        if name not in self._sessions and name not in self._configs:
            return {"error": True, "message": f"server '{name}' not found"}

        # Signal the per-server stop event
        stop_event = self._server_stops.get(name)
        if stop_event:
            self._loop.call_soon_threadsafe(stop_event.set)

        # Wait for the manage task to finish (with timeout)
        task = self._tasks.get(name)
        if task and self._loop and not task.done():
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    asyncio.wait_for(asyncio.shield(task), timeout=5.0),
                    self._loop,
                )
                fut.result(timeout=10.0)
            except Exception:
                # Force cancel if it didn't finish
                self._loop.call_soon_threadsafe(task.cancel)

        # Clean up registries
        self._sessions.pop(name, None)
        self._tools_cache.pop(name, None)
        self._configs.pop(name, None)
        self._server_stops.pop(name, None)
        self._tasks.pop(name, None)

        logger.info("mcp: server '%s' removed", name)
        return {"ok": True, "name": name}

    def list_servers(self) -> list[dict[str, Any]]:
        """Return list of all configured servers with their status."""
        result: list[dict[str, Any]] = []
        for name, cfg in self._configs.items():
            tools = self._tools_cache.get(name, [])
            result.append({
                "name": cfg.name,
                "transport": cfg.transport,
                "command": cfg.command,
                "args": cfg.args,
                "url": cfg.url,
                "env": cfg.env,
                "enabled": cfg.enabled,
                "connected": name in self._sessions,
                "tool_count": len(tools),
                "tools": [t.name for t in tools],
                "timeout": cfg.timeout,
            })
        return result

    # ---- Tool access ----

    def get_all_tool_defs(self) -> list[tuple[str, Any]]:
        """Return all discovered tools as (server_name, MCPTool) pairs."""
        result: list[tuple[str, Any]] = []
        for server_name, tools in self._tools_cache.items():
            for tool in tools:
                result.append((server_name, tool))
        return result

    def call_tool_sync(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Call a tool on an MCP server (blocking)."""
        if not self._started or not self._loop:
            raise RuntimeError("MCP client not started")
        if server_name not in self._sessions:
            raise RuntimeError(f"MCP server not connected: {server_name}")

        timeout = 60.0
        cfg = self._configs.get(server_name)
        if cfg:
            timeout = cfg.timeout

        future: Future = asyncio.run_coroutine_threadsafe(
            self._call_tool(server_name, tool_name, arguments), self._loop
        )
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            return {"error": True, "message": f"MCP tool call timed out ({timeout}s)"}
        except Exception as exc:
            logger.error("mcp: call_tool failed (%s/%s): %s", server_name, tool_name, exc)
            return {"error": True, "message": str(exc)}

    # ---- Internal async methods ----

    def _run_loop(self) -> None:
        """Run the asyncio event loop in the background thread."""
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect_servers(self, servers: list[MCPServerConfig]) -> None:
        """Connect to multiple servers concurrently."""
        enabled = [s for s in servers if s.enabled]
        tasks = [
            asyncio.ensure_future(self._connect_single(cfg))
            for cfg in enabled
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for cfg, result in zip(enabled, results):
            if isinstance(result, Exception):
                logger.error("mcp: failed to connect to '%s': %s", cfg.name, result)

    async def _connect_single(self, cfg: MCPServerConfig) -> dict[str, Any]:
        """Connect to a single MCP server and keep the session alive.

        Handles reconnection: if a stale task exists for the same name,
        it is cancelled and awaited before starting a new connection.
        """
        if cfg.name in self._sessions:
            return {"error": True, "message": f"already connected: {cfg.name}"}

        # Cancel any stale task from a previous (now disconnected) connection
        old_task = self._tasks.get(cfg.name)
        if old_task and not old_task.done():
            old_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(old_task), timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
        # Clean up any leftover state
        self._sessions.pop(cfg.name, None)
        self._tools_cache.pop(cfg.name, None)

        # Register config and create per-server stop event
        self._configs[cfg.name] = cfg
        stop_event = asyncio.Event()
        self._server_stops[cfg.name] = stop_event

        # Launch the manage coroutine as a background task
        task = asyncio.ensure_future(self._manage_server(cfg, stop_event))
        self._tasks[cfg.name] = task

        # Wait for the session to appear in _sessions (or task to fail)
        deadline = asyncio.get_event_loop().time() + 15.0
        while asyncio.get_event_loop().time() < deadline:
            if cfg.name in self._sessions:
                break
            if task.done() and cfg.name not in self._sessions:
                # Task finished without connecting — error already logged
                break
            await asyncio.sleep(0.15)

        tools = self._tools_cache.get(cfg.name, [])
        connected = cfg.name in self._sessions
        return {
            "ok": connected,
            "name": cfg.name,
            "connected": connected,
            "tools": [t.name for t in tools],
            "tool_count": len(tools),
            "error": False if connected else True,
            "message": "" if connected else f"failed to connect to '{cfg.name}'",
        }

    async def _manage_server(self, cfg: MCPServerConfig, stop_event: asyncio.Event) -> None:
        """Manage the lifecycle of a single MCP server connection."""
        try:
            if cfg.transport == "stdio":
                async with stdio_client(
                    StdioServerParameters(
                        command=cfg.command,
                        args=cfg.args,
                        env={**cfg.env} if cfg.env else None,
                    )
                ) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await self._init_session(cfg, session)
                        await stop_event.wait()
            elif cfg.transport == "http":
                async with streamablehttp_client(cfg.url) as (
                    read_stream,
                    write_stream,
                    _get_session_id,
                ):
                    async with ClientSession(read_stream, write_stream) as session:
                        await self._init_session(cfg, session)
                        await stop_event.wait()
        except asyncio.CancelledError:
            logger.debug("mcp: server '%s' cancelled", cfg.name)
        except Exception as exc:
            # Recursively unwrap ExceptionGroup (Python 3.11+) for clear error messages
            exc_msg = _unwrap_exception(exc)
            logger.error("mcp: server '%s' connection error: %s", cfg.name, exc_msg)
        finally:
            # Clean up session on exit
            self._sessions.pop(cfg.name, None)
            self._tools_cache.pop(cfg.name, None)

    async def _init_session(self, cfg: MCPServerConfig, session: ClientSession) -> None:
        """Initialize a session, discover tools, and register."""
        await session.initialize()
        result = await session.list_tools()
        self._sessions[cfg.name] = session
        self._tools_cache[cfg.name] = result.tools
        tool_names = [t.name for t in result.tools]
        logger.info(
            "mcp: server '%s' connected — %d tools: %s",
            cfg.name, len(tool_names), ", ".join(tool_names) or "(none)",
        )

    async def _call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Call a tool on an MCP server (async)."""
        session = self._sessions.get(server_name)
        if session is None:
            return {"error": True, "message": f"server not connected: {server_name}"}

        result = await session.call_tool(tool_name, arguments)
        return _normalize_mcp_result(result)


# ---- Result normalization ----


def _normalize_mcp_result(result: Any) -> dict[str, Any]:
    """Normalize MCP CallToolResult to the dict format expected by personal-agent."""
    is_error = getattr(result, "isError", False)
    content: list[Any] = getattr(result, "content", [])

    if is_error:
        texts = [
            c.text for c in content if isinstance(c, TextContent) if hasattr(c, "text")
        ]
        return {"error": True, "message": "\n".join(texts) or "MCP tool returned an error"}

    if len(content) == 1:
        item = content[0]
        if isinstance(item, TextContent) and hasattr(item, "text"):
            text = item.text
            try:
                import json
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
                return {"result": parsed}
            except (json.JSONDecodeError, ValueError):
                return {"result": text}
        if hasattr(item, "model_dump"):
            return {"result": item.model_dump()}
        return {"result": str(item)}

    parts: list[Any] = []
    for item in content:
        if isinstance(item, TextContent) and hasattr(item, "text"):
            parts.append(item.text)
        elif hasattr(item, "model_dump"):
            parts.append(item.model_dump())
        else:
            parts.append(str(item))
    return {"result": parts}


def _unwrap_exception(exc: BaseException, depth: int = 0) -> str:
    """Recursively unwrap ExceptionGroup to extract the root cause.

    Returns a human-readable string with indented sub-exceptions.
    """
    indent = "  " * depth
    msg = f"{type(exc).__name__}: {exc}"

    if hasattr(exc, "exceptions"):
        for sub in exc.exceptions:
            msg += f"\n{indent}└─ {_unwrap_exception(sub, depth + 1)}"

    return msg
