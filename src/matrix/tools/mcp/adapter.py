"""MCP Tool Adapter — bridges MCP server tools into personal-agent's ToolRegistry.

Converts each MCP-discovered tool into a ToolDefinition with a handler that
calls the MCP server via MCPClientManager. The existing ReAct loop, Action Space
pruning, ToolGuard, and all other tool pipeline logic work unchanged.

Tool naming convention: mcp_{server_name}_{tool_name}
- All underscores (no dots) to avoid DeepSeek API name conversion issues
- Agents with tools=[] (Commander) automatically get MCP tools
- Domain agents can opt-in by adding specific tool names or "mcp_*" patterns
"""

from __future__ import annotations

import logging
from typing import Any

from ..base import ToolDefinition
from ..registry import ToolRegistry
from .client import MCPClientManager
from .config import MCPServerConfig, load_mcp_config

logger = logging.getLogger(__name__)

# Prefix for all MCP-sourced tools
MCP_TOOL_PREFIX = "mcp_"


def make_tool_name(server_name: str, tool_name: str) -> str:
    """Generate a namespaced tool name: mcp_{server}_{tool}.

    Uses underscores (not dots) to avoid DeepSeek API name conversion issues.
    """
    # Sanitize: replace any non-alphanumeric chars with underscore
    safe_server = server_name.replace("-", "_").replace(".", "_")
    safe_tool = tool_name.replace("-", "_").replace(".", "_")
    return f"{MCP_TOOL_PREFIX}{safe_server}_{safe_tool}"


def _make_handler(
    manager: MCPClientManager, server_name: str, tool_name: str
):
    """Create a handler closure that calls the MCP server."""

    def handler(**kwargs: Any) -> dict[str, Any]:
        return manager.call_tool_sync(server_name, tool_name, kwargs)

    return handler


def register_mcp_tools(
    registry: ToolRegistry,
    manager: MCPClientManager,
) -> int:
    """Register all discovered MCP tools into the given ToolRegistry.

    Called after MCPClientManager.start() has connected to all servers
    and discovered their tools. Each MCP tool is wrapped as a ToolDefinition
    with a handler that dispatches to the MCP server.

    Args:
        registry: The ToolRegistry to register tools into
        manager: A started MCPClientManager with active connections

    Returns:
        Number of tools registered
    """
    tool_defs = manager.get_all_tool_defs()
    if not tool_defs:
        logger.info("mcp: no tools discovered from any server")
        return 0

    count = 0
    for server_name, mcp_tool in tool_defs:
        tool_name = make_tool_name(server_name, mcp_tool.name)

        # Build description with server context
        description = mcp_tool.description or ""
        if not description:
            description = f"MCP tool '{mcp_tool.name}' from server '{server_name}'"
        else:
            description = f"[MCP:{server_name}] {description}"

        # Convert MCP inputSchema to our format
        input_schema = _normalize_input_schema(mcp_tool)

        try:
            tool_def = ToolDefinition(
                name=tool_name,
                description=description,
                input_schema=input_schema,
                handler=_make_handler(manager, server_name, mcp_tool.name),
            )
            registry.register(tool_def)
            count += 1
            logger.debug("mcp: registered tool '%s'", tool_name)
        except ValueError:
            logger.warning("mcp: tool '%s' already registered, skipping", tool_name)
        except Exception as exc:
            logger.error("mcp: failed to register tool '%s': %s", tool_name, exc)

    logger.info("mcp: registered %d tool(s) into ToolRegistry", count)
    return count


def _normalize_input_schema(mcp_tool: Any) -> dict[str, Any]:
    """Convert MCP tool inputSchema to the JSON Schema format expected by ToolDefinition.

    MCP tools use JSON Schema for inputSchema, but some servers may not
    provide a complete schema. We ensure a minimal valid structure.
    """
    schema = getattr(mcp_tool, "inputSchema", None)
    if not schema or not isinstance(schema, dict):
        # Fallback: accept any object
        return {"type": "object", "properties": {}, "additionalProperties": True}

    # Ensure type is set
    if "type" not in schema:
        schema["type"] = "object"

    # Some MCP servers omit properties
    if "properties" not in schema:
        schema["properties"] = {}

    # Allow additional properties by default (MCP tools may have optional params)
    if "additionalProperties" not in schema:
        schema["additionalProperties"] = True

    return schema


def init_mcp_client(
    config_path: str | None,
) -> MCPClientManager | None:
    """Initialize MCP client from config file.

    Convenience function that loads config, starts the manager, and returns it.
    Always returns a started manager if MCP is available (even with zero servers),
    so that servers can be dynamically added later via the UI/API.
    Returns None only if MCP package is not installed.

    Args:
        config_path: Path to mcp_servers.json (or None to skip)

    Returns:
        Started MCPClientManager, or None
    """
    try:
        from .client import _MCP_AVAILABLE
    except ImportError:
        return None

    if not _MCP_AVAILABLE:
        logger.info("mcp: package not installed, skipping MCP client init")
        return None

    servers = load_mcp_config(config_path)
    manager = MCPClientManager()
    manager.start(servers)  # start() handles empty list gracefully
    return manager
