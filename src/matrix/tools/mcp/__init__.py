"""MCP (Model Context Protocol) client integration.

Allows personal-agent to consume tools from external MCP servers.
Acts as an MCP Client that discovers and calls tools on connected servers.
"""

from .config import MCPServerConfig, load_mcp_config, save_mcp_config
from .client import MCPClientManager
from .adapter import register_mcp_tools, init_mcp_client

__all__ = [
    "MCPServerConfig",
    "load_mcp_config",
    "save_mcp_config",
    "MCPClientManager",
    "register_mcp_tools",
    "init_mcp_client",
]
