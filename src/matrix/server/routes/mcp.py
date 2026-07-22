"""MCP server management endpoints — CRUD for MCP server connections.

Mirrors the Skill management pattern: list/create/update/delete MCP servers
with config persistence to mcp_servers.json and dynamic connection management.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...tools.mcp import MCPServerConfig, load_mcp_config, save_mcp_config

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_manager(request: Request) -> Any | None:
    """Get the MCPClientManager from app state, or None."""
    return getattr(request.app.state, "mcp_client", None)


def _get_config_path(request: Request) -> str:
    """Get the MCP config file path."""
    return request.app.state.chat.config.mcp_config_path


@router.get("/mcp/servers")
async def list_servers(request: Request):
    """List all configured MCP servers with their connection status."""
    manager = _get_manager(request)
    if manager is None:
        # MCP not initialized — load from config file
        servers = load_mcp_config(_get_config_path(request))
        return {
            "servers": [
                {
                    "name": s.name,
                    "transport": s.transport,
                    "command": s.command,
                    "args": s.args,
                    "url": s.url,
                    "env": s.env,
                    "enabled": s.enabled,
                    "connected": False,
                    "tool_count": 0,
                    "tools": [],
                    "timeout": s.timeout,
                }
                for s in servers
            ],
            "mcp_available": False,
        }
    return {"servers": manager.list_servers(), "mcp_available": True}


@router.post("/mcp/servers")
async def create_server(request: Request):
    """Add a new MCP server, connect to it, and persist config."""
    manager = _get_manager(request)
    if manager is None:
        return JSONResponse(
            {"error": "MCP client not available. Install with: pip install -e '.[mcp]'"},
            status_code=400,
        )

    payload = await request.json()
    name = str(payload.get("name", "")).strip()
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)

    # Build config from payload
    cfg = _build_config_from_payload(payload)
    errors = cfg.validate()
    if errors:
        return JSONResponse({"error": ", ".join(errors)}, status_code=400)

    if cfg.name in {s["name"] for s in manager.list_servers()}:
        return JSONResponse({"error": f"server '{cfg.name}' already exists"}, status_code=409)

    # Connect to the server
    result = manager.add_server(cfg)
    if result.get("error"):
        return JSONResponse(result, status_code=500)

    # Register tools into the ToolRegistry
    try:
        from ...tools.mcp import register_mcp_tools
        registry = request.app.state.tools
        register_mcp_tools(registry, manager)
    except Exception as exc:
        logger.warning("mcp: failed to register tools for '%s': %s", cfg.name, exc)

    # Persist config
    _persist_config(request, manager)

    return result


@router.put("/mcp/servers/{server_name}")
async def update_server(request: Request, server_name: str):
    """Update an MCP server config (reconnect)."""
    manager = _get_manager(request)
    if manager is None:
        return JSONResponse({"error": "MCP client not available"}, status_code=400)

    payload = await request.json()
    existing = [s for s in manager.list_servers() if s["name"] == server_name]
    if not existing:
        return JSONResponse({"error": f"server '{server_name}' not found"}, status_code=404)

    # Disconnect old, create new config, reconnect
    manager.remove_server(server_name)

    cfg = _build_config_from_payload(payload, name=server_name)
    errors = cfg.validate()
    if errors:
        return JSONResponse({"error": ", ".join(errors)}, status_code=400)

    result = manager.add_server(cfg)
    if result.get("error"):
        return JSONResponse(result, status_code=500)

    # Re-register tools
    try:
        from ...tools.mcp import register_mcp_tools
        registry = request.app.state.tools
        register_mcp_tools(registry, manager)
    except Exception as exc:
        logger.warning("mcp: failed to re-register tools for '%s': %s", cfg.name, exc)

    _persist_config(request, manager)
    return result


@router.delete("/mcp/servers/{server_name}")
async def delete_server(request: Request, server_name: str):
    """Remove an MCP server and disconnect."""
    manager = _get_manager(request)
    if manager is None:
        return JSONResponse({"error": "MCP client not available"}, status_code=400)

    existing = [s for s in manager.list_servers() if s["name"] == server_name]
    if not existing:
        return JSONResponse({"error": f"server '{server_name}' not found"}, status_code=404)

    manager.remove_server(server_name)
    _persist_config(request, manager)
    return {"ok": True, "name": server_name}


@router.post("/mcp/servers/{server_name}/toggle")
async def toggle_server(request: Request, server_name: str):
    """Toggle a server's enabled state (connect/disconnect without deleting config)."""
    manager = _get_manager(request)
    if manager is None:
        return JSONResponse({"error": "MCP client not available"}, status_code=400)

    servers = manager.list_servers()
    existing = [s for s in servers if s["name"] == server_name]
    if not existing:
        return JSONResponse({"error": f"server '{server_name}' not found"}, status_code=404)

    current = existing[0]
    if current["connected"]:
        # Disconnect
        manager.remove_server(server_name)
        # Re-add with enabled=False (config only, won't connect if manager checks enabled)
        cfg = _rebuild_config_from_status(current, enabled=False)
        manager._configs[cfg.name] = cfg
    else:
        # Reconnect
        cfg = _rebuild_config_from_status(current, enabled=True)
        result = manager.add_server(cfg)
        if result.get("error"):
            return JSONResponse(result, status_code=500)

        # Re-register tools
        try:
            from ...tools.mcp import register_mcp_tools
            registry = request.app.state.tools
            register_mcp_tools(registry, manager)
        except Exception as exc:
            logger.warning("mcp: failed to register tools for '%s': %s", cfg.name, exc)

    _persist_config(request, manager)
    return {"ok": True, "name": server_name}


# ---- Helpers ----


def _build_config_from_payload(payload: dict, name: str = "") -> MCPServerConfig:
    """Build MCPServerConfig from API request payload."""
    return MCPServerConfig(
        name=name or str(payload.get("name", "")).strip(),
        transport=str(payload.get("transport", "stdio")).strip().lower(),
        command=str(payload.get("command", "")).strip(),
        args=list(payload.get("args", [])),
        env=dict(payload.get("env", {})),
        url=str(payload.get("url", "")).strip(),
        enabled=payload.get("enabled", True),
        timeout=float(payload.get("timeout", 60.0)),
    )


def _rebuild_config_from_status(status: dict, enabled: bool = True) -> MCPServerConfig:
    """Rebuild MCPServerConfig from a list_servers() status dict."""
    return MCPServerConfig(
        name=status["name"],
        transport=status["transport"],
        command=status.get("command", ""),
        args=status.get("args", []),
        env=status.get("env", {}),
        url=status.get("url", ""),
        enabled=enabled,
        timeout=status.get("timeout", 60.0),
    )


def _persist_config(request: Request, manager: Any) -> None:
    """Save current manager state to config file."""
    try:
        servers = [
            _rebuild_config_from_status(s, s["enabled"])
            for s in manager.list_servers()
        ]
        save_mcp_config(_get_config_path(request), servers)
    except Exception as exc:
        logger.warning("mcp: failed to persist config: %s", exc)
