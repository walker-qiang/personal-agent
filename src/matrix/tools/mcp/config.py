"""MCP server configuration schema and loading."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Declarative configuration for a single MCP server connection."""

    name: str  # unique identifier, used for tool namespacing
    transport: str = "stdio"  # "stdio" | "http"
    # stdio transport
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # http transport
    url: str = ""
    # misc
    enabled: bool = True
    timeout: float = 60.0  # tool call timeout in seconds

    def validate(self) -> list[str]:
        """Return list of validation error messages (empty = valid)."""
        errors: list[str] = []
        if not self.name:
            errors.append("name is required")
        if self.transport not in ("stdio", "http"):
            errors.append(f"unsupported transport: {self.transport}")
        if self.transport == "stdio" and not self.command:
            errors.append("command is required for stdio transport")
        if self.transport == "http" and not self.url:
            errors.append("url is required for http transport")
        return errors


def load_mcp_config(config_path: str | Path | None) -> list[MCPServerConfig]:
    """Load MCP server configurations from a JSON file.

    Expected format:
    {
      "servers": [
        {
          "name": "filesystem",
          "transport": "stdio",
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"],
          "enabled": true
        },
        {
          "name": "http-api",
          "transport": "http",
          "url": "http://localhost:3000/mcp"
        }
      ]
    }
    """
    if not config_path:
        return []

    path = Path(config_path).expanduser()
    if not path.is_file():
        logger.debug("mcp config file not found: %s", path)
        return []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("mcp config load failed (%s): %s", path, exc)
        return []

    servers_raw: list[dict[str, Any]] = raw.get("servers", [])
    if not isinstance(servers_raw, list):
        logger.warning("mcp config: 'servers' must be a list, got %s", type(servers_raw))
        return []

    configs: list[MCPServerConfig] = []
    for i, srv in enumerate(servers_raw):
        if not isinstance(srv, dict):
            logger.warning("mcp config: server[%d] is not an object, skipped", i)
            continue
        cfg = MCPServerConfig(
            name=srv.get("name", "").strip(),
            transport=srv.get("transport", "stdio").strip().lower(),
            command=srv.get("command", "").strip(),
            args=list(srv.get("args", [])),
            env=dict(srv.get("env", {})),
            url=srv.get("url", "").strip(),
            enabled=srv.get("enabled", True),
            timeout=float(srv.get("timeout", 60.0)),
        )
        errors = cfg.validate()
        if errors:
            logger.warning("mcp config: server '%s' invalid: %s", cfg.name, ", ".join(errors))
            continue
        configs.append(cfg)

    logger.info("mcp config: loaded %d server(s) from %s", len(configs), path)
    return configs


def save_mcp_config(config_path: str | Path, servers: list[MCPServerConfig]) -> None:
    """Save MCP server configurations to a JSON file.

    Creates parent directory if it doesn't exist.
    """
    path = Path(config_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {"servers": []}
    for cfg in servers:
        srv: dict[str, Any] = {
            "name": cfg.name,
            "transport": cfg.transport,
            "enabled": cfg.enabled,
            "timeout": cfg.timeout,
        }
        if cfg.transport == "stdio":
            srv["command"] = cfg.command
            srv["args"] = cfg.args
            if cfg.env:
                srv["env"] = cfg.env
        elif cfg.transport == "http":
            srv["url"] = cfg.url
        data["servers"].append(srv)

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("mcp config: saved %d server(s) to %s", len(servers), path)
