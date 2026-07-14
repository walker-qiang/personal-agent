"""Centralized structured logging configuration for the Matrix Agent.

Provides a unified logger with request ID tracking and structured format.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any


LOG_FORMAT = (
    "%(asctime)s [%(levelname)s] %(name)s"
    " %(request_id)s"
    " %(message)s"
)


class RequestIdFilter(logging.Filter):
    """Inject request_id into log records from context."""

    _request_id: str | None = None

    def filter(self, record: logging.LogRecord) -> bool:
        rid = RequestIdFilter._request_id
        record.request_id = f"[{rid}]" if rid else ""
        return True

    @classmethod
    def set_request_id(cls, request_id: str | None) -> None:
        cls._request_id = request_id

    @classmethod
    def get_request_id(cls) -> str | None:
        return cls._request_id


def setup_logging(level: int = logging.INFO, log_dir: str = "") -> logging.Logger:
    """Configure root logger with structured format.

    Args:
        level: Logging level (default INFO).
        log_dir: Directory for rotating log files. Empty string disables file logging.

    Returns:
        The "matrix" logger instance.
    """
    formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S")

    # Console handler (stdout)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    # Clear any existing handlers to avoid duplicates
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # File handler with daily rotation, 7-day retention
    if log_dir:
        log_path = Path(log_dir) / "matrix.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = TimedRotatingFileHandler(
            str(log_path), when="midnight", backupCount=7, encoding="utf-8",
        )
        fh.setFormatter(formatter)
        fh.addFilter(RequestIdFilter())
        fh.setLevel(level)
        root.addHandler(fh)

    # Set matrix logger level
    logger = logging.getLogger("matrix")
    logger.setLevel(level)
    logger.propagate = True

    return logger


def get_logger(name: str = "matrix") -> logging.Logger:
    """Get a logger with matrix namespace.

    Args:
        name: Logger name suffix (default "matrix").

    Returns:
        A logging.Logger instance.
    """
    return logging.getLogger(name)


def log_request(
    request_id: str,
    method: str,
    path: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log an incoming HTTP request with structured context.

    Args:
        request_id: Unique request identifier.
        method: HTTP method.
        path: Request path.
        extra: Additional key-value pairs to include in the log line.
    """
    logger = get_logger("matrix.http")
    parts = [f"request={method} {path}"]
    if extra:
        parts.extend(f"{k}={v}" for k, v in extra.items())
    logger.info(" ".join(parts))