"""Role system for the Matrix Agent."""

from __future__ import annotations

from .base import RoleDefinition
from .investment_analyst import INVESTMENT_ANALYST

__all__ = [
    "RoleDefinition",
    "INVESTMENT_ANALYST",
]