"""Role system for the Matrix Agent."""

from __future__ import annotations

from .base import RoleDefinition
from .general_assistant import GENERAL_ASSISTANT
from .investment_analyst import INVESTMENT_ANALYST

__all__ = [
    "RoleDefinition",
    "GENERAL_ASSISTANT",
    "INVESTMENT_ANALYST",
]