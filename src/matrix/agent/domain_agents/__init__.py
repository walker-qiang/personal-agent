"""Domain agent definitions."""

from __future__ import annotations

from .coding_assistant import CODING_ASSISTANT
from .investment_analyst import INVESTMENT_ANALYST
from .knowledge_manager import KNOWLEDGE_MANAGER
from .media_generator import MEDIA_GENERATOR

__all__ = [
    "CODING_ASSISTANT",
    "INVESTMENT_ANALYST",
    "KNOWLEDGE_MANAGER",
    "MEDIA_GENERATOR",
]