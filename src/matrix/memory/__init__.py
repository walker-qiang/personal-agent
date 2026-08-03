"""Memory package: evolution, cross-session lesson store.

Public API::

    from matrix.memory import MemoryEvolution, EvolutionConfig, EvolutionReport
    from matrix.memory import LessonStore, Lesson
"""

from .evolution import (
    EvolutionConfig,
    EvolutionReport,
    MemoryEvolution,
    ScoredMemory,
)
from .lesson_store import Lesson, LessonStore

__all__ = [
    "MemoryEvolution",
    "EvolutionConfig",
    "EvolutionReport",
    "ScoredMemory",
    "LessonStore",
    "Lesson",
]
