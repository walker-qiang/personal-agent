"""Memory package: evolution, retrieval, write decisions, lessons, safety.

Public API::

    from matrix.memory import MemoryEvolution, EvolutionConfig, EvolutionReport
    from matrix.memory import LessonStore, Lesson
    from matrix.memory import MemoryRetriever, build_memory_block
    from matrix.memory import MemoryDecisionEngine, MemorySafetyGuard
    from matrix.memory import MemoryWriteWorker
"""

from .evolution import (
    EvolutionConfig,
    EvolutionReport,
    MemoryEvolution,
    ScoredMemory,
)
from .lesson_store import KIND_FAILURE, KIND_SUCCESS, Lesson, LessonStore
from .retriever import MemoryRetriever, RetrievedMemory
from .prompting import MemoryBlock, build_memory_block
from .decisions import MemoryDecisionEngine, WriteDecision
from .safety import MemorySafetyGuard
from .temporal import parse_fact_time, resolve_time_range
from .writer import MemoryWriteWorker

__all__ = [
    "MemoryEvolution",
    "EvolutionConfig",
    "EvolutionReport",
    "ScoredMemory",
    "LessonStore",
    "Lesson",
    "KIND_FAILURE",
    "KIND_SUCCESS",
    "MemoryRetriever",
    "RetrievedMemory",
    "MemoryBlock",
    "build_memory_block",
    "MemoryDecisionEngine",
    "WriteDecision",
    "MemorySafetyGuard",
    "parse_fact_time",
    "resolve_time_range",
    "MemoryWriteWorker",
]
