"""Memory evolution package: consolidation, conflict resolution, and forgetting.

Public API::

    from matrix.memory import MemoryEvolution, EvolutionConfig, EvolutionReport
"""

from .evolution import (
    EvolutionConfig,
    EvolutionReport,
    MemoryEvolution,
    ScoredMemory,
)

__all__ = [
    "MemoryEvolution",
    "EvolutionConfig",
    "EvolutionReport",
    "ScoredMemory",
]
