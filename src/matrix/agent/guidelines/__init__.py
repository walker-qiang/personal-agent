"""Agent guideline loader — inject domain-specific instructions into system prompt.

Guidelines are Markdown files stored in this directory. Each guideline provides
tool-specific usage rules (image generation, code execution, browser automation)
that are injected into the system prompt based on the agent's system_guidelines config.

Unlike skills (which are lazy-loaded by the LLM on demand), guidelines are
directly injected as full text because they are behavioral norms, not
executable workflows.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger("matrix.agent.guidelines")

_GUIDELINES_DIR = Path(__file__).parent

# TTL cache for guideline content (avoid re-reading files every turn)
_cache: dict[str, tuple[float, str]] = {}
_CACHE_TTL = 60  # seconds


def load_guideline(name: str) -> str:
    """Load a guideline Markdown file by name.

    Args:
        name: Guideline name without extension (e.g. "code_execution").

    Returns:
        The guideline content as a string, or empty string if the file
        does not exist or cannot be read (graceful degradation).
    """
    now = time.time()
    cached = _cache.get(name)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

    path = _GUIDELINES_DIR / f"{name}.md"
    try:
        if not path.exists() or not path.is_file():
            logger.warning("guideline not found: %s", path)
            _cache[name] = (now, "")
            return ""
        content = path.read_text(encoding="utf-8").strip()
        _cache[name] = (now, content)
        return content
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("failed to read guideline %s: %s", name, e)
        _cache[name] = (now, "")
        return ""


__all__ = ["load_guideline"]