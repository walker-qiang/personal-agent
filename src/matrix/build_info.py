"""Build metadata exposed by service health and capability endpoints."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _resolve_build_sha() -> str:
    configured = os.environ.get("MATRIX_BUILD_SHA", "").strip()
    if configured:
        return configured

    repository_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "--short=12", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"

    build_sha = result.stdout.strip()
    if not build_sha:
        return "unknown"
    try:
        dirty = subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
            timeout=1,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return build_sha
    return build_sha + ("-dirty" if dirty else "")


BUILD_SHA = _resolve_build_sha()
