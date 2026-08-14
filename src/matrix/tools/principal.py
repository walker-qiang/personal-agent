"""Request principal context for tools invoked by legacy and Runtime paths."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_principal: ContextVar[tuple[str, str, str, bool]] = ContextVar(
    "matrix_tool_principal", default=("default", "", "read_only", False)
)


@contextmanager
def tool_principal(
    owner_id: str = "default",
    session_id: str = "",
    mode: str = "read_only",
    allow_external_effects: bool = False,
) -> Iterator[None]:
    token = _principal.set((
        owner_id or "default", session_id or "", mode or "read_only", allow_external_effects,
    ))
    try:
        yield
    finally:
        _principal.reset(token)


def current_principal() -> tuple[str, str, str, bool]:
    return _principal.get()
