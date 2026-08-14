"""Errors owned by the independent runtime domain."""

from __future__ import annotations


class RuntimeErrorBase(Exception):
    """Base class for errors that can cross the runtime adapter boundary."""


class RuntimeValidationError(RuntimeErrorBase, ValueError):
    """A runtime request or state violates a declared contract."""


class OperationConflictError(RuntimeErrorBase):
    """A compare-and-set or single-active-operation check failed."""


class RuntimeNotImplementedError(RuntimeErrorBase, NotImplementedError):
    """A public runtime capability scheduled for a later work package."""
