from __future__ import annotations

import pytest

from matrix.runtime.domain.errors import RuntimeValidationError
from matrix.runtime.domain.operations import OperationPhase, OperationState
from matrix.runtime.domain.requests import ExecutionOptions, RunRequest
from matrix.runtime.domain.tools import RecoveryPolicy, ToolSpec


def test_run_request_keeps_owner_opaque_and_defaults_tools_to_manual() -> None:
    request = RunRequest(owner_id="user-a", session_id="session-1", agent_id="assistant")

    assert request.owner_id == "user-a"
    assert request.execution_options.tool_execution_mode == "sequential"
    assert request.tools == []
    assert ToolSpec(name="write", recovery_policy=RecoveryPolicy.MANUAL).recovery_policy == RecoveryPolicy.MANUAL


def test_run_request_rejects_missing_isolation_keys() -> None:
    with pytest.raises(RuntimeValidationError, match="owner_id"):
        RunRequest(owner_id="", session_id="session-1", agent_id="assistant")


def test_execution_options_reject_parallel_tools_in_wp1() -> None:
    with pytest.raises(RuntimeValidationError, match="sequential"):
        ExecutionOptions(tool_execution_mode="parallel")


def test_operation_state_starts_in_created_phase() -> None:
    operation = OperationState(
        operation_id="op-1",
        owner_id="user-a",
        session_id="session-1",
        agent_id="assistant",
    )

    assert operation.phase is OperationPhase.CREATED
    assert operation.version == 0
    assert not operation.is_terminal
