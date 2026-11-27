"""Unit tests for ``agentic_research.error_recovery.with_retry``."""

from __future__ import annotations

import pytest

from agentic_research.core.enums import TaskStatus
from agentic_research.core.exceptions import (
    InvalidStateTransitionError,
    RetryLimitExceededError,
)
from agentic_research.core.state import create_initial_state
from agentic_research.error_recovery import with_retry


def _make_flaky_fn(fail_times: int, result: dict | None = None):
    """Build a node function that raises RuntimeError `fail_times` times,
    then succeeds and returns `result` (default: {"task_status": TaskStatus.ROUTING})."""
    calls = {"count": 0}
    payload = result if result is not None else {"task_status": TaskStatus.ROUTING}

    def fn(state):
        calls["count"] += 1
        if calls["count"] <= fail_times:
            raise RuntimeError(f"transient failure #{calls['count']}")
        return payload

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


# --------------------------------------------------------------------------- #
# Success paths
# --------------------------------------------------------------------------- #
def test_succeeds_immediately_when_node_fn_does_not_fail() -> None:
    fn = _make_flaky_fn(fail_times=0)
    wrapped = with_retry(fn, max_retries=3, node_name="test_node")
    state = create_initial_state("query")

    result = wrapped(state)

    assert result == {"task_status": TaskStatus.ROUTING}
    assert fn.calls["count"] == 1
    assert "error_log" not in result  # no retries needed -> no log noise added


def test_recovers_after_transient_failures_within_budget() -> None:
    fn = _make_flaky_fn(fail_times=2)
    wrapped = with_retry(fn, max_retries=3, node_name="test_node")
    state = create_initial_state("query")

    result = wrapped(state)

    assert result["task_status"] == TaskStatus.ROUTING
    assert fn.calls["count"] == 3  # 2 failures + 1 success
    assert len(result["error_log"]) == 2
    assert all(e.node_name == "test_node" for e in result["error_log"])
    assert result["retry_count"] == 2


def test_preexisting_error_log_in_result_is_preserved_after_retry() -> None:
    """A node's own error_log entries (e.g. Module 4's tool failures) must
    survive being wrapped, with retry-attempt records prefixed before them."""
    from agentic_research.core.contracts import ErrorRecord

    own_error = ErrorRecord(node_name="test_node", error_type="ToolExecutionFailure", message="tool broke")
    fn = _make_flaky_fn(fail_times=1, result={"task_status": TaskStatus.ROUTING, "error_log": [own_error]})
    wrapped = with_retry(fn, max_retries=3, node_name="test_node")

    result = wrapped(create_initial_state("query"))

    assert len(result["error_log"]) == 2
    assert result["error_log"][-1] is own_error  # retry records come first


# --------------------------------------------------------------------------- #
# Exhausted retry budget
# --------------------------------------------------------------------------- #
def test_raises_retry_limit_exceeded_after_exhausting_budget() -> None:
    fn = _make_flaky_fn(fail_times=999)  # always fails
    wrapped = with_retry(fn, max_retries=2, node_name="test_node")

    with pytest.raises(RetryLimitExceededError) as exc_info:
        wrapped(create_initial_state("query"))

    assert fn.calls["count"] == 3  # initial attempt + 2 retries
    assert exc_info.value.node_name == "test_node"
    assert exc_info.value.attempts == 3


def test_zero_max_retries_means_exactly_one_attempt() -> None:
    fn = _make_flaky_fn(fail_times=1)
    wrapped = with_retry(fn, max_retries=0, node_name="test_node")

    with pytest.raises(RetryLimitExceededError):
        wrapped(create_initial_state("query"))
    assert fn.calls["count"] == 1


# --------------------------------------------------------------------------- #
# Non-retryable exceptions
# --------------------------------------------------------------------------- #
def test_non_retryable_exception_propagates_immediately_without_retrying() -> None:
    calls = {"count": 0}

    def always_raises_invalid_transition(state):
        calls["count"] += 1
        raise InvalidStateTransitionError(from_status="completed", to_status="routing")

    wrapped = with_retry(always_raises_invalid_transition, max_retries=5, node_name="test_node")

    with pytest.raises(InvalidStateTransitionError):
        wrapped(create_initial_state("query"))

    assert calls["count"] == 1  # no retries consumed on a non-retryable error


def test_custom_non_retryable_tuple_is_respected() -> None:
    calls = {"count": 0}

    def always_raises_value_error(state):
        calls["count"] += 1
        raise ValueError("deterministic bad input")

    wrapped = with_retry(
        always_raises_value_error,
        max_retries=5,
        node_name="test_node",
        non_retryable=(ValueError,),
    )

    with pytest.raises(ValueError):
        wrapped(create_initial_state("query"))
    assert calls["count"] == 1
