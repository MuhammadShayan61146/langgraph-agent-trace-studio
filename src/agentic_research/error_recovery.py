"""Error recovery: a retry wrapper for graph node functions.

Purpose
-------
Every node function in this project (`BaseAgent.run`, `supervisor_node`)
shares the same `state -> dict` shape. None of them currently catch a
genuinely unexpected failure at the node level — Module 4's `BaseAgent`
only catches *tool*-level failures (a known, scoped concern; see that
module). If an `LLMClient.complete()` call itself raised (a real provider
error, e.g.), nothing today stops that exception from crashing the whole
node. `with_retry` closes that gap by wrapping any node function with a
bounded retry policy, without modifying the node functions themselves.

Responsibilities
-----------------
* `with_retry(node_fn, *, max_retries, node_name)` — returns a new
  `state -> dict` function that retries `node_fn` on failure, recording
  one `ErrorRecord` per failed attempt, and raising
  `RetryLimitExceededError` once the budget is exhausted.

Design decisions
-----------------
* **A single wrapper function, not a new class hierarchy.** Every node in
  this project already has the identical `state -> dict` signature
  (`BaseAgent.run`, `supervisor_node`); a decorator-style higher-order
  function is the simplest thing that works for all of them, with no new
  abstraction beyond what's needed.
* **Reuses `RetryLimitExceededError` and `ErrorRecord`, both already
  defined** (Module 1 and Module 2 respectively) and unused until now —
  no new exception type or data contract was added for this module.
* **`max_retries` is a plain parameter, not a new config object.**
  `Settings.max_retries` (Module 1) already exists for exactly this
  purpose; callers pass `settings.max_retries` in. Adding a dedicated
  `RetryPolicy` dataclass here would duplicate that field for no benefit.
* **Wraps the node function from the outside — `agents/base.py` and
  `graph/supervisor.py` are unmodified.** This is a deliberate scope
  boundary consistent with Module 7: retry behavior is layered on, not
  built into, the node functions themselves. Actually applying this
  wrapper when registering nodes with a real `StateGraph` is left to a
  future graph-assembly step (see Module 8 report "Dependencies on Future
  Modules"), matching how Module 7's `human_approval_node` wiring was
  deferred.
* **`InvalidStateTransitionError` (Module 2) is treated as non-retryable
  by default.** It signals a deterministic state-machine inconsistency —
  retrying the identical call would fail identically every time, wasting
  the retry budget. Every other exception is treated as potentially
  transient and retried.
* **Does not duplicate Module 4's tool-level error handling.** A failed
  tool call already becomes a graceful `ToolResult`/`ErrorRecord` inside
  `BaseAgent.run` without raising — `with_retry` only ever sees an
  exception that escaped that handling entirely (i.e. a genuine node-level
  failure), so its concern is strictly complementary, not overlapping.

Limitations
------------
* Retries happen synchronously, in a loop, with no backoff/delay between
  attempts. Acceptable for this project's scope (no real, rate-limited
  provider is actually called yet); a production deployment against a real
  `LLMClient` would likely want exponential backoff, which is not
  implemented here.
* `state` is passed unchanged to every retry attempt — if `node_fn`
  mutates its input (none of this project's node functions do; `GraphState`
  updates are always returned, not applied in place), retries would see
  altered input. Documented as a precondition on `node_fn`, not enforced.
"""

from __future__ import annotations

from typing import Any, Callable

from agentic_research.core.contracts import ErrorRecord
from agentic_research.core.exceptions import (
    InvalidStateTransitionError,
    RetryLimitExceededError,
)
from agentic_research.core.state import GraphState

#: Exceptions treated as non-retryable by default: retrying would fail
#: identically every time, so retry attempts are wasted on them.
DEFAULT_NON_RETRYABLE: tuple[type[Exception], ...] = (InvalidStateTransitionError,)


def with_retry(
    node_fn: Callable[[GraphState], dict[str, Any]],
    *,
    max_retries: int,
    node_name: str,
    non_retryable: tuple[type[Exception], ...] = DEFAULT_NON_RETRYABLE,
) -> Callable[[GraphState], dict[str, Any]]:
    """Wrap a node function with a bounded retry policy.

    Args:
        node_fn: Any ``state -> dict`` node function (e.g. a `BaseAgent`
            instance's `.run`, or `supervisor_node`).
        max_retries: Maximum number of *retries* after the initial attempt
            (so up to ``max_retries + 1`` total calls to ``node_fn``).
            Typically sourced from `agentic_research.config.Settings.max_retries`.
        node_name: Identifies the wrapped node in recorded `ErrorRecord`s
            and in `RetryLimitExceededError`.
        non_retryable: Exception types that should propagate immediately
            without consuming a retry attempt. Defaults to
            `DEFAULT_NON_RETRYABLE`.

    Returns:
        A new ``state -> dict`` function with identical successful-case
        behavior to ``node_fn``, plus retry-on-failure handling. On a run
        that needed one or more retries before succeeding, the returned
        update's ``error_log`` is prefixed with one `ErrorRecord` per
        failed attempt, and ``retry_count`` reflects how many retries were
        needed.

    Raises (from the returned function, not from `with_retry` itself):
        RetryLimitExceededError: If ``node_fn`` still fails after
            ``max_retries`` retries.
        Exception: Any exception in ``non_retryable`` propagates
            immediately on its first occurrence, unretried.
    """

    def wrapped(state: GraphState) -> dict[str, Any]:
        """Run ``node_fn`` with retry-on-failure; see `with_retry` docstring."""
        attempt = 0
        error_records: list[ErrorRecord] = []

        while True:
            attempt += 1
            try:
                result = node_fn(state)
            except non_retryable:
                raise
            except Exception as exc:  # noqa: BLE001 - intentional: see module docstring
                error_records.append(
                    ErrorRecord(
                        node_name=node_name,
                        error_type=type(exc).__name__,
                        message=str(exc),
                        attempt=attempt,
                        recoverable=attempt <= max_retries,
                    )
                )
                if attempt > max_retries:
                    raise RetryLimitExceededError(
                        node_name=node_name, attempts=attempt, last_error=str(exc)
                    ) from exc
                continue
            else:
                if error_records:
                    result = dict(result)
                    result["error_log"] = [
                        *error_records,
                        *result.get("error_log", []),
                    ]
                    result["retry_count"] = attempt - 1
                return result

    return wrapped
