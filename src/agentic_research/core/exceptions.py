"""Application-wide exception hierarchy.

Purpose
-------
Centralizes every custom exception type used across the codebase so that
callers (CLI, API layer, graph nodes) can catch failures at the right level
of granularity instead of bare ``except Exception``.

Design decisions
-----------------
* A single root, ``AgenticResearchError``, lets any layer catch "anything
  that went wrong inside this application" without swallowing unrelated
  third-party or stdlib errors.
* Sub-hierarchies (``ConfigurationError``, ``AgentError``, ``ToolError``,
  ``GraphExecutionError``) map 1:1 onto the architectural layers (Module 1
  config, Module 4 agents, Module 3 tools, Module 5+ graph orchestration) so
  the exception module can grow alongside the project without becoming a
  dumping ground.
* Every exception accepts an optional ``details: dict`` so structured
  logging (Module 1 logging_config) can attach machine-readable context
  without callers needing to format strings themselves.

Tradeoffs
---------
Using a shallow, purpose-built hierarchy instead of a generic
``ApplicationError(code, message)`` trades a small amount of flexibility for
much better static discoverability (``except ToolExecutionError`` is
self-documenting) and IDE support.

Future improvements
--------------------
* Add error codes (enum) once the API surface (Module 9) needs to serialize
  errors to clients in a stable, versioned format.
"""

from __future__ import annotations

from typing import Any


class AgenticResearchError(Exception):
    """Root exception for every error raised by this application.

    Args:
        message: Human-readable description of the failure.
        details: Optional structured context (e.g. ``{"agent": "researcher"}``)
            useful for logging and debugging without parsing free-text
            messages.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the error with a message and optional structured details."""
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        """Return the message, appending details if any were provided."""
        if self.details:
            return f"{self.message} | details={self.details}"
        return self.message


# --------------------------------------------------------------------------- #
# Configuration layer (Module 1)
# --------------------------------------------------------------------------- #
class ConfigurationError(AgenticResearchError):
    """Raised when application configuration is missing or invalid."""


class MissingEnvironmentVariableError(ConfigurationError):
    """Raised when a required environment variable is absent."""

    def __init__(self, variable_name: str) -> None:
        """Build the error for a specific missing environment variable name."""
        super().__init__(
            message=f"Required environment variable '{variable_name}' is not set.",
            details={"variable_name": variable_name},
        )
        self.variable_name = variable_name


# --------------------------------------------------------------------------- #
# Data contract layer (Module 2)
# --------------------------------------------------------------------------- #
class ContractValidationError(AgenticResearchError):
    """Raised when a core data contract (e.g. ``AgentMessage``) is constructed
    with invalid field values."""


class StateError(AgenticResearchError):
    """Raised for violations of the graph state's structural or lifecycle rules."""


class InvalidStateTransitionError(StateError):
    """Raised when a ``TaskStatus`` transition is not permitted."""

    def __init__(self, from_status: str, to_status: str) -> None:
        """Build the error describing the illegal status transition attempted."""
        super().__init__(
            message=(
                f"Illegal task status transition: '{from_status}' -> '{to_status}'"
            ),
            details={"from_status": from_status, "to_status": to_status},
        )
        self.from_status = from_status
        self.to_status = to_status


class StateSerializationError(StateError):
    """Raised when ``GraphState`` or a core data contract fails to
    serialize to or deserialize from a persisted representation (Module 6
    memory layer)."""


# --------------------------------------------------------------------------- #
# Agent layer (Module 4+)
# --------------------------------------------------------------------------- #
class AgentError(AgenticResearchError):
    """Raised for failures originating inside an agent's reasoning loop."""


class AgentInvocationError(AgentError):
    """Raised when an underlying LLM call made by an agent fails."""


# --------------------------------------------------------------------------- #
# Tool layer (Module 3)
# --------------------------------------------------------------------------- #
class ToolError(AgenticResearchError):
    """Raised for failures related to tool registration or execution."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool name is not registered."""

    def __init__(self, tool_name: str, available_tools: list[str]) -> None:
        """Build the error listing the missing tool and what is registered."""
        super().__init__(
            message=f"Tool '{tool_name}' is not registered.",
            details={"tool_name": tool_name, "available_tools": available_tools},
        )
        self.tool_name = tool_name


class ToolExecutionError(ToolError):
    """Raised when a tool raises during execution."""


# --------------------------------------------------------------------------- #
# Graph orchestration layer (Module 5+)
# --------------------------------------------------------------------------- #
class GraphExecutionError(AgenticResearchError):
    """Raised when the LangGraph state graph fails to execute a step."""


class RoutingError(GraphExecutionError):
    """Raised when conditional routing logic cannot determine a next node."""


class RetryLimitExceededError(GraphExecutionError):
    """Raised when a recoverable error persists past the configured retry limit."""

    def __init__(self, node_name: str, attempts: int, last_error: str) -> None:
        """Build the error with the failing node's name, attempt count, and cause."""
        super().__init__(
            message=(
                f"Node '{node_name}' failed after {attempts} attempt(s); "
                f"last error: {last_error}"
            ),
            details={
                "node_name": node_name,
                "attempts": attempts,
                "last_error": last_error,
            },
        )
        self.node_name = node_name
        self.attempts = attempts
