"""Core, framework-agnostic building blocks (exceptions, shared types).

Nothing in ``agentic_research.core`` may import LangGraph, LangChain, or any
LLM provider SDK. This keeps the core layer testable without those
dependencies installed and enforces the Dependency Inversion Principle:
higher layers (agents, graph) depend on core abstractions, never the reverse.
"""

from agentic_research.core.contracts import (
    AgentMessage,
    ErrorRecord,
    ResearchFinding,
    RoutingDecision,
)
from agentic_research.core.enums import (
    TASK_STATUS_TRANSITIONS,
    AgentRole,
    MessageRole,
    TaskStatus,
)
from agentic_research.core.exceptions import (
    AgentError,
    AgentInvocationError,
    AgenticResearchError,
    ConfigurationError,
    ContractValidationError,
    GraphExecutionError,
    InvalidStateTransitionError,
    MissingEnvironmentVariableError,
    RetryLimitExceededError,
    RoutingError,
    StateError,
    StateSerializationError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
)
from agentic_research.core.state import (
    GraphState,
    accumulate_list,
    create_initial_state,
    validate_transition,
)

__all__ = [
    # exceptions
    "AgenticResearchError",
    "ConfigurationError",
    "MissingEnvironmentVariableError",
    "ContractValidationError",
    "StateError",
    "InvalidStateTransitionError",
    "StateSerializationError",
    "AgentError",
    "AgentInvocationError",
    "ToolError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "GraphExecutionError",
    "RoutingError",
    "RetryLimitExceededError",
    # enums
    "MessageRole",
    "AgentRole",
    "TaskStatus",
    "TASK_STATUS_TRANSITIONS",
    # contracts
    "AgentMessage",
    "ResearchFinding",
    "RoutingDecision",
    "ErrorRecord",
    # state
    "GraphState",
    "accumulate_list",
    "create_initial_state",
    "validate_transition",
]
