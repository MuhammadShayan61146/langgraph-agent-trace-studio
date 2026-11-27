"""Core data contracts: the immutable records that flow through graph state.

Purpose
-------
Defines every structured record type that will live inside the LangGraph
``GraphState`` (Module 2's ``state.py``): conversation messages, research
findings, routing decisions, and error records. These are the "nouns" of
the system; ``state.py`` defines how collections of them accumulate.

Responsibilities
-----------------
* Provide one dataclass per concern, each independently constructible and
  independently testable.
* Validate field values eagerly (fail at construction, not later when
  something tries to render or log a malformed record).

Design decisions
-----------------
* **All contracts are frozen dataclasses.** Once a message was sent, a
  finding was recorded, or a routing decision was made, it is a historical
  fact and must not be mutated in place — only appended to the state's
  list-typed fields. This mirrors event-sourcing style design and makes
  concurrent/replayed graph execution safer to reason about (no node can
  accidentally rewrite another node's prior output).
* **Validation lives in ``__post_init__``**, consistent with the pattern
  established in Module 1's ``Settings``. Errors raise
  ``ContractValidationError`` (added to ``core.exceptions`` in this module)
  rather than a bare ``ValueError``, keeping every application-raised error
  under the ``AgenticResearchError`` root per Module 1's exception design.
* **Timestamps default to "now" via ``default_factory``** but can be
  injected explicitly. This is a small but deliberate Dependency Injection
  point: tests need deterministic timestamps, and later modules (replay /
  audit tooling) need to reconstruct records with historical timestamps.
* **``confidence`` on ``ResearchFinding`` is a plain float in [0.0, 1.0]**
  rather than a richer type, matching the same simplicity-first tradeoff
  documented for ``Settings`` in Module 1 (no pydantic available/desired
  here).

Tradeoffs
---------
* Frozen dataclasses mean "updating" a record means constructing a new one
  (e.g. re-scoring a finding's confidence). This is intentional friction —
  it forces state changes to go through the state layer's reducers instead
  of ad-hoc mutation, at the cost of a small amount of object churn.

Limitations
------------
* No schema versioning yet. If a record's shape changes in a later module,
  old checkpointed state (once persistence exists, Module 6) may not
  deserialize cleanly. Flagged as a known risk in the Module 2 report.

Future improvements
--------------------
* Add a ``schema_version`` field once real persistence (Module 6, memory)
  makes backward compatibility a concrete concern rather than a
  theoretical one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agentic_research.core.enums import AgentRole, MessageRole
from agentic_research.core.exceptions import ContractValidationError


def _utc_now() -> datetime:
    """Return the current UTC time.

    Extracted as a standalone function (rather than inlined in every
    ``default_factory=...``) so it is one single, patchable seam if a future
    module needs to freeze time in integration tests.
    """
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class AgentMessage:
    """A single message in the shared conversation log.

    Attributes:
        role: Who authored the message (system/user/assistant/tool).
        content: The message text. Must be non-empty.
        agent: Which agent produced this message, if ``role`` is
            ``ASSISTANT`` or ``TOOL``. ``None`` for ``USER``/``SYSTEM``
            messages.
        metadata: Free-form structured context (e.g. token usage, tool call
            id). Defaults to an empty dict, never ``None``, so callers never
            need a null check.
        timestamp: When the message was created (UTC). Defaults to now.
    """

    role: MessageRole
    content: str
    agent: AgentRole | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        """Validate that the message content is non-empty."""
        if not self.content or not self.content.strip():
            raise ContractValidationError(
                "AgentMessage.content must be a non-empty string.",
                details={"role": self.role},
            )


@dataclass(frozen=True, slots=True)
class ResearchFinding:
    """A single piece of information collected during research.

    Attributes:
        source: Where the finding came from (URL, document name, tool
            name). Must be non-empty.
        content: The finding itself, in plain text. Must be non-empty.
        collected_by: Which agent collected this finding.
        confidence: Self-assessed confidence in [0.0, 1.0] that the finding
            is accurate and relevant.
        timestamp: When the finding was collected (UTC). Defaults to now.
    """

    source: str
    content: str
    collected_by: AgentRole
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        """Validate source/content non-emptiness and confidence range."""
        errors: list[str] = []
        if not self.source or not self.source.strip():
            errors.append("source must be a non-empty string")
        if not self.content or not self.content.strip():
            errors.append("content must be a non-empty string")
        if not (0.0 <= self.confidence <= 1.0):
            errors.append(
                f"confidence must be within [0.0, 1.0], got {self.confidence!r}"
            )
        if errors:
            raise ContractValidationError(
                "Invalid ResearchFinding: " + "; ".join(errors),
                details={"errors": errors},
            )


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """A record of one supervisor routing decision.

    Attributes:
        from_agent: The agent (or supervisor) making the routing call.
        to_agent: The agent selected to act next.
        reason: A short human-readable justification, for auditability and
            debugging conditional routing logic.
        timestamp: When the decision was made (UTC). Defaults to now.
    """

    from_agent: AgentRole
    to_agent: AgentRole
    reason: str
    timestamp: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        """Validate that a justification for the routing decision was given."""
        if not self.reason or not self.reason.strip():
            raise ContractValidationError(
                "RoutingDecision.reason must be a non-empty string.",
                details={"from_agent": self.from_agent, "to_agent": self.to_agent},
            )


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    """A record of one recoverable (or terminal) error encountered by a node.

    Attributes:
        node_name: Name of the graph node where the error occurred.
        error_type: The exception class name (e.g. ``"ToolExecutionError"``).
        message: The error message.
        attempt: Which retry attempt this was (1-indexed).
        recoverable: Whether the graph attempted (or will attempt) to
            recover from this error, as opposed to it being terminal.
        timestamp: When the error occurred (UTC). Defaults to now.
    """

    node_name: str
    error_type: str
    message: str
    attempt: int = 1
    recoverable: bool = True
    timestamp: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        """Validate node_name non-emptiness and attempt is a positive integer."""
        errors: list[str] = []
        if not self.node_name or not self.node_name.strip():
            errors.append("node_name must be a non-empty string")
        if self.attempt < 1:
            errors.append(f"attempt must be >= 1, got {self.attempt!r}")
        if errors:
            raise ContractValidationError(
                "Invalid ErrorRecord: " + "; ".join(errors),
                details={"errors": errors},
            )
