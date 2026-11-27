"""Serialization: converting `GraphState` and its contracts to/from JSON.

Purpose
-------
Every persistence mechanism in this module (checkpointing, long-term
memory) needs to turn `GraphState` into something writable (JSON) and
back. This is the one place that logic lives, so checkpoint.py and
long_term.py both depend on it rather than each inventing their own
(de)serialization (DRY).

Design decisions
-----------------
* **Per-contract explicit to_dict/from_dict pairs, not
  `dataclasses.asdict`.** `asdict` would handle the happy path but loses
  enum/datetime type information on the way back in — `from_dict` needs to
  reconstruct real `MessageRole`/`AgentRole`/`datetime` values, not leave
  them as strings, or downstream code (agents, the supervisor) would break
  on `msg.role == MessageRole.USER`-style comparisons. Explicit functions
  make that reconstruction visible and testable per-field, rather than
  hidden in a generic recursive converter.
* **Deserialization failures raise `StateSerializationError`** (added to
  `core.exceptions` in this module), never a bare `KeyError`/`ValueError`,
  consistent with every prior module's fail-fast-with-a-typed-exception
  discipline.
* **`state_to_json`/`state_from_json` are thin wrappers around
  `state_to_dict`/`state_from_dict`.** Keeping the dict-level functions
  separate lets `checkpoint.py` reuse them directly against
  `JSONFileStore` (which already deals in dicts) without a redundant
  JSON-encode-then-immediately-decode round trip.

Limitations
------------
* No schema versioning (flagged as a risk back in Module 2's contracts
  docstring — this is where it becomes concrete). A future field added to
  `GraphState` will break `state_from_dict` on old persisted checkpoints
  unless a migration path is added then.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from agentic_research.core.contracts import (
    AgentMessage,
    ErrorRecord,
    ResearchFinding,
    RoutingDecision,
)
from agentic_research.core.enums import AgentRole, MessageRole, TaskStatus
from agentic_research.core.exceptions import StateSerializationError
from agentic_research.core.state import GraphState

# --------------------------------------------------------------------------- #
# AgentMessage
# --------------------------------------------------------------------------- #
def _agent_message_to_dict(message: AgentMessage) -> dict[str, Any]:
    """Convert an `AgentMessage` into a JSON-serializable dict."""
    return {
        "role": message.role.value,
        "content": message.content,
        "agent": message.agent.value if message.agent is not None else None,
        "metadata": message.metadata,
        "timestamp": message.timestamp.isoformat(),
    }


def _agent_message_from_dict(data: dict[str, Any]) -> AgentMessage:
    """Reconstruct an `AgentMessage` from its serialized dict form.

    Raises:
        StateSerializationError: If a required field is missing or a
            role/agent value isn't a recognized enum member.
    """
    try:
        return AgentMessage(
            role=MessageRole(data["role"]),
            content=data["content"],
            agent=AgentRole(data["agent"]) if data.get("agent") is not None else None,
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )
    except (KeyError, ValueError) as exc:
        raise StateSerializationError(
            f"Invalid AgentMessage payload: {exc}", details={"payload": data}
        ) from exc


# --------------------------------------------------------------------------- #
# ResearchFinding
# --------------------------------------------------------------------------- #
def _research_finding_to_dict(finding: ResearchFinding) -> dict[str, Any]:
    """Convert a `ResearchFinding` into a JSON-serializable dict."""
    return {
        "source": finding.source,
        "content": finding.content,
        "collected_by": finding.collected_by.value,
        "confidence": finding.confidence,
        "timestamp": finding.timestamp.isoformat(),
    }


def _research_finding_from_dict(data: dict[str, Any]) -> ResearchFinding:
    """Reconstruct a `ResearchFinding` from its serialized dict form.

    Raises:
        StateSerializationError: If a required field is missing or invalid.
    """
    try:
        return ResearchFinding(
            source=data["source"],
            content=data["content"],
            collected_by=AgentRole(data["collected_by"]),
            confidence=data["confidence"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )
    except (KeyError, ValueError) as exc:
        raise StateSerializationError(
            f"Invalid ResearchFinding payload: {exc}", details={"payload": data}
        ) from exc


# --------------------------------------------------------------------------- #
# RoutingDecision
# --------------------------------------------------------------------------- #
def _routing_decision_to_dict(decision: RoutingDecision) -> dict[str, Any]:
    """Convert a `RoutingDecision` into a JSON-serializable dict."""
    return {
        "from_agent": decision.from_agent.value,
        "to_agent": decision.to_agent.value,
        "reason": decision.reason,
        "timestamp": decision.timestamp.isoformat(),
    }


def _routing_decision_from_dict(data: dict[str, Any]) -> RoutingDecision:
    """Reconstruct a `RoutingDecision` from its serialized dict form.

    Raises:
        StateSerializationError: If a required field is missing or invalid.
    """
    try:
        return RoutingDecision(
            from_agent=AgentRole(data["from_agent"]),
            to_agent=AgentRole(data["to_agent"]),
            reason=data["reason"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )
    except (KeyError, ValueError) as exc:
        raise StateSerializationError(
            f"Invalid RoutingDecision payload: {exc}", details={"payload": data}
        ) from exc


# --------------------------------------------------------------------------- #
# ErrorRecord
# --------------------------------------------------------------------------- #
def _error_record_to_dict(record: ErrorRecord) -> dict[str, Any]:
    """Convert an `ErrorRecord` into a JSON-serializable dict."""
    return {
        "node_name": record.node_name,
        "error_type": record.error_type,
        "message": record.message,
        "attempt": record.attempt,
        "recoverable": record.recoverable,
        "timestamp": record.timestamp.isoformat(),
    }


def _error_record_from_dict(data: dict[str, Any]) -> ErrorRecord:
    """Reconstruct an `ErrorRecord` from its serialized dict form.

    Raises:
        StateSerializationError: If a required field is missing or invalid.
    """
    try:
        return ErrorRecord(
            node_name=data["node_name"],
            error_type=data["error_type"],
            message=data["message"],
            attempt=data["attempt"],
            recoverable=data["recoverable"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )
    except (KeyError, ValueError) as exc:
        raise StateSerializationError(
            f"Invalid ErrorRecord payload: {exc}", details={"payload": data}
        ) from exc


# --------------------------------------------------------------------------- #
# GraphState
# --------------------------------------------------------------------------- #
def state_to_dict(state: GraphState) -> dict[str, Any]:
    """Convert a `GraphState` into a JSON-serializable plain dict.

    Args:
        state: The graph state to serialize.

    Returns:
        A plain dict containing only JSON-native types (str, int, float,
        bool, None, list, dict).
    """
    return {
        "original_query": state["original_query"],
        "messages": [_agent_message_to_dict(m) for m in state["messages"]],
        "research_findings": [
            _research_finding_to_dict(f) for f in state["research_findings"]
        ],
        "routing_history": [
            _routing_decision_to_dict(r) for r in state["routing_history"]
        ],
        "error_log": [_error_record_to_dict(e) for e in state["error_log"]],
        "current_agent": state["current_agent"].value,
        "task_status": state["task_status"].value,
        "retry_count": state["retry_count"],
        "requires_human_approval": state["requires_human_approval"],
        "human_approval_status": state["human_approval_status"],
        "final_report": state["final_report"],
    }


def state_from_dict(data: dict[str, Any]) -> GraphState:
    """Reconstruct a `GraphState` from its serialized dict form.

    Args:
        data: A dict previously produced by `state_to_dict` (or an
            equivalent JSON payload).

    Returns:
        A fully reconstructed `GraphState`, with every contract, enum, and
        timestamp restored to its real type.

    Raises:
        StateSerializationError: If a required top-level field is missing,
            or any nested contract fails to reconstruct.
    """
    try:
        return GraphState(
            original_query=data["original_query"],
            messages=[_agent_message_from_dict(m) for m in data["messages"]],
            research_findings=[
                _research_finding_from_dict(f) for f in data["research_findings"]
            ],
            routing_history=[
                _routing_decision_from_dict(r) for r in data["routing_history"]
            ],
            error_log=[_error_record_from_dict(e) for e in data["error_log"]],
            current_agent=AgentRole(data["current_agent"]),
            task_status=TaskStatus(data["task_status"]),
            retry_count=data["retry_count"],
            requires_human_approval=data["requires_human_approval"],
            human_approval_status=data.get("human_approval_status"),
            final_report=data.get("final_report"),
        )
    except (KeyError, ValueError) as exc:
        raise StateSerializationError(
            f"Invalid GraphState payload: missing or invalid field {exc}",
            details={"payload_keys": list(data.keys())},
        ) from exc


def state_to_json(state: GraphState) -> str:
    """Serialize a `GraphState` directly to a JSON string.

    Args:
        state: The graph state to serialize.

    Returns:
        A JSON string, indented for human readability (checkpoint files are
        meant to be diffable/inspectable during development).
    """
    return json.dumps(state_to_dict(state), indent=2)


def state_from_json(json_str: str) -> GraphState:
    """Deserialize a `GraphState` directly from a JSON string.

    Args:
        json_str: A JSON string previously produced by `state_to_json`.

    Returns:
        The reconstructed `GraphState`.

    Raises:
        StateSerializationError: If ``json_str`` is not valid JSON, or the
            parsed payload fails `state_from_dict`'s validation.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise StateSerializationError(f"Invalid JSON: {exc}") from exc
    return state_from_dict(data)
