"""Unit tests for ``agentic_research.core.contracts``."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agentic_research.core.contracts import (
    AgentMessage,
    ErrorRecord,
    ResearchFinding,
    RoutingDecision,
)
from agentic_research.core.enums import AgentRole, MessageRole
from agentic_research.core.exceptions import ContractValidationError

FIXED_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# AgentMessage
# --------------------------------------------------------------------------- #
def test_agent_message_constructs_with_defaults() -> None:
    msg = AgentMessage(role=MessageRole.USER, content="hello")
    assert msg.content == "hello"
    assert msg.agent is None
    assert msg.metadata == {}
    assert isinstance(msg.timestamp, datetime)


def test_agent_message_rejects_empty_content() -> None:
    with pytest.raises(ContractValidationError):
        AgentMessage(role=MessageRole.USER, content="")


def test_agent_message_rejects_whitespace_only_content() -> None:
    with pytest.raises(ContractValidationError):
        AgentMessage(role=MessageRole.USER, content="   ")


def test_agent_message_is_immutable() -> None:
    msg = AgentMessage(role=MessageRole.USER, content="hello")
    with pytest.raises(Exception):
        msg.content = "changed"  # type: ignore[misc]


def test_agent_message_accepts_explicit_timestamp_for_determinism() -> None:
    msg = AgentMessage(role=MessageRole.ASSISTANT, content="hi", timestamp=FIXED_TIME)
    assert msg.timestamp == FIXED_TIME


# --------------------------------------------------------------------------- #
# ResearchFinding
# --------------------------------------------------------------------------- #
def test_research_finding_constructs_with_valid_values() -> None:
    finding = ResearchFinding(
        source="https://example.com",
        content="Some fact.",
        collected_by=AgentRole.RESEARCHER,
        confidence=0.8,
    )
    assert finding.confidence == 0.8


def test_research_finding_rejects_empty_source() -> None:
    with pytest.raises(ContractValidationError):
        ResearchFinding(
            source="", content="fact", collected_by=AgentRole.RESEARCHER
        )


def test_research_finding_rejects_empty_content() -> None:
    with pytest.raises(ContractValidationError):
        ResearchFinding(
            source="src", content="", collected_by=AgentRole.RESEARCHER
        )


def test_research_finding_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ContractValidationError):
        ResearchFinding(
            source="src",
            content="fact",
            collected_by=AgentRole.RESEARCHER,
            confidence=1.5,
        )
    with pytest.raises(ContractValidationError):
        ResearchFinding(
            source="src",
            content="fact",
            collected_by=AgentRole.RESEARCHER,
            confidence=-0.1,
        )


def test_research_finding_default_confidence_is_full() -> None:
    finding = ResearchFinding(
        source="src", content="fact", collected_by=AgentRole.RESEARCHER
    )
    assert finding.confidence == 1.0


# --------------------------------------------------------------------------- #
# RoutingDecision
# --------------------------------------------------------------------------- #
def test_routing_decision_constructs_with_valid_values() -> None:
    decision = RoutingDecision(
        from_agent=AgentRole.SUPERVISOR,
        to_agent=AgentRole.RESEARCHER,
        reason="Query requires factual lookup.",
    )
    assert decision.to_agent == AgentRole.RESEARCHER


def test_routing_decision_rejects_empty_reason() -> None:
    with pytest.raises(ContractValidationError):
        RoutingDecision(
            from_agent=AgentRole.SUPERVISOR, to_agent=AgentRole.RESEARCHER, reason=""
        )


# --------------------------------------------------------------------------- #
# ErrorRecord
# --------------------------------------------------------------------------- #
def test_error_record_constructs_with_defaults() -> None:
    err = ErrorRecord(
        node_name="researcher", error_type="ToolExecutionError", message="timeout"
    )
    assert err.attempt == 1
    assert err.recoverable is True


def test_error_record_rejects_empty_node_name() -> None:
    with pytest.raises(ContractValidationError):
        ErrorRecord(node_name="", error_type="ToolExecutionError", message="timeout")


def test_error_record_rejects_non_positive_attempt() -> None:
    with pytest.raises(ContractValidationError):
        ErrorRecord(
            node_name="researcher",
            error_type="ToolExecutionError",
            message="timeout",
            attempt=0,
        )
