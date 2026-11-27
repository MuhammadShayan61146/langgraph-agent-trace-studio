"""Unit tests for ``agentic_research.memory.serialization``."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agentic_research.core.contracts import (
    AgentMessage,
    ErrorRecord,
    ResearchFinding,
    RoutingDecision,
)
from agentic_research.core.enums import AgentRole, MessageRole, TaskStatus
from agentic_research.core.exceptions import StateSerializationError
from agentic_research.core.state import create_initial_state
from agentic_research.memory.serialization import (
    state_from_dict,
    state_from_json,
    state_to_dict,
    state_to_json,
)

_FIXED_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _populated_state():
    state = create_initial_state("What is LangGraph?", requires_human_approval=True)
    state["messages"] = [
        AgentMessage(
            role=MessageRole.ASSISTANT,
            agent=AgentRole.RESEARCHER,
            content="Found some info.",
            metadata={"tokens": 42},
            timestamp=_FIXED_TIME,
        )
    ]
    state["research_findings"] = [
        ResearchFinding(
            source="https://example.com",
            content="LangGraph is a graph-based agent framework.",
            collected_by=AgentRole.RESEARCHER,
            confidence=0.9,
            timestamp=_FIXED_TIME,
        )
    ]
    state["routing_history"] = [
        RoutingDecision(
            from_agent=AgentRole.SUPERVISOR,
            to_agent=AgentRole.RESEARCHER,
            reason="Needs research.",
            timestamp=_FIXED_TIME,
        )
    ]
    state["error_log"] = [
        ErrorRecord(
            node_name="researcher",
            error_type="ToolExecutionFailure",
            message="timeout",
            attempt=2,
            recoverable=True,
            timestamp=_FIXED_TIME,
        )
    ]
    state["task_status"] = TaskStatus.RESEARCHING
    state["current_agent"] = AgentRole.RESEARCHER
    state["retry_count"] = 1
    state["final_report"] = None
    return state


# --------------------------------------------------------------------------- #
# Round trips
# --------------------------------------------------------------------------- #
def test_state_to_dict_and_back_round_trips_exactly() -> None:
    original = _populated_state()
    restored = state_from_dict(state_to_dict(original))

    assert restored["original_query"] == original["original_query"]
    assert restored["task_status"] == original["task_status"]
    assert restored["current_agent"] == original["current_agent"]
    assert restored["requires_human_approval"] == original["requires_human_approval"]

    assert restored["messages"][0].content == original["messages"][0].content
    assert restored["messages"][0].role == MessageRole.ASSISTANT
    assert restored["messages"][0].agent == AgentRole.RESEARCHER
    assert restored["messages"][0].metadata == {"tokens": 42}
    assert restored["messages"][0].timestamp == _FIXED_TIME

    assert restored["research_findings"][0].confidence == 0.9
    assert restored["routing_history"][0].to_agent == AgentRole.RESEARCHER
    assert restored["error_log"][0].attempt == 2


def test_state_to_json_and_back_round_trips_exactly() -> None:
    original = _populated_state()
    restored = state_from_json(state_to_json(original))
    assert restored["original_query"] == original["original_query"]
    assert restored["research_findings"][0].source == "https://example.com"


def test_empty_initial_state_round_trips() -> None:
    original = create_initial_state("simple query")
    restored = state_from_dict(state_to_dict(original))
    assert restored == original or (
        restored["original_query"] == original["original_query"]
        and restored["messages"] == []
    )


def test_agent_none_round_trips_as_none() -> None:
    """A USER message has agent=None; must survive the round trip as None,
    not as some sentinel string."""
    state = create_initial_state("query")
    state["messages"] = [AgentMessage(role=MessageRole.USER, content="hi")]
    restored = state_from_dict(state_to_dict(state))
    assert restored["messages"][0].agent is None


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
def test_state_from_json_rejects_invalid_json() -> None:
    with pytest.raises(StateSerializationError):
        state_from_json("{not valid json")


def test_state_from_dict_rejects_missing_required_field() -> None:
    data = state_to_dict(create_initial_state("query"))
    del data["original_query"]
    with pytest.raises(StateSerializationError):
        state_from_dict(data)


def test_state_from_dict_rejects_invalid_enum_value() -> None:
    data = state_to_dict(create_initial_state("query"))
    data["task_status"] = "not_a_real_status"
    with pytest.raises(StateSerializationError):
        state_from_dict(data)


def test_state_from_dict_rejects_malformed_nested_message() -> None:
    data = state_to_dict(create_initial_state("query"))
    data["messages"] = [{"role": "not_a_real_role", "content": "hi"}]
    with pytest.raises(StateSerializationError):
        state_from_dict(data)
