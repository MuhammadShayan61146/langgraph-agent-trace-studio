"""Unit tests for ``agentic_research.graph.supervisor``."""

from __future__ import annotations

import pytest

from agentic_research.core.contracts import AgentMessage
from agentic_research.core.enums import AgentRole, MessageRole, TaskStatus
from agentic_research.core.exceptions import InvalidStateTransitionError
from agentic_research.core.state import create_initial_state
from agentic_research.graph.supervisor import (
    END_SENTINEL,
    decide_next_agent,
    route_after_supervisor,
    supervisor_node,
)


def _message_from(agent: AgentRole) -> AgentMessage:
    return AgentMessage(role=MessageRole.ASSISTANT, agent=agent, content="done")


# --------------------------------------------------------------------------- #
# decide_next_agent
# --------------------------------------------------------------------------- #
def test_decide_next_agent_with_no_participants_returns_researcher() -> None:
    state = create_initial_state("query")
    assert decide_next_agent(state) == AgentRole.RESEARCHER


def test_decide_next_agent_after_researcher_returns_analyst() -> None:
    state = create_initial_state("query")
    state["messages"] = [_message_from(AgentRole.RESEARCHER)]
    assert decide_next_agent(state) == AgentRole.ANALYST


def test_decide_next_agent_after_researcher_and_analyst_returns_writer() -> None:
    state = create_initial_state("query")
    state["messages"] = [
        _message_from(AgentRole.RESEARCHER),
        _message_from(AgentRole.ANALYST),
    ]
    assert decide_next_agent(state) == AgentRole.WRITER


def test_decide_next_agent_after_all_three_returns_human() -> None:
    state = create_initial_state("query")
    state["messages"] = [
        _message_from(AgentRole.RESEARCHER),
        _message_from(AgentRole.ANALYST),
        _message_from(AgentRole.WRITER),
    ]
    assert decide_next_agent(state) == AgentRole.HUMAN


def test_decide_next_agent_ignores_messages_with_no_agent() -> None:
    state = create_initial_state("query")
    state["messages"] = [AgentMessage(role=MessageRole.USER, content="hi")]
    assert decide_next_agent(state) == AgentRole.RESEARCHER


# --------------------------------------------------------------------------- #
# supervisor_node - first turn (PENDING)
# --------------------------------------------------------------------------- #
def test_supervisor_node_first_turn_routes_to_researcher() -> None:
    state = create_initial_state("query")  # task_status defaults to PENDING
    update = supervisor_node(state)

    assert update["task_status"] == TaskStatus.RESEARCHING
    assert update["current_agent"] == AgentRole.SUPERVISOR
    assert len(update["routing_history"]) == 1
    assert update["routing_history"][0].to_agent == AgentRole.RESEARCHER


# --------------------------------------------------------------------------- #
# supervisor_node - mid-flow turns
# --------------------------------------------------------------------------- #
def test_supervisor_node_after_researcher_routes_to_analyst() -> None:
    state = create_initial_state("query")
    state["task_status"] = TaskStatus.ROUTING
    state["messages"] = [_message_from(AgentRole.RESEARCHER)]

    update = supervisor_node(state)
    assert update["task_status"] == TaskStatus.ANALYZING
    assert update["routing_history"][0].to_agent == AgentRole.ANALYST


def test_supervisor_node_after_analyst_routes_to_writer() -> None:
    state = create_initial_state("query")
    state["task_status"] = TaskStatus.ROUTING
    state["messages"] = [
        _message_from(AgentRole.RESEARCHER),
        _message_from(AgentRole.ANALYST),
    ]

    update = supervisor_node(state)
    assert update["task_status"] == TaskStatus.WRITING
    assert update["routing_history"][0].to_agent == AgentRole.WRITER


# --------------------------------------------------------------------------- #
# supervisor_node - approval gate auto-advance
# --------------------------------------------------------------------------- #
def test_supervisor_node_auto_completes_when_approval_not_required() -> None:
    state = create_initial_state("query", requires_human_approval=False)
    state["task_status"] = TaskStatus.AWAITING_HUMAN_APPROVAL

    update = supervisor_node(state)
    assert update["task_status"] == TaskStatus.COMPLETED
    assert update["human_approval_status"] == "approved"


def test_supervisor_node_parks_when_approval_required() -> None:
    state = create_initial_state("query", requires_human_approval=True)
    state["task_status"] = TaskStatus.AWAITING_HUMAN_APPROVAL

    update = supervisor_node(state)
    assert "task_status" not in update
    assert "human_approval_status" not in update


# --------------------------------------------------------------------------- #
# supervisor_node - illegal state raises rather than silently proceeding
# --------------------------------------------------------------------------- #
def test_supervisor_node_raises_on_illegal_transition() -> None:
    """COMPLETED is terminal; if it somehow still lacks a Researcher message
    (an invalid/corrupted state), the supervisor must refuse to proceed
    rather than silently routing anyway."""
    state = create_initial_state("query")
    state["task_status"] = TaskStatus.COMPLETED

    with pytest.raises(InvalidStateTransitionError):
        supervisor_node(state)


# --------------------------------------------------------------------------- #
# route_after_supervisor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "status,expected_node",
    [
        (TaskStatus.RESEARCHING, "researcher"),
        (TaskStatus.ANALYZING, "analyst"),
        (TaskStatus.WRITING, "writer"),
        (TaskStatus.COMPLETED, END_SENTINEL),
        (TaskStatus.AWAITING_HUMAN_APPROVAL, END_SENTINEL),
        (TaskStatus.PENDING, END_SENTINEL),
    ],
)
def test_route_after_supervisor_maps_status_to_expected_node(
    status: TaskStatus, expected_node: str
) -> None:
    state = create_initial_state("query")
    state["task_status"] = status
    assert route_after_supervisor(state) == expected_node
