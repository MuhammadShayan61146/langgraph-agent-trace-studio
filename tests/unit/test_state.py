"""Unit tests for ``agentic_research.core.state``.

"""

from __future__ import annotations

import typing

import pytest

from agentic_research.core.contracts import AgentMessage, ResearchFinding
from agentic_research.core.enums import AgentRole, MessageRole, TaskStatus
from agentic_research.core.exceptions import InvalidStateTransitionError
from agentic_research.core.state import (
    GraphState,
    accumulate_list,
    create_initial_state,
    validate_transition,
)


# --------------------------------------------------------------------------- #
# accumulate_list reducer
# --------------------------------------------------------------------------- #
def test_accumulate_list_concatenates_in_order() -> None:
    assert accumulate_list([1, 2], [3, 4]) == [1, 2, 3, 4]


def test_accumulate_list_handles_empty_existing() -> None:
    assert accumulate_list([], [1]) == [1]


def test_accumulate_list_handles_empty_new() -> None:
    assert accumulate_list([1], []) == [1]


def test_accumulate_list_does_not_mutate_inputs() -> None:
    existing = [1, 2]
    new = [3]
    accumulate_list(existing, new)
    assert existing == [1, 2]
    assert new == [3]


def test_accumulate_list_works_with_dataclass_instances() -> None:
    msg1 = AgentMessage(role=MessageRole.USER, content="hi")
    msg2 = AgentMessage(role=MessageRole.ASSISTANT, content="hello")
    result = accumulate_list([msg1], [msg2])
    assert result == [msg1, msg2]


# --------------------------------------------------------------------------- #
# GraphState schema shape
# --------------------------------------------------------------------------- #
def test_graph_state_annotations_mark_list_fields_with_accumulate_list() -> None:
    """Confirm the reducer wiring on GraphState is actually what we expect.

    This directly inspects TypedDict's __annotations__/Annotated metadata,
    which is exactly what langgraph.graph.StateGraph reads at compile time
    to decide how to merge partial state updates -- so this test is a real
    (if indirect) check of Module 5 compatibility.
    """
    hints = typing.get_type_hints(GraphState, include_extras=True)
    for field_name in (
        "messages",
        "research_findings",
        "routing_history",
        "error_log",
    ):
        annotation = hints[field_name]
        assert typing.get_origin(annotation) is typing.Annotated
        metadata = annotation.__metadata__
        assert accumulate_list in metadata, f"{field_name} missing accumulate_list"


def test_graph_state_scalar_fields_have_no_reducer_annotation() -> None:
    hints = typing.get_type_hints(GraphState, include_extras=True)
    for field_name in ("current_agent", "task_status", "retry_count"):
        annotation = hints[field_name]
        assert typing.get_origin(annotation) is not typing.Annotated


# --------------------------------------------------------------------------- #
# create_initial_state
# --------------------------------------------------------------------------- #
def test_create_initial_state_sets_expected_defaults() -> None:
    state = create_initial_state("What is LangGraph?")
    assert state["original_query"] == "What is LangGraph?"
    assert state["messages"] == []
    assert state["research_findings"] == []
    assert state["routing_history"] == []
    assert state["error_log"] == []
    assert state["current_agent"] == AgentRole.SUPERVISOR
    assert state["task_status"] == TaskStatus.PENDING
    assert state["retry_count"] == 0
    assert state["requires_human_approval"] is False
    assert state["human_approval_status"] is None
    assert state["final_report"] is None


def test_create_initial_state_honors_requires_human_approval_flag() -> None:
    state = create_initial_state("query", requires_human_approval=True)
    assert state["requires_human_approval"] is True


def test_create_initial_state_rejects_empty_query() -> None:
    with pytest.raises(ValueError):
        create_initial_state("")
    with pytest.raises(ValueError):
        create_initial_state("   ")


def test_initial_state_is_a_plain_dict_at_runtime() -> None:
    """TypedDict instances are plain dicts at runtime -- confirm that holds,
    since langgraph's merge machinery operates on plain dict updates."""
    state = create_initial_state("query")
    assert isinstance(state, dict)


def test_state_can_simulate_a_node_partial_update_via_reducer() -> None:
    """Simulate what LangGraph does internally: take existing state's list
    field, merge it with a node's partial-update list using the declared
    reducer, exactly as langgraph.graph.StateGraph would at runtime."""
    state = create_initial_state("query")
    node_update = {
        "research_findings": [
            ResearchFinding(
                source="src", content="fact", collected_by=AgentRole.RESEARCHER
            )
        ]
    }
    merged = accumulate_list(
        state["research_findings"], node_update["research_findings"]
    )
    assert len(merged) == 1
    assert merged[0].source == "src"


# --------------------------------------------------------------------------- #
# validate_transition
# --------------------------------------------------------------------------- #
def test_validate_transition_allows_legal_move() -> None:
    validate_transition(TaskStatus.PENDING, TaskStatus.ROUTING)  # must not raise


def test_validate_transition_rejects_illegal_move() -> None:
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(TaskStatus.PENDING, TaskStatus.COMPLETED)


def test_validate_transition_rejects_moves_from_terminal_state() -> None:
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(TaskStatus.COMPLETED, TaskStatus.ROUTING)


def test_validate_transition_error_carries_both_statuses() -> None:
    try:
        validate_transition(TaskStatus.PENDING, TaskStatus.COMPLETED)
        assert False, "expected InvalidStateTransitionError"
    except InvalidStateTransitionError as exc:
        assert exc.from_status == "pending"
        assert exc.to_status == "completed"
