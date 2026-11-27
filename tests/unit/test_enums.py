"""Unit tests for ``agentic_research.core.enums``."""

from __future__ import annotations

from agentic_research.core.enums import TASK_STATUS_TRANSITIONS, AgentRole, TaskStatus


def test_message_role_values_are_lowercase_strings() -> None:
    from agentic_research.core.enums import MessageRole

    assert MessageRole.USER == "user"
    assert MessageRole.ASSISTANT.value == "assistant"


def test_agent_role_members_are_string_comparable() -> None:
    assert AgentRole.SUPERVISOR == "supervisor"
    assert AgentRole.RESEARCHER.value == "researcher"


def test_every_task_status_appears_as_a_key_in_transition_table() -> None:
    """Every enum member must have (even if empty) a transition entry."""
    for status in TaskStatus:
        assert status in TASK_STATUS_TRANSITIONS, f"{status} missing from table"


def test_completed_is_a_terminal_state() -> None:
    assert TASK_STATUS_TRANSITIONS[TaskStatus.COMPLETED] == frozenset()


def test_pending_can_only_advance_to_routing() -> None:
    assert TASK_STATUS_TRANSITIONS[TaskStatus.PENDING] == frozenset(
        {TaskStatus.ROUTING}
    )


def test_all_transition_targets_are_valid_task_status_members() -> None:
    """Every value referenced in the table must itself be a real TaskStatus."""
    all_targets = {
        target for targets in TASK_STATUS_TRANSITIONS.values() for target in targets
    }
    for target in all_targets:
        assert isinstance(target, TaskStatus)


def test_failed_can_route_back_to_routing_for_recovery() -> None:
    assert TaskStatus.ROUTING in TASK_STATUS_TRANSITIONS[TaskStatus.FAILED]
