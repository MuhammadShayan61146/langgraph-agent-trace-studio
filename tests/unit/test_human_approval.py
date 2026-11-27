"""Unit tests for ``agentic_research.human.approval``."""

from __future__ import annotations

import pytest

from agentic_research.core.contracts import AgentMessage
from agentic_research.core.enums import AgentRole, MessageRole, TaskStatus
from agentic_research.core.state import create_initial_state
from agentic_research.human.approval import apply_human_decision, build_approval_request
from agentic_research.human.decision import ApprovalDecision


def _state_awaiting_approval(final_report: str = "Draft report text.") -> dict:
    state = create_initial_state("query", requires_human_approval=True)
    state["task_status"] = TaskStatus.AWAITING_HUMAN_APPROVAL
    state["final_report"] = final_report
    return state


# --------------------------------------------------------------------------- #
# build_approval_request
# --------------------------------------------------------------------------- #
def test_build_approval_request_includes_expected_fields() -> None:
    state = _state_awaiting_approval("The final draft.")
    request = build_approval_request(state)

    assert request["original_query"] == "query"
    assert request["draft_report"] == "The final draft."
    assert request["supporting_findings_count"] == 0
    assert request["revision_cycle"] == 0


def test_build_approval_request_counts_prior_writer_turns() -> None:
    state = _state_awaiting_approval()
    state["messages"] = [
        AgentMessage(role=MessageRole.ASSISTANT, agent=AgentRole.WRITER, content="v1"),
        AgentMessage(role=MessageRole.USER, content="please revise"),
        AgentMessage(role=MessageRole.ASSISTANT, agent=AgentRole.WRITER, content="v2"),
    ]
    request = build_approval_request(state)
    assert request["revision_cycle"] == 2


# --------------------------------------------------------------------------- #
# apply_human_decision - approved
# --------------------------------------------------------------------------- #
def test_apply_human_decision_approved_completes_task() -> None:
    state = _state_awaiting_approval()
    update = apply_human_decision(state, ApprovalDecision.APPROVED)

    assert update["task_status"] == TaskStatus.COMPLETED
    assert update["human_approval_status"] == "approved"


# --------------------------------------------------------------------------- #
# apply_human_decision - rejected
# --------------------------------------------------------------------------- #
def test_apply_human_decision_rejected_fails_task_with_error_record() -> None:
    state = _state_awaiting_approval()
    update = apply_human_decision(state, ApprovalDecision.REJECTED, feedback="Too vague.")

    assert update["task_status"] == TaskStatus.FAILED
    assert update["human_approval_status"] == "rejected"
    assert len(update["error_log"]) == 1
    assert "Too vague." in update["error_log"][0].message
    assert update["error_log"][0].recoverable is False


def test_apply_human_decision_rejected_without_feedback_uses_default_message() -> None:
    state = _state_awaiting_approval()
    update = apply_human_decision(state, ApprovalDecision.REJECTED)
    assert "rejected" in update["error_log"][0].message.lower()


# --------------------------------------------------------------------------- #
# apply_human_decision - revision requested
# --------------------------------------------------------------------------- #
def test_apply_human_decision_revision_requested_returns_to_writing() -> None:
    state = _state_awaiting_approval()
    update = apply_human_decision(
        state, ApprovalDecision.REVISION_REQUESTED, feedback="Add more detail."
    )

    assert update["task_status"] == TaskStatus.WRITING
    assert update["human_approval_status"] is None
    assert len(update["messages"]) == 1
    assert update["messages"][0].role == MessageRole.USER
    assert "Add more detail." in update["messages"][0].content


def test_apply_human_decision_revision_requested_without_feedback_has_placeholder() -> None:
    state = _state_awaiting_approval()
    update = apply_human_decision(state, ApprovalDecision.REVISION_REQUESTED)
    assert "no feedback given" in update["messages"][0].content


# --------------------------------------------------------------------------- #
# apply_human_decision - misuse guard
# --------------------------------------------------------------------------- #
def test_apply_human_decision_raises_if_not_awaiting_approval() -> None:
    state = create_initial_state("query")  # task_status defaults to PENDING
    with pytest.raises(ValueError):
        apply_human_decision(state, ApprovalDecision.APPROVED)
