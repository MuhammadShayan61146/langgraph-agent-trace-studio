"""Unit tests for ``agentic_research.human.decision.ApprovalDecision``."""

from __future__ import annotations

import pytest

from agentic_research.human.decision import ApprovalDecision


def test_approval_decision_members_are_string_comparable() -> None:
    assert ApprovalDecision.APPROVED == "approved"
    assert ApprovalDecision.REJECTED.value == "rejected"
    assert ApprovalDecision.REVISION_REQUESTED.value == "revision_requested"


def test_approval_decision_constructs_from_valid_string() -> None:
    assert ApprovalDecision("approved") == ApprovalDecision.APPROVED


def test_approval_decision_rejects_unrecognized_string() -> None:
    with pytest.raises(ValueError):
        ApprovalDecision("maybe_later")
