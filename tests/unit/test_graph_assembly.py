"""Unit tests for ``agentic_research.graph.supervisor.route_from_supervisor``
 — the real routing function `graph/build.py` uses for
the supervisor's conditional edges.
"""

from __future__ import annotations

import pytest

from agentic_research.core.enums import TaskStatus
from agentic_research.core.state import create_initial_state
from agentic_research.graph.supervisor import END_SENTINEL, route_from_supervisor


@pytest.mark.parametrize(
    "status,expected_node",
    [
        (TaskStatus.RESEARCHING, "researcher"),
        (TaskStatus.ANALYZING, "analyst"),
        (TaskStatus.WRITING, "writer"),
        (TaskStatus.AWAITING_HUMAN_APPROVAL, "human_approval"),
        (TaskStatus.COMPLETED, END_SENTINEL),
        (TaskStatus.FAILED, END_SENTINEL),
        (TaskStatus.PENDING, END_SENTINEL),
    ],
)
def test_route_from_supervisor_maps_status_to_expected_node(
    status: TaskStatus, expected_node: str
) -> None:
    state = create_initial_state("query")
    state["task_status"] = status
    assert route_from_supervisor(state) == expected_node


def test_route_from_supervisor_differs_from_route_after_supervisor_only_for_approval_gate() -> None:
    """The one behavioral difference this Module 9 addition introduces:
    AWAITING_HUMAN_APPROVAL routes to a real node instead of ending."""
    from agentic_research.graph.supervisor import route_after_supervisor

    state = create_initial_state("query")
    state["task_status"] = TaskStatus.AWAITING_HUMAN_APPROVAL

    assert route_after_supervisor(state) == END_SENTINEL  # Module 5, unchanged
    assert route_from_supervisor(state) == "human_approval"  # Module 9 addition
