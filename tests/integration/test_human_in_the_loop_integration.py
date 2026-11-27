"""Integration test: a full human-in-the-loop cycle, including a revision
round trip, driven entirely through *unmodified*  functions
(`supervisor_node`, `route_after_supervisor`) plus this new
`apply_human_decision`.


"""

from __future__ import annotations

from agentic_research.core.enums import TaskStatus
from agentic_research.core.state import accumulate_list, create_initial_state
from agentic_research.agents.analyst import AnalystAgent
from agentic_research.agents.base import LLMResponse, MockLLMClient
from agentic_research.agents.researcher import ResearcherAgent
from agentic_research.agents.writer import WriterAgent
from agentic_research.graph.supervisor import (
    END_SENTINEL,
    route_after_supervisor,
    supervisor_node,
)
from agentic_research.human.approval import apply_human_decision
from agentic_research.human.decision import ApprovalDecision
from agentic_research.tools.calculator import CalculatorTool
from agentic_research.tools.registry import ToolRegistry
from agentic_research.tools.search import InMemorySearchBackend, WebSearchTool

_LIST_FIELDS = ("messages", "research_findings", "routing_history", "error_log")


def _merge_update(state: dict, update: dict) -> dict:
    merged = dict(state)
    for key, value in update.items():
        if key in _LIST_FIELDS:
            merged[key] = accumulate_list(merged.get(key, []), value)
        else:
            merged[key] = value
    return merged


def _run_until_approval_gate(state: dict, agents_by_node: dict) -> dict:
    """Drive the graph (via unmodified Module 5 functions) until it parks
    at AWAITING_HUMAN_APPROVAL or ends."""
    node = "supervisor"
    for _ in range(20):
        if node == "supervisor":
            state = _merge_update(state, supervisor_node(state))
            if state["task_status"] == TaskStatus.AWAITING_HUMAN_APPROVAL:
                return state
            node = route_after_supervisor(state)
            if node == END_SENTINEL:
                return state
        else:
            state = _merge_update(state, agents_by_node[node].run(state))
            node = "supervisor"
    raise AssertionError("did not reach approval gate or end within 20 steps")


def _build_agents(report_texts: list[str]) -> dict:
    """Build agents where the Writer returns each text in `report_texts`
    on successive calls (simulating a revised draft each round)."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool(backend=InMemorySearchBackend()))
    return {
        "researcher": ResearcherAgent(
            llm_client=MockLLMClient([LLMResponse(content="Researched.")]),
            tool_registry=registry,
        ),
        "analyst": AnalystAgent(
            llm_client=MockLLMClient([LLMResponse(content="Analyzed.")]),
            tool_registry=registry,
        ),
        "writer": WriterAgent(
            llm_client=MockLLMClient([LLMResponse(content=t) for t in report_texts]),
            tool_registry=registry,
        ),
    }


def test_full_cycle_approve_immediately() -> None:
    agents = _build_agents(["First draft."])
    state = create_initial_state("query", requires_human_approval=True)

    state = _run_until_approval_gate(state, agents)
    assert state["task_status"] == TaskStatus.AWAITING_HUMAN_APPROVAL
    assert state["final_report"] == "First draft."

    state = _merge_update(state, apply_human_decision(state, ApprovalDecision.APPROVED))
    assert state["task_status"] == TaskStatus.COMPLETED
    assert state["human_approval_status"] == "approved"


def test_full_cycle_reject_ends_task_as_failed() -> None:
    agents = _build_agents(["Only draft."])
    state = create_initial_state("query", requires_human_approval=True)
    state = _run_until_approval_gate(state, agents)

    state = _merge_update(
        state,
        apply_human_decision(state, ApprovalDecision.REJECTED, feedback="Not accurate."),
    )
    assert state["task_status"] == TaskStatus.FAILED
    assert state["human_approval_status"] == "rejected"
    assert any("Not accurate." in e.message for e in state["error_log"])


def test_full_cycle_revision_requested_then_approved() -> None:
    """The real end-to-end proof: reviewer asks for a revision, the Writer
    (unmodified Module 4 code) produces a second draft incorporating the
    feedback message, and the reviewer then approves it."""
    agents = _build_agents(["First draft.", "Revised draft with more detail."])
    state = create_initial_state("query", requires_human_approval=True)

    # --- First pass: reach the gate, request a revision ---
    state = _run_until_approval_gate(state, agents)
    assert state["final_report"] == "First draft."

    state = _merge_update(
        state,
        apply_human_decision(
            state, ApprovalDecision.REVISION_REQUESTED, feedback="Add more detail."
        ),
    )
    assert state["task_status"] == TaskStatus.WRITING
    assert state["human_approval_status"] is None
    # The revision request must be visible to the Writer's next LLM call.
    revision_messages = [m for m in state["messages"] if "Add more detail." in m.content]
    assert len(revision_messages) == 1

    # --- Second pass: Writer runs again (its own LLM call sees the revision
    # request in conversation history via _messages_from_state, Module 4,
    # unmodified), produces the second scripted draft, gate is reached again ---
    state = _merge_update(state, agents["writer"].run(state))
    assert state["task_status"] == TaskStatus.AWAITING_HUMAN_APPROVAL
    assert state["final_report"] == "Revised draft with more detail."

    # --- Reviewer approves the revised draft ---
    state = _merge_update(state, apply_human_decision(state, ApprovalDecision.APPROVED))
    assert state["task_status"] == TaskStatus.COMPLETED
