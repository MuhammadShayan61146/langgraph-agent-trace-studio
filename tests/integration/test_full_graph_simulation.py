""" integration test: a framework-free simulation of the exact
node/edge sequence `graph/build.py` now wires into a real `StateGraph`.

**What this proves**: that `supervisor_node`, `route_from_supervisor`,
`route_after_supervisor`, every agent's `.run`, and 
`with_retry` -- the real functions `build_graph` imports and registers --
compose into a correct end-to-end control flow, including the human
approval loop.


**One necessary substitution**: `human_approval_node`
(`agentic_research.human.interrupt_node`) itself imports
`langgraph.types.interrupt` and cannot be called in this sandbox. Its
underlying, framework-free logic -- `apply_human_decision`  -- is
called directly here instead, wrapped in `with_retry` exactly as
`build_graph` wraps the real node. This is the one place this simulation
necessarily diverges from `graph/build.py`'s literal code path; everything
else (`supervisor_node`, `route_from_supervisor`, `route_after_supervisor`,
every agent's `.run`) is imported and called unmodified.
"""

from __future__ import annotations

from agentic_research.core.enums import TaskStatus
from agentic_research.core.state import accumulate_list, create_initial_state
from agentic_research.agents.analyst import AnalystAgent
from agentic_research.agents.base import LLMResponse, MockLLMClient
from agentic_research.agents.researcher import ResearcherAgent
from agentic_research.agents.writer import WriterAgent
from agentic_research.config.settings import Settings
from agentic_research.error_recovery import with_retry
from agentic_research.graph.supervisor import (
    END_SENTINEL,
    route_after_supervisor,
    route_from_supervisor,
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


def _build_wrapped_nodes(settings: Settings, report_texts: list, decisions: list):
    """Build every node exactly as `build_graph` does: real agents, each
    node wrapped in `with_retry` using `settings.max_retries`, and the one
    substitution documented in the module docstring for human_approval."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool(backend=InMemorySearchBackend()))

    researcher = ResearcherAgent(
        llm_client=MockLLMClient([LLMResponse(content="Researched.")]),
        tool_registry=registry,
    )
    analyst = AnalystAgent(
        llm_client=MockLLMClient([LLMResponse(content="Analyzed.")]),
        tool_registry=registry,
    )
    writer = WriterAgent(
        llm_client=MockLLMClient([LLMResponse(content=t) for t in report_texts]),
        tool_registry=registry,
    )
    decision_queue = list(decisions)

    def human_approval_stand_in(state):
        decision, feedback = decision_queue.pop(0)
        return apply_human_decision(state, decision, feedback)

    def wrap(fn, name):
        return with_retry(fn, max_retries=settings.max_retries, node_name=name)

    return {
        "supervisor": wrap(supervisor_node, "supervisor"),
        "researcher": wrap(researcher.run, "researcher"),
        "analyst": wrap(analyst.run, "analyst"),
        "writer": wrap(writer.run, "writer"),
        "human_approval": wrap(human_approval_stand_in, "human_approval"),
    }


def _run_graph(state: dict, nodes: dict) -> dict:
    """Drive the exact edge structure `build_graph` wires: entry at
    "supervisor"; supervisor's conditional edges via `route_from_supervisor`;
    human_approval's conditional edges via `route_after_supervisor`;
    researcher/analyst/writer unconditionally edge back to "supervisor"."""
    node = "supervisor"
    for _ in range(30):
        state = _merge_update(state, nodes[node](state))
        if node == "supervisor":
            node = route_from_supervisor(state)
        elif node == "human_approval":
            node = route_after_supervisor(state)
        else:
            node = "supervisor"  # researcher/analyst/writer -> supervisor
        if node == END_SENTINEL:
            return state
    raise AssertionError("simulation did not terminate within 30 steps")


def test_full_graph_simulation_completes_without_human_approval_required() -> None:
    settings = Settings(max_retries=2)
    nodes = _build_wrapped_nodes(settings, ["Final report."], decisions=[])
    state = create_initial_state("query", requires_human_approval=False)

    final_state = _run_graph(state, nodes)

    assert final_state["task_status"] == TaskStatus.COMPLETED
    assert final_state["human_approval_status"] == "approved"
    assert final_state["final_report"] == "Final report."


def test_full_graph_simulation_routes_through_human_approval_node_when_required() -> None:
    settings = Settings(max_retries=2)
    nodes = _build_wrapped_nodes(
        settings,
        ["Draft report."],
        decisions=[(ApprovalDecision.APPROVED, None)],
    )
    state = create_initial_state("query", requires_human_approval=True)

    final_state = _run_graph(state, nodes)

    assert final_state["task_status"] == TaskStatus.COMPLETED
    assert final_state["human_approval_status"] == "approved"


def test_full_graph_simulation_handles_a_revision_cycle_end_to_end() -> None:
    """The full loop this module exists to wire up: gate -> revision
    request -> writer runs again (bypassing supervisor, per build.py's
    documented design decision) -> gate again -> approved -> COMPLETED."""
    settings = Settings(max_retries=2)
    nodes = _build_wrapped_nodes(
        settings,
        ["First draft.", "Revised draft."],
        decisions=[
            (ApprovalDecision.REVISION_REQUESTED, "Needs more detail."),
            (ApprovalDecision.APPROVED, None),
        ],
    )
    state = create_initial_state("query", requires_human_approval=True)

    final_state = _run_graph(state, nodes)

    assert final_state["task_status"] == TaskStatus.COMPLETED
    assert final_state["final_report"] == "Revised draft."
    assert any("Needs more detail." in m.content for m in final_state["messages"])


def test_full_graph_simulation_rejection_ends_task_as_failed() -> None:
    settings = Settings(max_retries=2)
    nodes = _build_wrapped_nodes(
        settings,
        ["Draft."],
        decisions=[(ApprovalDecision.REJECTED, "Inaccurate.")],
    )
    state = create_initial_state("query", requires_human_approval=True)

    final_state = _run_graph(state, nodes)

    assert final_state["task_status"] == TaskStatus.FAILED
    assert any("Inaccurate." in e.message for e in final_state["error_log"])
