"""Framework-free simulation of the full graph, driving the exact same
functions `graph/build.py`'s real `StateGraph` would call, in the same
order LangGraph would call them.


"""

from __future__ import annotations

from agentic_research.core.enums import AgentRole, TaskStatus
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
from agentic_research.tools.registry import ToolRegistry

_LIST_FIELDS = ("messages", "research_findings", "routing_history", "error_log")


def _registry_with_all_tools() -> ToolRegistry:
    """Build a registry containing every tool an agent's `allowed_tools`
    references. Required even when the scripted MockLLMClient never
    actually requests a tool call: `BaseAgent._tool_schemas()` looks up
    every allowed tool unconditionally to build the schemas offered to the
    LLM (Module 4 design) -- an agent whose allowed tool isn't registered
    fails at schema-generation time, before the LLM is ever called.
    """
    from agentic_research.tools.calculator import CalculatorTool
    from agentic_research.tools.search import InMemorySearchBackend, WebSearchTool

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool(backend=InMemorySearchBackend()))
    return registry


def _merge_update(state: dict, update: dict) -> dict:
    """Merge a node's partial update using GraphState's real reducers
    (accumulate for list fields, overwrite for everything else) -- same
    helper as tests/integration/test_agent_pipeline_integration.py."""
    merged = dict(state)
    for key, value in update.items():
        if key in _LIST_FIELDS:
            merged[key] = accumulate_list(merged.get(key, []), value)
        else:
            merged[key] = value
    return merged


def _run_graph_manually(state: dict, agents_by_node: dict) -> dict:
    """Drive supervisor -> agent -> supervisor -> ... -> END, exactly as
    `graph/build.py`'s conditional + unconditional edges specify, using
    plain Python control flow instead of a compiled StateGraph."""
    node = "supervisor"
    max_steps = 20  # simple infinite-loop guard for a test, not production logic
    for _ in range(max_steps):
        if node == "supervisor":
            state = _merge_update(state, supervisor_node(state))
            node = route_after_supervisor(state)
            if node == END_SENTINEL:
                return state
        else:
            agent = agents_by_node[node]
            state = _merge_update(state, agent.run(state))
            node = "supervisor"  # every agent node edges back to supervisor
    raise AssertionError("Graph simulation did not terminate within max_steps")


def test_graph_simulation_reaches_completed_without_human_approval() -> None:
    registry = _registry_with_all_tools()
    agents_by_node = {
        "researcher": ResearcherAgent(
            llm_client=MockLLMClient([LLMResponse(content="Researched.")]),
            tool_registry=registry,
        ),
        "analyst": AnalystAgent(
            llm_client=MockLLMClient([LLMResponse(content="Analyzed.")]),
            tool_registry=registry,
        ),
        "writer": WriterAgent(
            llm_client=MockLLMClient([LLMResponse(content="Final report.")]),
            tool_registry=registry,
        ),
    }
    state = create_initial_state("query", requires_human_approval=False)

    final_state = _run_graph_manually(state, agents_by_node)

    assert final_state["task_status"] == TaskStatus.COMPLETED
    assert final_state["human_approval_status"] == "approved"
    assert final_state["final_report"] == "Final report."
    # Exactly 3 routing decisions: to researcher, to analyst, to writer.
    assert [d.to_agent for d in final_state["routing_history"]] == [
        AgentRole.RESEARCHER,
        AgentRole.ANALYST,
        AgentRole.WRITER,
    ]


def test_graph_simulation_parks_at_approval_gate_when_required() -> None:
    registry = _registry_with_all_tools()
    agents_by_node = {
        "researcher": ResearcherAgent(
            llm_client=MockLLMClient([LLMResponse(content="Researched.")]),
            tool_registry=registry,
        ),
        "analyst": AnalystAgent(
            llm_client=MockLLMClient([LLMResponse(content="Analyzed.")]),
            tool_registry=registry,
        ),
        "writer": WriterAgent(
            llm_client=MockLLMClient([LLMResponse(content="Final report.")]),
            tool_registry=registry,
        ),
    }
    state = create_initial_state("query", requires_human_approval=True)

    final_state = _run_graph_manually(state, agents_by_node)

    # Graph ends (per route_after_supervisor's current design) without
    # actually completing -- see supervisor.py "Design decisions" and the
    # Module 5 report's Known Risks for why this is an honest limitation,
    # not a bug, pending Module 7.
    assert final_state["task_status"] == TaskStatus.AWAITING_HUMAN_APPROVAL
    assert final_state["human_approval_status"] is None
    assert final_state["final_report"] == "Final report."
