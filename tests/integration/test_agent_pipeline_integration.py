"""Integration test: Researcher -> Analyst -> Writer through real GraphState merging.

This simulates exactly what compiled graph will do when it runs
these three nodes in sequence: take each agent's partial-update dict and
merge it into the running state using the real `accumulate_list` reducer, not a hand-rolled test-only merge. No `langgraph` import is
needed to prove this merging behaves correctly end-to-end.
"""

from __future__ import annotations

from agentic_research.core.enums import AgentRole, TaskStatus
from agentic_research.core.state import accumulate_list, create_initial_state
from agentic_research.agents.analyst import AnalystAgent
from agentic_research.agents.base import LLMResponse, MockLLMClient, ToolCallRequest
from agentic_research.agents.researcher import ResearcherAgent
from agentic_research.agents.writer import WriterAgent
from agentic_research.tools.calculator import CalculatorTool
from agentic_research.tools.registry import ToolRegistry
from agentic_research.tools.search import InMemorySearchBackend, SearchResultItem, WebSearchTool


def _merge_update(state: dict, update: dict) -> dict:
    """Apply a node's partial update to state using GraphState's real reducers.

    Mirrors exactly what LangGraph's StateGraph does at runtime for
    Annotated-list fields (concatenate) vs scalar fields (overwrite) --
    see Module 2's `state.py` for the reducer wiring this replicates.
    """
    merged = dict(state)
    for key, value in update.items():
        if key in ("messages", "research_findings", "routing_history", "error_log"):
            merged[key] = accumulate_list(merged.get(key, []), value)
        else:
            merged[key] = value
    return merged


def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(
        WebSearchTool(
            backend=InMemorySearchBackend(
                {
                    "electric vehicle sales": [
                        SearchResultItem(
                            title="EV Sales Report 2026",
                            snippet="Global EV sales reached 18 million units.",
                            url="https://example.com/ev-report",
                        )
                    ]
                }
            )
        )
    )
    return registry


def test_full_researcher_analyst_writer_pipeline_via_real_state_merging() -> None:
    registry = _build_registry()
    state = create_initial_state("How many EVs were sold and what's double that?")
    state["task_status"] = TaskStatus.RESEARCHING

    # --- Researcher turn ---
    researcher_llm = MockLLMClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        tool_name="web_search",
                        arguments={"query": "electric vehicle sales"},
                        call_id="c1",
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(content="Found EV sales data: 18 million units in 2026."),
        ]
    )
    researcher = ResearcherAgent(llm_client=researcher_llm, tool_registry=registry)
    state = _merge_update(state, researcher.run(state))

    assert state["task_status"] == TaskStatus.ROUTING
    assert len(state["research_findings"]) == 1
    assert "18 million" in state["research_findings"][0].content

    # --- Analyst turn (supervisor would set ANALYZING; simulated here) ---
    state["task_status"] = TaskStatus.ANALYZING
    analyst_llm = MockLLMClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        tool_name="calculator",
                        arguments={"expression": "18 * 2"},
                        call_id="c2",
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(content="Double the sales figure is 36 million units."),
        ]
    )
    analyst = AnalystAgent(llm_client=analyst_llm, tool_registry=registry)
    state = _merge_update(state, analyst.run(state))

    assert state["task_status"] == TaskStatus.ROUTING
    # Researcher's finding must still be present -- accumulate_list, not overwrite.
    assert len(state["research_findings"]) == 1

    # --- Writer turn ---
    state["task_status"] = TaskStatus.WRITING
    writer_llm = MockLLMClient(
        [
            LLMResponse(
                content=(
                    "18 million EVs were sold in 2026; double that figure is "
                    "36 million."
                )
            )
        ]
    )
    writer = WriterAgent(llm_client=writer_llm, tool_registry=registry)
    state = _merge_update(state, writer.run(state))

    assert state["task_status"] == TaskStatus.AWAITING_HUMAN_APPROVAL
    assert state["final_report"] is not None
    assert "36 million" in state["final_report"]

    # --- Full audit trail accumulated across all three agents ---
    assert state["current_agent"] == AgentRole.WRITER
    # 2 TOOL/ASSISTANT messages from researcher + 2 from analyst + 1 from writer
    assert len(state["messages"]) == 5
    assert len(state["error_log"]) == 0
