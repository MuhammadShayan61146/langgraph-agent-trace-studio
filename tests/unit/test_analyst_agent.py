"""Unit tests for ``agentic_research.agents.analyst.AnalystAgent``."""

from __future__ import annotations

from agentic_research.core.enums import TaskStatus
from agentic_research.core.state import create_initial_state
from agentic_research.agents.analyst import AnalystAgent
from agentic_research.agents.base import LLMResponse, MockLLMClient, ToolCallRequest
from agentic_research.tools.calculator import CalculatorTool
from agentic_research.tools.registry import ToolRegistry


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return registry


def test_analyst_only_has_calculator_available() -> None:
    llm = MockLLMClient([LLMResponse(content="no math needed")])
    agent = AnalystAgent(llm_client=llm, tool_registry=_registry())
    state = create_initial_state("query")
    state["task_status"] = TaskStatus.ANALYZING

    agent.run(state)

    schemas_offered = llm.call_history[0]["tools"]
    assert {s["name"] for s in schemas_offered} == {"calculator"}


def test_analyst_uses_calculator_and_transitions_to_routing() -> None:
    llm = MockLLMClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        tool_name="calculator",
                        arguments={"expression": "100 / 4"},
                        call_id="c1",
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(content="The result is 25."),
        ]
    )
    agent = AnalystAgent(llm_client=llm, tool_registry=_registry())
    state = create_initial_state("query")
    state["task_status"] = TaskStatus.ANALYZING

    update = agent.run(state)

    assert update["task_status"] == TaskStatus.ROUTING
    assert update["research_findings"] == []  # analyst has no findings hook
    assert "25" in update["messages"][-1].content
