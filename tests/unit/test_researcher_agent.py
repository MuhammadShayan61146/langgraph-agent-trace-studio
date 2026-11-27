"""Unit tests for ``agentic_research.agents.researcher.ResearcherAgent``."""

from __future__ import annotations

from agentic_research.core.enums import TaskStatus
from agentic_research.core.state import create_initial_state
from agentic_research.agents.base import LLMResponse, MockLLMClient, ToolCallRequest
from agentic_research.agents.researcher import ResearcherAgent
from agentic_research.tools.registry import ToolRegistry
from agentic_research.tools.search import InMemorySearchBackend, SearchResultItem, WebSearchTool

_FIXTURES = {
    "langgraph": [
        SearchResultItem(
            title="LangGraph Docs",
            snippet="A library for stateful multi-agent apps.",
            url="https://example.com/langgraph",
        ),
        SearchResultItem(
            title="LangGraph Tutorial",
            snippet="Building your first graph.",
            url="https://example.com/tutorial",
        ),
    ]
}


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WebSearchTool(backend=InMemorySearchBackend(_FIXTURES)))
    return registry


def test_researcher_only_has_web_search_available() -> None:
    llm = MockLLMClient([LLMResponse(content="no search needed")])
    agent = ResearcherAgent(llm_client=llm, tool_registry=_registry())
    state = create_initial_state("query")
    state["task_status"] = TaskStatus.RESEARCHING

    agent.run(state)

    schemas_offered = llm.call_history[0]["tools"]
    assert {s["name"] for s in schemas_offered} == {"web_search"}


def test_researcher_converts_search_results_into_findings() -> None:
    llm = MockLLMClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        tool_name="web_search",
                        arguments={"query": "langgraph"},
                        call_id="c1",
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(content="Found two relevant sources on LangGraph."),
        ]
    )
    agent = ResearcherAgent(llm_client=llm, tool_registry=_registry())
    state = create_initial_state("Tell me about LangGraph")
    state["task_status"] = TaskStatus.RESEARCHING

    update = agent.run(state)

    findings = update["research_findings"]
    assert len(findings) == 2
    assert findings[0].source == "https://example.com/langgraph"
    assert "LangGraph Docs" in findings[0].content
    assert findings[0].confidence == 0.7
    assert update["task_status"] == TaskStatus.ROUTING


def test_researcher_produces_no_findings_when_search_returns_nothing() -> None:
    llm = MockLLMClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        tool_name="web_search",
                        arguments={"query": "unregistered topic"},
                        call_id="c1",
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(content="Nothing found."),
        ]
    )
    agent = ResearcherAgent(llm_client=llm, tool_registry=_registry())
    state = create_initial_state("query")
    state["task_status"] = TaskStatus.RESEARCHING

    update = agent.run(state)
    assert update["research_findings"] == []
