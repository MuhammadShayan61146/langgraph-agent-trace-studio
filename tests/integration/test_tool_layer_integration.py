"""Integration test for the tool layer: registry + multiple real tools together.

Unlike the unit tests (one component in isolation), this exercises the path
an agent node will actually use in : build a registry, register
several tools, generate the combined LLM tool schema, and execute tools by
name through the registry rather than by holding direct references to tool
instances.
"""

from __future__ import annotations

import pytest

from agentic_research.core.exceptions import ToolNotFoundError
from agentic_research.tools import (
    CalculatorTool,
    InMemorySearchBackend,
    SearchResultItem,
    ToolRegistry,
    WebSearchTool,
)


def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(
        WebSearchTool(
            backend=InMemorySearchBackend(
                {
                    "langgraph": [
                        SearchResultItem(
                            title="LangGraph Docs",
                            snippet="A library for stateful, multi-agent apps.",
                            url="https://example.com/langgraph",
                        )
                    ]
                }
            )
        )
    )
    return registry


def test_registry_holds_both_tools() -> None:
    registry = _build_registry()
    assert len(registry) == 2
    assert registry.list_tool_names() == ["calculator", "web_search"]


def test_registry_executes_calculator_by_name() -> None:
    registry = _build_registry()
    result = registry.execute("calculator", expression="6 * 7")
    assert result.success is True
    assert result.output == 42


def test_registry_executes_search_by_name() -> None:
    registry = _build_registry()
    result = registry.execute("web_search", query="langgraph")
    assert result.success is True
    assert result.output[0]["title"] == "LangGraph Docs"


def test_registry_unknown_tool_name_raises_with_helpful_context() -> None:
    registry = _build_registry()
    with pytest.raises(ToolNotFoundError) as exc_info:
        registry.execute("not_a_real_tool")
    assert exc_info.value.details["available_tools"] == ["calculator", "web_search"]


def test_registry_generates_combined_schema_for_llm_tool_calling() -> None:
    registry = _build_registry()
    schemas = registry.to_anthropic_tool_schemas()
    names = {s["name"] for s in schemas}
    assert names == {"calculator", "web_search"}
    for schema in schemas:
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"


def test_a_failing_tool_call_does_not_affect_other_tools_in_the_registry() -> None:
    """One tool's internal failure must not corrupt registry state or block
    subsequent, unrelated tool calls -- this is the resilience property the
    whole ToolResult-never-raises design (base.py) exists to guarantee."""
    registry = _build_registry()

    bad_result = registry.execute("calculator", expression="__import__('os')")
    assert bad_result.success is False

    good_result = registry.execute("calculator", expression="1 + 1")
    assert good_result.success is True
    assert good_result.output == 2
