"""Real LangGraph demonstration .

This script exercises graph/build.py through the actual LangGraph
StateGraph API rather than the framework-free simulation.

The demonstration intentionally disables human approval for the first
run so that the compiled graph can be verified end-to-end without
requiring an interrupt/resume interaction.

The LLM is a deterministic MockLLMClient.
"""

from agentic_research.agents.base import LLMResponse, MockLLMClient
from agentic_research.config.settings import Settings
from agentic_research.core.state import create_initial_state
from agentic_research.graph.build import build_graph
from agentic_research.tools.calculator import CalculatorTool
from agentic_research.tools.registry import ToolRegistry
from agentic_research.tools.search import InMemorySearchBackend, WebSearchTool


def main() -> None:
    """Build and execute the real LangGraph graph."""

    # Shared deterministic LLM used for the offline demonstration.
    llm_client = MockLLMClient(
        [
            LLMResponse(content="Researched information."),
            LLMResponse(content="Analyzed information."),
            LLMResponse(content="Final demonstration report."),
        ]
    )

    # Shared tool registry, matching the project's integration tests.
    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())
    tool_registry.register(
        WebSearchTool(backend=InMemorySearchBackend())
    )

    # Graph configuration.
    settings = Settings(max_retries=2)

    # Build the ACTUAL LangGraph StateGraph through graph/build.py.
    graph = build_graph(
        llm_client=llm_client,
        tool_registry=tool_registry,
        settings=settings,
    )

    # Start with human approval disabled for this first real execution.
    initial_state = create_initial_state(
        "Demonstrate the multi-agent research workflow",
        requires_human_approval=False,
    )

    print("=" * 70)
    print("TASK 4 — REAL LANGGRAPH DEMONSTRATION")
    print("=" * 70)
    print()
    print("Graph compiled successfully.")
    print("Invoking compiled LangGraph...")
    print()

    final_state = graph.invoke(initial_state)

    print("-" * 70)
    print("EXECUTION COMPLETED")
    print("-" * 70)
    print(f"Task status: {final_state['task_status']}")
    print(f"Human approval: {final_state['human_approval_status']}")
    print(f"Final report: {final_state['final_report']}")
    print()
    print("Messages:")
    for message in final_state["messages"]:
        print(f"  - {message.content}")

    print()
    print("=" * 70)
    print("REAL LANGGRAPH DEMONSTRATION FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()