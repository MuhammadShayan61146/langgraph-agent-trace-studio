"""
"""

from __future__ import annotations

import importlib.util


def test_graph_package_imports_without_langgraph_installed() -> None:
    """The framework-free symbols must always be importable, regardless of
    whether langgraph is present -- this is the whole point of
    graph/__init__.py's try/except design."""
    import agentic_research.graph as graph_pkg

    assert hasattr(graph_pkg, "supervisor_node")
    assert hasattr(graph_pkg, "route_after_supervisor")
    assert hasattr(graph_pkg, "decide_next_agent")


def test_build_graph_availability_matches_langgraph_installation() -> None:
    """build_graph should be exported if and only if langgraph is installed."""
    import agentic_research.graph as graph_pkg

    langgraph_available = importlib.util.find_spec("langgraph") is not None
    assert ("build_graph" in graph_pkg.__all__) == langgraph_available
    if not langgraph_available:
        print(
            "NOTE: langgraph is not installed in this sandbox; "
            "agentic_research.graph.build is syntax-checked only, not executed. "
            "See docs/MODULE_05_REPORT.md 'Known Risks'."
        )
