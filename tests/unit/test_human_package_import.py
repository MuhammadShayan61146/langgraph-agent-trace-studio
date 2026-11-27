"""
"""

from __future__ import annotations

import importlib.util


def test_human_package_imports_without_langgraph_installed() -> None:
    import agentic_research.human as human_pkg

    assert hasattr(human_pkg, "ApprovalDecision")
    assert hasattr(human_pkg, "build_approval_request")
    assert hasattr(human_pkg, "apply_human_decision")


def test_human_approval_node_availability_matches_langgraph_installation() -> None:
    import agentic_research.human as human_pkg

    langgraph_available = importlib.util.find_spec("langgraph") is not None
    assert ("human_approval_node" in human_pkg.__all__) == langgraph_available
    if not langgraph_available:
        print(
            "NOTE: langgraph is not installed in this sandbox; "
            "agentic_research.human.interrupt_node is syntax-checked only, "
            "not executed. See docs/MODULE_07_REPORT.md 'Known Risks'."
        )
