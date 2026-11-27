"""Human-in-the-loop layer: approval decisions (framework-free) + the real
LangGraph interrupt node.

Design decision
----------------
Same pattern as `agentic_research.graph.__init__` (Module 5): the
framework-free symbols (`ApprovalDecision`, `build_approval_request`,
`apply_human_decision`) are always importable; `human_approval_node`
(which imports `langgraph.types.interrupt`) is imported lazily inside a
`try/except ImportError` so this package stays importable in this sandbox
regardless.
"""

from agentic_research.human.approval import apply_human_decision, build_approval_request
from agentic_research.human.decision import ApprovalDecision

__all__ = [
    "ApprovalDecision",
    "build_approval_request",
    "apply_human_decision",
]

try:
    from agentic_research.human.interrupt_node import human_approval_node  # noqa: F401

    __all__.append("human_approval_node")
except ImportError:
    # langgraph is not installed in this environment -- see module
    # docstring. agentic_research.human.interrupt_node still exists as a
    # file and can be imported directly wherever langgraph is available.
    pass
