"""Graph layer: conditional routing logic (framework-free) + LangGraph assembly.

Design decision
----------------
`agentic_research.graph.supervisor` is framework-free and always
importable. `agentic_research.graph.build` imports `langgraph`, which is
**not installed in this sandbox**. If this `__init__.py` eagerly imported
`build_graph`, simply running ``import agentic_research.graph`` would raise
`ModuleNotFoundError` here — breaking every test and audit script that
touches this package for a reason unrelated to what they're testing.

Instead, the framework-free symbols are exported unconditionally, and
`build_graph` is imported lazily inside a `try/except ImportError`, only
available in `__all__` when `langgraph` is actually present. This keeps
`agentic_research.graph` importable everywhere while still offering
`build_graph` as first-class API in an environment that has `langgraph`
installed (i.e. anywhere outside this development sandbox).
"""

from agentic_research.graph.supervisor import (
    END_SENTINEL,
    decide_next_agent,
    route_after_supervisor,
    route_from_supervisor,
    supervisor_node,
)

__all__ = [
    "END_SENTINEL",
    "decide_next_agent",
    "supervisor_node",
    "route_after_supervisor",
    "route_from_supervisor",
]

try:
    from agentic_research.graph.build import build_graph  # noqa: F401

    __all__.append("build_graph")
except ImportError:
    # langgraph is not installed in this environment -- see module
    # docstring. `agentic_research.graph.build` still exists as a file and
    # can be imported directly wherever langgraph *is* available.
    pass
