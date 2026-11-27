"""Static description of the graph, served to the dashboard.

This mirrors the wiring in `graph/build.py`. Because `build.py` imports
`langgraph` (not installed everywhere), a unit test parses its source with
`ast` and asserts every node and edge below still exists there, so this
description cannot silently drift.
"""

from __future__ import annotations

START = "__start__"
END = "__end__"

NODES: list[dict[str, str]] = [
    {"id": START, "label": "Start", "kind": "terminal", "blurb": "Initial state enters the graph."},
    {"id": "supervisor", "label": "Supervisor", "kind": "supervisor",
     "blurb": "Decides who acts next from who has already contributed."},
    {"id": "researcher", "label": "Researcher", "kind": "agent",
     "blurb": "Gathers findings with the web_search tool."},
    {"id": "analyst", "label": "Analyst", "kind": "agent",
     "blurb": "Reasons over findings, using the calculator tool."},
    {"id": "writer", "label": "Writer", "kind": "agent",
     "blurb": "Synthesises the final report."},
    {"id": "human_approval", "label": "Human gate", "kind": "gate",
     "blurb": "Interrupts for approve / reject / revise."},
    {"id": END, "label": "End", "kind": "terminal", "blurb": "Run finished."},
]

# (source, target, kind)
EDGES: list[tuple[str, str, str]] = [
    (START, "supervisor", "entry"),
    ("supervisor", "researcher", "dispatch"),
    ("supervisor", "analyst", "dispatch"),
    ("supervisor", "writer", "dispatch"),
    ("supervisor", "human_approval", "gate"),
    ("supervisor", END, "end"),
    ("researcher", "supervisor", "return"),
    ("analyst", "supervisor", "return"),
    ("writer", "supervisor", "return"),
    ("human_approval", "writer", "revise"),
    ("human_approval", END, "end"),
]


def topology() -> dict[str, object]:
    return {
        "nodes": NODES,
        "edges": [{"source": s, "target": t, "kind": k} for s, t, k in EDGES],
    }
