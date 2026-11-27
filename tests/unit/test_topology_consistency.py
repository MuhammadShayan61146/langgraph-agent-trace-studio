"""Guards the dashboard's static topology against drift from graph/build.py.

`build.py` imports langgraph, so it is parsed with `ast` rather than imported.
"""

from __future__ import annotations

import ast
from pathlib import Path

from agentic_research.telemetry.topology import EDGES, END, NODES, START

BUILD = Path(__file__).resolve().parents[2] / "src" / "agentic_research" / "graph" / "build.py"


def _calls(name: str) -> list[ast.Call]:
    tree = ast.parse(BUILD.read_text())
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == name]


def test_every_node_in_build_py_is_in_topology() -> None:
    built = {c.args[0].value for c in _calls("add_node")}
    topo = {n["id"] for n in NODES} - {START, END}
    assert built == topo


def test_plain_edges_match_build_py() -> None:
    built = {(c.args[0].value, c.args[1].value) for c in _calls("add_edge")}
    topo = {(s, t) for s, t, k in EDGES if k == "return"}
    assert built == topo


def test_conditional_edges_match_build_py() -> None:
    built: set[tuple[str, str]] = set()
    for call in _calls("add_conditional_edges"):
        source = call.args[0].value
        for target in call.args[2].values:
            if isinstance(target, ast.Constant):
                built.add((source, target.value))
            else:  # END_SENTINEL name maps to langgraph END
                built.add((source, END))
    conditional_kinds = {"dispatch", "gate", "end", "revise"}
    topo = {(s, t) for s, t, k in EDGES if k in conditional_kinds}
    assert built == topo


def test_entry_point_is_supervisor() -> None:
    entry = [c.args[0].value for c in _calls("set_entry_point")]
    assert entry == ["supervisor"] and (START, "supervisor", "entry") in EDGES
