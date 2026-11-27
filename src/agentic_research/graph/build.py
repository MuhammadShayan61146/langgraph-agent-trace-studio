"""LangGraph `StateGraph` assembly: full integration of Modules 4, 5, 7, 8.

Purpose
-------
This is the one file in the project whose entire job is to import and call
`langgraph`. Every other module (`core`, `tools`, `agents`, `graph.supervisor`,
`human.approval`, `error_recovery`) was deliberately kept framework-free so
it could be verified in this sandbox; this file is where that
framework-free logic actually gets attached to LangGraph's `StateGraph`.

**Module 9 update**: this file now wires in the two pieces Modules 5 and 7
left deliberately deferred — the human-approval interrupt node and the
Module 8 retry wrapper — completing the graph rather than leaving it a
three-agent partial assembly. No node/routing *logic* was duplicated to do
this: every node function and the one new routing function
(`route_from_supervisor`, added to `graph/supervisor.py` this module) are
reused exactly as written by their owning modules.

**Status: Design-only.** `langgraph` is not installed in this sandbox (no
network egress — see Module 1's environment disclosure). This file is
syntax-checked (`python -m py_compile`) and reviewed against LangGraph's
publicly documented API shape (including `interrupt()`/`Command(resume=...)`
for the human-approval node), but has never been imported or executed.
Everything it imports (`GraphState`, `supervisor_node`,
`route_from_supervisor`, `human_approval_node`, `with_retry`, every agent's
`.run`) is independently Verified; this file's *use* of them through the
real `StateGraph` API is not — that gap is proxied, not closed, by the
framework-free simulation in
`tests/integration/test_full_graph_simulation.py` (see that file's
docstring for exactly what it does and does not prove).

Responsibilities
-----------------
* Construct a `StateGraph(GraphState)`.
* Register the supervisor node, the three Module 4 agent nodes, and the
  Module 7 human-approval node — every node wrapped in Module 8's
  `with_retry`.
* Wire the supervisor's conditional edges via `route_from_supervisor`
  (Module 9 addition to `graph/supervisor.py`, reusing
  `route_after_supervisor` unmodified), the human-approval node's
  conditional edges via `route_after_supervisor` directly, and the
  unconditional "return to supervisor" edges from each working agent.
* Compile and return the runnable graph.

Design decisions
-----------------
* **Agents are constructed inside `build_graph`, not at import time**,
  matching the DI discipline used everywhere in this project (Settings,
  SearchBackend, LLMClient) — unchanged from Module 5.
* **`Settings` (Module 1) is now a required parameter**, sourcing
  `max_retries` for every `with_retry`-wrapped node. This is the first use
  of `Settings` inside the graph layer.
* **The human-approval node's revision loop bypasses `supervisor_node`
  entirely.** After `human_approval_node` sets `task_status=WRITING` for a
  revision request, execution routes directly back to `"writer"` (via
  `route_after_supervisor`, reused unmodified) rather than back through
  `"supervisor"`. Routing through the supervisor first would incorrectly
  invoke `decide_next_agent`, which — being presence-based (Module 5) —
  would see the Writer already spoke and misroute to `AgentRole.HUMAN`
  instead of back to the Writer. This exact bypass was proven correct in
  Module 7's `test_full_cycle_revision_requested_then_approved`, which
  drives the identical sequence.
* **The `"__end__"` -> `END` translation happens only here**, in the one
  file allowed to know about `langgraph.graph.END`.

Limitations
------------
* Not executed. Structural correctness (node names match edge targets,
  every referenced node is registered) was checked by hand and by the
  accompanying framework-free simulation in
  `tests/integration/test_full_graph_simulation.py`.
* The interrupt/resume mechanics inside `human_approval_node`
  (`agentic_research.human.interrupt_node`) remain unverified against a
  real `langgraph.types.interrupt()` call — inherited, unchanged risk from
  Module 7.

Future improvements
--------------------
* Add a real `langgraph.checkpoint` checkpointer, adapting Module 6's
  `CheckpointStore` concept, once cross-invocation persistence through the
  compiled graph itself (not just this project's own `FileCheckpointStore`)
  is needed.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from agentic_research.agents.analyst import AnalystAgent
from agentic_research.agents.base import LLMClient
from agentic_research.agents.researcher import ResearcherAgent
from agentic_research.agents.writer import WriterAgent
from agentic_research.config.settings import Settings
from agentic_research.core.state import GraphState
from agentic_research.error_recovery import with_retry
from agentic_research.graph.supervisor import (
    END_SENTINEL,
    route_after_supervisor,
    route_from_supervisor,
    supervisor_node,
)
from agentic_research.human.interrupt_node import human_approval_node
from agentic_research.tools.registry import ToolRegistry


def build_graph(
    llm_client: LLMClient, tool_registry: ToolRegistry, settings: Settings
) -> Any:
    """Construct and compile the full multi-agent research graph.

    Args:
        llm_client: Injected into every agent (`ResearcherAgent`,
            `AnalystAgent`, `WriterAgent`). Production callers pass a real
            `LLMClient` implementation (see
            `agentic_research.agents.anthropic_client`, not yet
            implemented); tests pass `MockLLMClient`.
        tool_registry: The shared `ToolRegistry` every agent draws its
            `allowed_tools` subset from.
        settings: Provides `max_retries` for every node's `with_retry`
            wrapper (Module 8).

    Returns:
        A compiled LangGraph graph (LangGraph's `CompiledStateGraph`),
        ready to invoke with an initial `GraphState` (see
        `agentic_research.core.state.create_initial_state`). Typed as
        `Any` rather than importing the exact compiled-graph class name —
        unchanged rationale from Module 5.

    Note:
        Not executed in this sandbox — see module docstring. The node/edge
        wiring below is structurally verified by
        `tests/integration/test_full_graph_simulation.py`, which drives
        the same nodes and routing functions directly.
    """
    researcher = ResearcherAgent(llm_client=llm_client, tool_registry=tool_registry)
    analyst = AnalystAgent(llm_client=llm_client, tool_registry=tool_registry)
    writer = WriterAgent(llm_client=llm_client, tool_registry=tool_registry)

    def _wrap(node_fn, node_name: str):
        return with_retry(node_fn, max_retries=settings.max_retries, node_name=node_name)

    graph = StateGraph(GraphState)
    graph.add_node("supervisor", _wrap(supervisor_node, "supervisor"))
    graph.add_node("researcher", _wrap(researcher.run, "researcher"))
    graph.add_node("analyst", _wrap(analyst.run, "analyst"))
    graph.add_node("writer", _wrap(writer.run, "writer"))
    graph.add_node("human_approval", _wrap(human_approval_node, "human_approval"))

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "human_approval": "human_approval",
            END_SENTINEL: END,
        },
    )
    graph.add_conditional_edges(
        "human_approval",
        route_after_supervisor,
        {
            "writer": "writer",
            END_SENTINEL: END,
        },
    )

    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("writer", "supervisor")

    return graph.compile()
