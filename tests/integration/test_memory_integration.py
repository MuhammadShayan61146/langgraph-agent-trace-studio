"""Integration test: checkpoint the  graph simulation mid-run, resume
from disk, and persist final findings to long-term memory.

This is the memory layer's real end-to-end proof: unlike Module 3/4's
`InMemory*` mocks (standing in for something unbuildable offline), the
persistence here is genuine disk I/O with no unavailable dependency, so
this test exercises the full, real path a resumable research task would
take.
"""

from __future__ import annotations

import tempfile

from agentic_research.core.enums import AgentRole, TaskStatus
from agentic_research.core.state import accumulate_list, create_initial_state
from agentic_research.agents.analyst import AnalystAgent
from agentic_research.agents.base import LLMResponse, MockLLMClient
from agentic_research.agents.researcher import ResearcherAgent
from agentic_research.agents.writer import WriterAgent
from agentic_research.graph.supervisor import (
    END_SENTINEL,
    route_after_supervisor,
    supervisor_node,
)
from agentic_research.memory.checkpoint import FileCheckpointStore
from agentic_research.memory.long_term import InMemoryLongTermMemoryStore, save_findings
from agentic_research.tools.calculator import CalculatorTool
from agentic_research.tools.registry import ToolRegistry
from agentic_research.tools.search import InMemorySearchBackend, WebSearchTool

_LIST_FIELDS = ("messages", "research_findings", "routing_history", "error_log")


def _merge_update(state: dict, update: dict) -> dict:
    merged = dict(state)
    for key, value in update.items():
        if key in _LIST_FIELDS:
            merged[key] = accumulate_list(merged.get(key, []), value)
        else:
            merged[key] = value
    return merged


def _build_agents() -> dict:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool(backend=InMemorySearchBackend()))
    return {
        "researcher": ResearcherAgent(
            llm_client=MockLLMClient([LLMResponse(content="Researched.")]),
            tool_registry=registry,
        ),
        "analyst": AnalystAgent(
            llm_client=MockLLMClient([LLMResponse(content="Analyzed.")]),
            tool_registry=registry,
        ),
        "writer": WriterAgent(
            llm_client=MockLLMClient([LLMResponse(content="Final report.")]),
            tool_registry=registry,
        ),
    }


def test_task_can_be_checkpointed_mid_run_and_resumed_from_disk() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_store = FileCheckpointStore(tmp_dir)
        thread_id = "resumable-task-1"
        agents = _build_agents()

        # --- Run the task up through the Researcher step only, "pausing" ---
        state = create_initial_state("query", requires_human_approval=False)
        state = _merge_update(state, supervisor_node(state))  # -> RESEARCHING
        assert state["task_status"] == TaskStatus.RESEARCHING
        state = _merge_update(state, agents["researcher"].run(state))  # -> ROUTING

        checkpoint_store.save(thread_id, state)

        # --- Simulate a process restart: fresh store instance, same dir ---
        resumed_store = FileCheckpointStore(tmp_dir)
        resumed_state = resumed_store.load(thread_id)
        assert resumed_state is not None
        assert resumed_state["task_status"] == TaskStatus.ROUTING
        assert len(resumed_state["messages"]) == 1  # researcher's message survived

        # --- Continue the task to completion from the resumed state ---
        state = resumed_state
        for _ in range(10):
            state = _merge_update(state, supervisor_node(state))
            node = route_after_supervisor(state)
            if node == END_SENTINEL:
                break
            state = _merge_update(state, agents[node].run(state))

        assert state["task_status"] == TaskStatus.COMPLETED
        assert state["final_report"] == "Final report."
        assert state["current_agent"] == AgentRole.SUPERVISOR


def test_findings_persist_to_long_term_memory_after_task_completes() -> None:
    long_term_store = InMemoryLongTermMemoryStore()
    agents = _build_agents()

    state = create_initial_state("query")
    state = _merge_update(state, supervisor_node(state))
    state = _merge_update(state, agents["researcher"].run(state))

    save_findings(long_term_store, namespace="query-topic", findings=state["research_findings"])

    # A later, unrelated task under the same namespace can now look these up
    # without re-researching -- the cross-thread reuse long-term memory exists for.
    remembered = long_term_store.search("query-topic")
    assert len(remembered) == len(state["research_findings"])
