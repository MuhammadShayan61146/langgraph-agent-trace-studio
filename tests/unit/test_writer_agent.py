"""Unit tests for ``agentic_research.agents.writer.WriterAgent``."""

from __future__ import annotations

from agentic_research.core.enums import TaskStatus
from agentic_research.core.state import create_initial_state
from agentic_research.agents.base import LLMResponse, MockLLMClient
from agentic_research.agents.writer import WriterAgent
from agentic_research.tools.registry import ToolRegistry


def test_writer_has_no_tools_available() -> None:
    llm = MockLLMClient([LLMResponse(content="Final report text.")])
    agent = WriterAgent(llm_client=llm, tool_registry=ToolRegistry())
    state = create_initial_state("query")
    state["task_status"] = TaskStatus.WRITING

    agent.run(state)

    assert llm.call_history[0]["tools"] == []


def test_writer_sets_final_report_and_awaits_approval() -> None:
    llm = MockLLMClient([LLMResponse(content="This is the synthesized report.")])
    agent = WriterAgent(llm_client=llm, tool_registry=ToolRegistry())
    state = create_initial_state("query")
    state["task_status"] = TaskStatus.WRITING

    update = agent.run(state)

    assert update["final_report"] == "This is the synthesized report."
    assert update["task_status"] == TaskStatus.AWAITING_HUMAN_APPROVAL


def test_writer_cannot_run_from_pending_status() -> None:
    """WRITING is reachable only via ROUTING per TASK_STATUS_TRANSITIONS;
    a task still PENDING has no legal path to AWAITING_HUMAN_APPROVAL."""
    import pytest
    from agentic_research.core.exceptions import InvalidStateTransitionError

    llm = MockLLMClient([LLMResponse(content="irrelevant")])
    agent = WriterAgent(llm_client=llm, tool_registry=ToolRegistry())
    state = create_initial_state("query")  # task_status defaults to PENDING

    with pytest.raises(InvalidStateTransitionError):
        agent.run(state)
