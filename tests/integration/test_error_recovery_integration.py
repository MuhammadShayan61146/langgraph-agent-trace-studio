"""Integration test: demonstrating
reuse across config, core exceptions/contracts, tools, and agents without
modifying any of them.
"""

from __future__ import annotations

import pytest

from agentic_research.config.settings import Settings
from agentic_research.core.enums import TaskStatus
from agentic_research.core.exceptions import RetryLimitExceededError
from agentic_research.core.state import create_initial_state
from agentic_research.agents.base import LLMResponse
from agentic_research.agents.researcher import ResearcherAgent
from agentic_research.error_recovery import with_retry
from agentic_research.tools.registry import ToolRegistry
from agentic_research.tools.search import InMemorySearchBackend, WebSearchTool


class _FlakyThenSuccessfulLLMClient:
    """A real `LLMClient`-satisfying stub: raises on its first `fail_times`
    calls (simulating a transient provider error), then returns
    `final_response`. Not a MockLLMClient subclass -- this is a distinct,
    minimal implementation proving `with_retry` works against anything
    satisfying the `LLMClient` Protocol, not just Module 4's test double.
    """

    def __init__(self, fail_times: int, final_response: LLMResponse) -> None:
        self._fail_times = fail_times
        self._final_response = final_response
        self.call_count = 0

    def complete(self, *, system, messages, tools=None):
        self.call_count += 1
        if self.call_count <= self._fail_times:
            raise ConnectionError(f"simulated transient provider error #{self.call_count}")
        return self._final_response


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WebSearchTool(backend=InMemorySearchBackend()))
    return registry


def test_with_retry_recovers_a_real_agent_from_a_transient_llm_failure() -> None:
    settings = Settings(max_retries=3)  # Module 1 config, sourced via DI
    llm = _FlakyThenSuccessfulLLMClient(
        fail_times=2, final_response=LLMResponse(content="Researched successfully.")
    )
    agent = ResearcherAgent(llm_client=llm, tool_registry=_registry())  # unmodified Module 4 class

    wrapped = with_retry(agent.run, max_retries=settings.max_retries, node_name="researcher")
    state = create_initial_state("query")
    state["task_status"] = TaskStatus.RESEARCHING

    update = wrapped(state)

    assert update["task_status"] == TaskStatus.ROUTING
    assert update["messages"][-1].content == "Researched successfully."
    assert llm.call_count == 3  # 2 failures + 1 success
    assert len(update["error_log"]) == 2
    assert all(e.node_name == "researcher" for e in update["error_log"])
    assert update["retry_count"] == 2


def test_with_retry_exhausts_settings_max_retries_and_raises() -> None:
    settings = Settings(max_retries=1)
    llm = _FlakyThenSuccessfulLLMClient(
        fail_times=99, final_response=LLMResponse(content="never reached")
    )
    agent = ResearcherAgent(llm_client=llm, tool_registry=_registry())

    wrapped = with_retry(agent.run, max_retries=settings.max_retries, node_name="researcher")
    state = create_initial_state("query")
    state["task_status"] = TaskStatus.RESEARCHING

    with pytest.raises(RetryLimitExceededError) as exc_info:
        wrapped(state)

    assert llm.call_count == settings.max_retries + 1
    assert exc_info.value.node_name == "researcher"
