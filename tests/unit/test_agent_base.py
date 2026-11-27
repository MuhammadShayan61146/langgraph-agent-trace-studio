"""Unit tests for ``agentic_research.agents.base.BaseAgent``.

Exercises the shared loop via a minimal concrete subclass, since
`BaseAgent` itself is abstract.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentic_research.core.contracts import ResearchFinding
from agentic_research.core.enums import AgentRole, MessageRole, TaskStatus
from agentic_research.core.exceptions import InvalidStateTransitionError
from agentic_research.core.state import create_initial_state
from agentic_research.agents.base import (
    BaseAgent,
    LLMResponse,
    MockLLMClient,
    ToolCallRequest,
)
from agentic_research.tools.base import BaseTool, ToolParameter
from agentic_research.tools.registry import ToolRegistry


class _EchoTool(BaseTool):
    name = "echo"
    description = "Echoes the given text."
    parameters = [ToolParameter(name="text", type="string", description="text")]

    def _run(self, **kwargs: Any) -> str:
        return kwargs["text"]


class _AlwaysFailsTool(BaseTool):
    name = "always_fails"
    description = "Always fails."
    parameters = []

    def _run(self, **kwargs: Any) -> Any:
        raise RuntimeError("simulated tool failure")


class _TestAgent(BaseAgent):
    """Minimal concrete BaseAgent subclass for testing the shared loop."""

    role = AgentRole.RESEARCHER
    system_prompt = "test system prompt"
    next_status = TaskStatus.ROUTING
    allowed_tools = ["echo", "always_fails"]


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_EchoTool())
    registry.register(_AlwaysFailsTool())
    return registry


# --------------------------------------------------------------------------- #
# Subclass configuration enforcement
# --------------------------------------------------------------------------- #
def test_base_agent_rejects_subclass_missing_required_class_attributes() -> None:
    class _Incomplete(BaseAgent):
        role = AgentRole.RESEARCHER
        # system_prompt and next_status deliberately omitted

    with pytest.raises(TypeError):
        _Incomplete(llm_client=MockLLMClient([]), tool_registry=_registry())


# --------------------------------------------------------------------------- #
# No tool calls - single round trip
# --------------------------------------------------------------------------- #
def test_agent_with_no_tool_calls_returns_single_assistant_message() -> None:
    llm = MockLLMClient([LLMResponse(content="final answer")])
    agent = _TestAgent(llm_client=llm, tool_registry=_registry())
    state = create_initial_state("What is X?")
    state["task_status"] = TaskStatus.RESEARCHING

    update = agent.run(state)

    assert len(update["messages"]) == 1
    assert update["messages"][0].role == MessageRole.ASSISTANT
    assert update["messages"][0].content == "final answer"
    assert update["current_agent"] == AgentRole.RESEARCHER
    assert update["task_status"] == TaskStatus.ROUTING
    assert len(llm.call_history) == 1


# --------------------------------------------------------------------------- #
# Tool-calling loop
# --------------------------------------------------------------------------- #
def test_agent_executes_tool_call_and_makes_followup_call() -> None:
    llm = MockLLMClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        tool_name="echo", arguments={"text": "hi"}, call_id="c1"
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(content="done, echo said hi"),
        ]
    )
    agent = _TestAgent(llm_client=llm, tool_registry=_registry())
    state = create_initial_state("query")
    state["task_status"] = TaskStatus.RESEARCHING

    update = agent.run(state)

    assert len(llm.call_history) == 2
    # Second call's conversation must include the tool result.
    second_call_messages = llm.call_history[1]["messages"]
    assert any("hi" in m["content"] for m in second_call_messages)
    # One TOOL message + one ASSISTANT message.
    roles = [m.role for m in update["messages"]]
    assert roles == [MessageRole.TOOL, MessageRole.ASSISTANT]
    assert update["messages"][-1].content == "done, echo said hi"


def test_agent_stops_looping_after_max_tool_iterations() -> None:
    """If the LLM keeps requesting tools forever, the loop must not hang."""
    always_wants_tool = LLMResponse(
        content="still working",
        tool_calls=[
            ToolCallRequest(tool_name="echo", arguments={"text": "x"}, call_id="c")
        ],
        stop_reason="tool_use",
    )
    llm = MockLLMClient([always_wants_tool] * 3)  # exactly max_tool_iterations
    agent = _TestAgent(llm_client=llm, tool_registry=_registry())
    state = create_initial_state("query")
    state["task_status"] = TaskStatus.RESEARCHING

    update = agent.run(state)

    assert len(llm.call_history) == 3  # capped, did not exceed max_tool_iterations
    assert update["messages"][-1].content == "still working"


# --------------------------------------------------------------------------- #
# Error capture: unknown tool / failing tool
# --------------------------------------------------------------------------- #
def test_agent_captures_unknown_tool_name_as_error_record_not_a_crash() -> None:
    llm = MockLLMClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        tool_name="does_not_exist", arguments={}, call_id="c1"
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(content="handled the error"),
        ]
    )
    agent = _TestAgent(llm_client=llm, tool_registry=_registry())
    state = create_initial_state("query")
    state["task_status"] = TaskStatus.RESEARCHING

    update = agent.run(state)  # must not raise

    assert len(update["error_log"]) == 1
    assert update["error_log"][0].error_type == "ToolNotFoundError"
    # The failure must be visible to the LLM in the next call.
    assert "not available" in update["messages"][0].content


def test_agent_captures_failing_tool_execution_as_error_record() -> None:
    llm = MockLLMClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(tool_name="always_fails", arguments={}, call_id="c1")
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(content="handled it"),
        ]
    )
    agent = _TestAgent(llm_client=llm, tool_registry=_registry())
    state = create_initial_state("query")
    state["task_status"] = TaskStatus.RESEARCHING

    update = agent.run(state)

    assert len(update["error_log"]) == 1
    assert update["error_log"][0].error_type == "ToolExecutionFailure"
    assert "simulated tool failure" in update["error_log"][0].message


# --------------------------------------------------------------------------- #
# State transition validation
# --------------------------------------------------------------------------- #
def test_agent_raises_on_illegal_incoming_task_status() -> None:
    """PENDING -> ROUTING is legal, but this agent's next_status is ROUTING and
    it should reject running from an incoming status that can't legally reach
    ROUTING (e.g. COMPLETED, a terminal state)."""
    llm = MockLLMClient([LLMResponse(content="irrelevant")])
    agent = _TestAgent(llm_client=llm, tool_registry=_registry())
    state = create_initial_state("query")
    state["task_status"] = TaskStatus.COMPLETED

    with pytest.raises(InvalidStateTransitionError):
        agent.run(state)


# --------------------------------------------------------------------------- #
# _process_tool_result hook default behavior
# --------------------------------------------------------------------------- #
def test_default_process_tool_result_hook_returns_no_findings() -> None:
    llm = MockLLMClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        tool_name="echo", arguments={"text": "hi"}, call_id="c1"
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(content="done"),
        ]
    )
    agent = _TestAgent(llm_client=llm, tool_registry=_registry())
    state = create_initial_state("query")
    state["task_status"] = TaskStatus.RESEARCHING

    update = agent.run(state)

    assert update["research_findings"] == []


def test_extra_state_update_hook_is_merged_into_result() -> None:
    class _AgentWithExtra(_TestAgent):
        def _extra_state_update(self, state, final_content):  # noqa: ANN001
            return {"custom_field": "custom_value"}

    llm = MockLLMClient([LLMResponse(content="final")])
    agent = _AgentWithExtra(llm_client=llm, tool_registry=_registry())
    state = create_initial_state("query")
    state["task_status"] = TaskStatus.RESEARCHING

    update = agent.run(state)
    assert update["custom_field"] == "custom_value"
