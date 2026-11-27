"""Agent base infrastructure: LLM abstraction + the shared agent execution loop.


"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentic_research.core.contracts import AgentMessage, ErrorRecord, ResearchFinding
from agentic_research.core.enums import AgentRole, MessageRole, TaskStatus
from agentic_research.core.exceptions import ToolNotFoundError
from agentic_research.core.state import GraphState, validate_transition
from agentic_research.tools.base import ToolResult
from agentic_research.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ToolCallRequest:
    """One tool invocation requested by the LLM.

    Attributes:
        tool_name: Name of the tool to call, as registered in a
            `ToolRegistry`.
        arguments: Keyword arguments to pass to the tool.
        call_id: An opaque identifier for this specific call, used to
            correlate the eventual tool result back to the request (mirrors
            Anthropic's `tool_use` block `id` field).
    """

    tool_name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """A provider-agnostic LLM completion result.

    Attributes:
        content: The model's text output. May be empty if the model only
            requested tool calls.
        tool_calls: Zero or more tools the model wants invoked before it
            will produce a final answer.
        stop_reason: Why generation stopped (e.g. ``"end_turn"``,
            ``"tool_use"``). Informational; `BaseAgent` decides whether to
            loop based on ``bool(tool_calls)``, not this field, to stay
            robust to provider-specific stop-reason naming.
    """

    content: str
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    stop_reason: str = "end_turn"


class LLMClient(Protocol):
    """Structural interface any LLM backend must satisfy.

    Mirrors `agentic_research.tools.search.SearchBackend`'s role: agents
    depend on this Protocol, never on a concrete provider SDK.
    """

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Generate a completion given a system prompt and conversation."""
        ...


class MockLLMClient:
    """A scripted, deterministic `LLMClient` for tests. **Not a real LLM.**

    Returns pre-configured `LLMResponse` objects in order, one per call to
    `complete`. Records every call it receives so tests can assert on what
    an agent actually sent (system prompt, message history, tool schemas).
    """

    def __init__(self, scripted_responses: list[LLMResponse]) -> None:
        """Initialize with the ordered sequence of responses to return.

        Args:
            scripted_responses: Responses returned in order, one per call.
                Must contain at least as many entries as the test expects
                `complete` to be called.
        """
        self._responses: list[LLMResponse] = list(scripted_responses)
        self.call_history: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Record the call and return the next scripted response.

        Raises:
            RuntimeError: If more calls are made than responses were
                scripted — almost always indicates a test's script doesn't
                match the agent logic under test (e.g. an unexpectedly
                long tool-calling loop).
        """
        self.call_history.append(
            {"system": system, "messages": list(messages), "tools": tools}
        )
        if not self._responses:
            raise RuntimeError(
                "MockLLMClient: complete() called but no scripted responses "
                "remain. Check the test's response script length against "
                "the expected number of LLM round-trips."
            )
        return self._responses.pop(0)


def _format_tool_result_for_llm(tool_name: str, result: ToolResult | None) -> str:
    """Render a `ToolResult` (or an unresolvable tool name) as LLM-readable text.

    Args:
        tool_name: The name of the tool that was invoked.
        result: The `ToolResult` returned, or ``None`` if the tool name
            could not be resolved at all (registry lookup failure).

    Returns:
        A short string suitable for feeding back into the conversation.
    """
    if result is None:
        return f"Error: tool '{tool_name}' is not available."
    if result.success:
        return f"{result.output}"
    return f"Error: {result.error_message}"


class BaseAgent(ABC):
    """Shared Template Method base for every concrete agent node.

    Class attributes subclasses must set:
        role: The `AgentRole` this agent represents.
        system_prompt: The system prompt sent on every LLM call.
        next_status: The `TaskStatus` the task moves to once this agent
            completes (validated against the incoming status via
            `validate_transition`).
        allowed_tools: Names of tools (from the injected `ToolRegistry`)
            this agent may call. ``None`` means "every registered tool".
    """

    role: AgentRole
    system_prompt: str
    next_status: TaskStatus
    allowed_tools: list[str] | None = None
    max_tool_iterations: int = 3

    _REQUIRED_CLASS_ATTRS = ("role", "system_prompt", "next_status")

    def __init__(self, llm_client: LLMClient, tool_registry: ToolRegistry) -> None:
        """Construct the agent with its injected LLM client and tool registry.

        Args:
            llm_client: Any object satisfying `LLMClient`. Production code
                injects a real provider client (see `anthropic_client.py`);
                tests inject `MockLLMClient`.
            tool_registry: The registry of tools available to the system.
                This agent will only actually call the subset named in
                `self.allowed_tools`, but receives the full registry so
                tool wiring stays centralized (Module 3's design).

        Raises:
            TypeError: If the concrete subclass has not set one of
                `role`/`system_prompt`/`next_status`. `BaseAgent` has no
                `@abstractmethod`-decorated methods (some agents, like
                `AnalystAgent`, override nothing), so ABC's usual
                instantiation guard doesn't apply here; this check is the
                real enforcement that a subclass is properly configured.
        """
        missing = [
            attr for attr in self._REQUIRED_CLASS_ATTRS if not hasattr(self, attr)
        ]
        if missing:
            raise TypeError(
                f"{type(self).__name__} must define class attribute(s): "
                f"{', '.join(missing)}"
            )
        self._llm_client = llm_client
        self._tool_registry = tool_registry

    def _tool_schemas(self) -> list[dict[str, Any]]:
        """Build the Anthropic-format tool schemas this agent may use.

        Returns:
            One schema dict per allowed tool, via `BaseTool.to_anthropic_schema`.
        """
        names = self.allowed_tools
        if names is None:
            names = self._tool_registry.list_tool_names()
        return [self._tool_registry.get(name).to_anthropic_schema() for name in names]

    def _messages_from_state(self, state: GraphState) -> list[dict[str, str]]:
        """Build the role/content message list sent to the LLM.

        Args:
            state: The current graph state.

        Returns:
            The original query as the first user message, followed by
            every prior `AgentMessage` collapsed to a plain role/content
            dict (see module docstring, "Design decisions", for why this
            is intentionally simple).
        """
        messages: list[dict[str, str]] = [
            {"role": "user", "content": state["original_query"]}
        ]
        for msg in state["messages"]:
            role = "assistant" if msg.role == MessageRole.ASSISTANT else "user"
            messages.append({"role": role, "content": msg.content})
        return messages

    def _process_tool_result(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: ToolResult | None,
    ) -> list[ResearchFinding]:
        """Hook: turn a tool result into zero or more `ResearchFinding`s.

        Default implementation returns no findings. Override in subclasses
        that should record structured findings from specific tools (e.g.
        `ResearcherAgent` overrides this for ``"web_search"`` results).

        Args:
            tool_name: The tool that was called.
            arguments: The arguments it was called with.
            result: The `ToolResult`, or ``None`` if the tool name could
                not be resolved.

        Returns:
            A list of `ResearchFinding` to append to graph state (may be
            empty).
        """
        del tool_name, arguments, result  # unused in the default hook
        return []

    def _extra_state_update(
        self, state: GraphState, final_content: str
    ) -> dict[str, Any]:
        """Hook: additional partial-state fields beyond the common ones.

        Default implementation adds nothing. Override in subclasses that
        need to set agent-specific fields (e.g. `WriterAgent` sets
        ``final_report``).

        Args:
            state: The current graph state.
            final_content: The LLM's final text output for this run.

        Returns:
            Additional key/value pairs to merge into this node's returned
            state update.
        """
        del state, final_content  # unused in the default hook
        return {}

    def run(self, state: GraphState) -> dict[str, Any]:
        """Execute this agent's turn: the shared tool-calling loop.

        This is the actual LangGraph node function body — Module 5 will
        register ``self.run`` directly as a node.

        Args:
            state: The current graph state.

        Returns:
            A partial state update dict: ``messages``, ``current_agent``,
            ``task_status``, ``research_findings``, ``error_log``, plus
            whatever `_extra_state_update` contributes.

        Raises:
            InvalidStateTransitionError: If moving from the incoming
                ``state["task_status"]`` to ``self.next_status`` is not a
                legal transition per `TASK_STATUS_TRANSITIONS` (Module 2).
        """
        validate_transition(state["task_status"], self.next_status)

        conversation = self._messages_from_state(state)
        tool_schemas = self._tool_schemas()

        new_messages: list[AgentMessage] = []
        findings: list[ResearchFinding] = []
        error_records: list[ErrorRecord] = []
        final_content = ""

        for _ in range(self.max_tool_iterations):
            response = self._llm_client.complete(
                system=self.system_prompt, messages=conversation, tools=tool_schemas
            )
            final_content = response.content

            if not response.tool_calls:
                break

            for call in response.tool_calls:
                try:
                    result: ToolResult | None = self._tool_registry.execute(
                        call.tool_name, **call.arguments
                    )
                except ToolNotFoundError as exc:
                    result = None
                    error_records.append(
                        ErrorRecord(
                            node_name=self.role.value,
                            error_type="ToolNotFoundError",
                            message=str(exc),
                        )
                    )
                else:
                    if not result.success:
                        error_records.append(
                            ErrorRecord(
                                node_name=self.role.value,
                                error_type="ToolExecutionFailure",
                                message=result.error_message or "unknown error",
                            )
                        )

                findings.extend(
                    self._process_tool_result(call.tool_name, call.arguments, result)
                )

                result_text = _format_tool_result_for_llm(call.tool_name, result)
                new_messages.append(
                    AgentMessage(
                        role=MessageRole.TOOL,
                        agent=self.role,
                        content=result_text,
                        metadata={"tool_name": call.tool_name, "call_id": call.call_id},
                    )
                )
                conversation.append(
                    {
                        "role": "user",
                        "content": f"[Tool result for {call.tool_name}]: {result_text}",
                    }
                )
        # Loop ends either via `break` (no more tool calls) or exhausting
        # max_tool_iterations; `final_content` holds the last response seen
        # either way (see module docstring, "Tradeoffs").

        new_messages.append(
            AgentMessage(
                role=MessageRole.ASSISTANT,
                agent=self.role,
                content=final_content or "(no content produced)",
            )
        )

        update: dict[str, Any] = {
            "messages": new_messages,
            "research_findings": findings,
            "error_log": error_records,
            "current_agent": self.role,
            "task_status": self.next_status,
        }
        update.update(self._extra_state_update(state, final_content))
        return update
