"""The Writer agent: synthesizes the final report from gathered context.

"""

from __future__ import annotations

from typing import Any

from agentic_research.core.enums import AgentRole, TaskStatus
from agentic_research.core.state import GraphState
from agentic_research.agents.base import BaseAgent


class WriterAgent(BaseAgent):
    """Synthesizes the final report text from the conversation so far."""

    role = AgentRole.WRITER
    system_prompt = (
        "You are the Writer agent in a multi-agent research system. "
        "Synthesize the research findings and analysis gathered so far "
        "into a clear, well-structured final report answering the user's "
        "original query."
    )
    next_status = TaskStatus.AWAITING_HUMAN_APPROVAL
    allowed_tools: list[str] = []

    def _extra_state_update(
        self, state: GraphState, final_content: str
    ) -> dict[str, Any]:
        """Set ``final_report`` to this run's synthesized text.

        Args:
            state: The current graph state (unused; the report is built
                entirely from ``final_content``, which already reflects the
                full conversation via `BaseAgent._messages_from_state`).
            final_content: The LLM's final text output for this run.

        Returns:
            ``{"final_report": final_content}``.
        """
        del state  # unused
        return {"final_report": final_content}
