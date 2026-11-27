"""The Researcher agent: gathers information using the web_search tool.


"""

from __future__ import annotations

from typing import Any

from agentic_research.core.contracts import ResearchFinding
from agentic_research.core.enums import AgentRole, TaskStatus
from agentic_research.tools.base import ToolResult
from agentic_research.agents.base import BaseAgent

#: Fixed confidence assigned to findings extracted from raw search results.
#: Not model-assessed -- see module docstring.
_DEFAULT_SEARCH_FINDING_CONFIDENCE = 0.7


class ResearcherAgent(BaseAgent):
    """Gathers information relevant to the task via the ``web_search`` tool."""

    role = AgentRole.RESEARCHER
    system_prompt = (
        "You are the Researcher agent in a multi-agent research system. "
        "Use the web_search tool to gather factual information relevant to "
        "the user's query. Once you have sufficient information, summarize "
        "what you found in plain text."
    )
    next_status = TaskStatus.ROUTING
    allowed_tools = ["web_search"]

    def _process_tool_result(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: ToolResult | None,
    ) -> list[ResearchFinding]:
        """Convert successful ``web_search`` results into `ResearchFinding`s.

        Args:
            tool_name: The tool that was called.
            arguments: Arguments the tool was called with (unused; findings
                are built from the result, not the request).
            result: The `ToolResult` from executing the tool.

        Returns:
            One `ResearchFinding` per search result item, or an empty list
            if the tool wasn't ``web_search``, failed, or returned nothing.
        """
        del arguments  # unused
        if tool_name != "web_search" or result is None or not result.success:
            return []
        return [
            ResearchFinding(
                source=item["url"],
                content=f"{item['title']}: {item['snippet']}",
                collected_by=self.role,
                confidence=_DEFAULT_SEARCH_FINDING_CONFIDENCE,
            )
            for item in result.output
        ]
