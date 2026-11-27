"""The Analyst agent: reasons over collected findings, using the calculator tool.


"""

from __future__ import annotations

from agentic_research.core.enums import AgentRole, TaskStatus
from agentic_research.agents.base import BaseAgent


class AnalystAgent(BaseAgent):
    """Analyzes research findings, optionally performing calculations."""

    role = AgentRole.ANALYST
    system_prompt = (
        "You are the Analyst agent in a multi-agent research system. "
        "Review the research findings gathered so far and identify key "
        "insights, trends, or numeric conclusions. Use the calculator tool "
        "for any arithmetic. Summarize your analysis in plain text."
    )
    next_status = TaskStatus.ROUTING
    allowed_tools = ["calculator"]
