"""Agent layer: LLM abstraction, the shared agent loop, and concrete agents.

Public API re-exported here, matching the pattern established by
`agentic_research.config`/`core`/`tools` 
"""

from agentic_research.agents.analyst import AnalystAgent
from agentic_research.agents.anthropic_client import AnthropicLLMClient
from agentic_research.agents.base import (
    BaseAgent,
    LLMClient,
    LLMResponse,
    MockLLMClient,
    ToolCallRequest,
)
from agentic_research.agents.researcher import ResearcherAgent
from agentic_research.agents.writer import WriterAgent

__all__ = [
    "LLMClient",
    "LLMResponse",
    "ToolCallRequest",
    "MockLLMClient",
    "BaseAgent",
    "ResearcherAgent",
    "AnalystAgent",
    "WriterAgent",
    "AnthropicLLMClient",
]
