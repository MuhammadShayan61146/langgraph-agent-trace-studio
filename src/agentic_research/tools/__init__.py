"""Tool layer: the base tool contract, registry, and concrete tools.

Public API re-exported here so callers write::

    from agentic_research.tools import ToolRegistry, CalculatorTool, WebSearchTool

instead of reaching into submodules directly, matching the pattern
established by `agentic_research.config` and `agentic_research.core` in
Modules 1-2.
"""

from agentic_research.tools.base import BaseTool, ToolParameter, ToolResult
from agentic_research.tools.calculator import CalculatorTool, safe_eval
from agentic_research.tools.registry import ToolRegistry
from agentic_research.tools.search import (
    InMemorySearchBackend,
    SearchBackend,
    SearchResultItem,
    WebSearchTool,
)

__all__ = [
    "BaseTool",
    "ToolParameter",
    "ToolResult",
    "ToolRegistry",
    "CalculatorTool",
    "safe_eval",
    "WebSearchTool",
    "SearchBackend",
    "SearchResultItem",
    "InMemorySearchBackend",
]
