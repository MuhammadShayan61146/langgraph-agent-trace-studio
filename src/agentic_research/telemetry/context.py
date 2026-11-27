"""Context variables linking low-level hooks (LLM/tool calls) to the active node.

`BaseAgent.run` calls the LLM client and tool registry without telling them
which agent is calling. Rather than change that contract, the node wrapper
publishes the active node name here and the traced client/registry read it.
"""

from __future__ import annotations

from contextvars import ContextVar

current_node: ContextVar[str | None] = ContextVar("agentic_current_node", default=None)
current_attempt: ContextVar[int] = ContextVar("agentic_current_attempt", default=1)
