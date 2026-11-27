"""Tool registry: central lookup for every tool available to agents.

Purpose
-------
Provides a single place where tools are registered and from which agent
nodes (Module 4) look them up by name, instead of every agent hard-coding
references to specific tool classes.

Responsibilities
-----------------
* Register `BaseTool` instances under their `name`.
* Look up a tool by name, raising a clear, actionable error (listing what
  *is* available) when the name doesn't exist.
* List registered tool names and generate the combined Anthropic tool
  schema list for all registered tools, ready to pass into an LLM call.

Design decisions
-----------------
* **The registry holds tool *instances*, not classes.** Tools like
  `WebSearchTool` take constructor-injected dependencies (a search
  backend); the registry's job is to hold already-configured, ready-to-use
  tools, not to know how to construct them. This keeps `ToolRegistry`
  itself free of any tool-specific construction logic (SRP) and keeps
  dependency injection decisions at the call site that builds the registry
  (e.g. a future `build_default_registry(settings)` factory in Module 4),
  not inside the registry class.
* **Duplicate registration is rejected, not silently overwritten.** Two
  tools accidentally sharing a name is very likely a bug (an agent asking
  for "search" could silently get the wrong tool); failing fast at
  registration time is safer than failing subtly later when the wrong tool
  runs.
* **Lookup failures raise `ToolNotFoundError`** (defined in Module 1's
  exception hierarchy, extended with `available_tools` context) rather than
  returning `None`, so callers can't accidentally proceed with a missing
  tool silently — consistent with this project's fail-fast philosophy
  established in Module 1's `Settings` validation.

Tradeoffs
---------
* Rejecting duplicate names is stricter than some registries (which allow
  "last registered wins" for easy overriding in tests). Tests that need a
  different tool implementation under the same name should build a fresh
  `ToolRegistry` rather than re-register into a shared one — this is a
  minor test-authoring cost traded for a real safety guarantee.

Limitations
------------
* No namespacing/versioning of tools (e.g. `"search:v2"`). Not needed with
  two tools; revisit if the tool count grows large enough for name
  collisions across unrelated concerns to become likely.

Future improvements
--------------------
* Add a `build_default_registry(settings: Settings) -> ToolRegistry`
  factory once Module 4 needs to wire up production tool instances
  (including one backed by a real search API) from configuration.
"""

from __future__ import annotations

from agentic_research.core.exceptions import ToolNotFoundError
from agentic_research.tools.base import BaseTool, ToolResult


class ToolRegistry:
    """Holds a set of configured `BaseTool` instances, keyed by name.

    This class deliberately has no knowledge of *which* tools exist in the
    system (no import of `CalculatorTool`/`WebSearchTool`); callers register
    whatever tools they've constructed. This keeps the registry reusable
    for any future tool without modification (Open/Closed Principle).
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance under its ``name``.

        Args:
            tool: A fully constructed, ready-to-use tool.

        Raises:
            ValueError: If a tool with the same ``name`` is already
                registered.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"A tool named '{tool.name}' is already registered. "
                "Tool names must be unique within a registry."
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """Look up a registered tool by name.

        Args:
            name: The tool's registered name.

        Returns:
            The registered `BaseTool` instance.

        Raises:
            ToolNotFoundError: If no tool with ``name`` is registered.
                Carries the list of currently available tool names so the
                caller (or an LLM reading the error) can self-correct.
        """
        if name not in self._tools:
            raise ToolNotFoundError(
                tool_name=name, available_tools=self.list_tool_names()
            )
        return self._tools[name]

    def execute(self, name: str, **kwargs: object) -> ToolResult:
        """Look up and execute a tool by name in one call.

        Args:
            name: The tool's registered name.
            **kwargs: Arguments forwarded to the tool's ``execute()``.

        Returns:
            The `ToolResult` produced by the tool. Note this only raises
            `ToolNotFoundError` for an unknown *name*; a failure *inside*
            the tool itself is returned as ``ToolResult(success=False,
            ...)``, never raised (see `base.py` module docstring).

        Raises:
            ToolNotFoundError: If no tool with ``name`` is registered.
        """
        tool = self.get(name)
        return tool.execute(**kwargs)

    def list_tool_names(self) -> list[str]:
        """Return the names of all currently registered tools, sorted.

        Returns:
            A sorted list of tool names, for deterministic output in
            error messages, logs, and tests.
        """
        return sorted(self._tools.keys())

    def to_anthropic_tool_schemas(self) -> list[dict]:
        """Build the combined tool-definition list for all registered tools.

        Returns:
            A list of dicts, one per registered tool, in the shape produced
            by `BaseTool.to_anthropic_schema()`, ready to pass as the
            ``tools`` parameter of a real Anthropic Messages API call.
        """
        return [
            tool.to_anthropic_schema()
            for _, tool in sorted(self._tools.items())
        ]

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Return whether a tool with ``name`` is registered."""
        return name in self._tools
