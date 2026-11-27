"""Tool abstraction: the base contract every callable tool implements.

Purpose
-------
Defines what a "tool" is in this system, independent of any specific tool
(calculator, web search, ...) and independent of any LLM provider's
function-calling wire format. Concrete tools (Module 3's `calculator.py`,
`search.py`) implement this contract; agent nodes (Module 4) will consume
it via `ToolRegistry` (this module's `registry.py`).

Responsibilities
-----------------
* `ToolParameter` — describes one input a tool accepts, in a
  provider-agnostic shape.
* `ToolResult` — the uniform outcome of running a tool: either a success
  with output, or a failure with an error message. Callers (agents) always
  get a `ToolResult` back and never need to catch tool-internal exceptions
  themselves.
* `BaseTool` — the abstract base every tool subclasses. Provides the
  *shared* execution wrapper (`execute`) so subclasses only implement the
  tool-specific logic (`_run`) and never re-implement error handling or
  schema generation.

Design decisions
-----------------
* **Template Method pattern**: `BaseTool.execute()` is concrete and final in
  spirit (not intended to be overridden); it calls the abstract `_run()`
  that subclasses implement. This guarantees every tool gets identical
  argument validation and exception-to-`ToolResult` translation without
  each tool re-writing that logic — directly satisfying DRY and the
  Liskov Substitution Principle (any `BaseTool` can be executed the same
  way by calling code, regardless of which concrete tool it is).
* **`to_anthropic_schema()` lives on the base class**, generating the
  provider tool-definition dict from `name`/`description`/`parameters`.
  Every concrete tool gets this for free; none needs to hand-write its own
  JSON schema. The output shape matches Anthropic's Messages API `tools`
  parameter format (`{"name", "description", "input_schema": {"type":
  "object", "properties": {...}, "required": [...]}}`), so it is exactly
  what Module 4/5 will pass into a real API call — but note it has not
  been validated against a live Anthropic API call in this sandbox (no
  network); it is checked here against the *documented* schema shape only.
* **`ToolResult` never raises.** A tool that fails (bad input, runtime
  error) returns `ToolResult(success=False, error_message=...)` rather
  than propagating an exception to the agent loop. This mirrors how LLM
  tool-calling actually works in production: the failure needs to become
  text the model can read and react to, not a Python exception that
  crashes the graph. `ToolNotFoundError` (raised by the *registry*, not by
  a tool) is the one deliberate exception to this — see `registry.py`.

Tradeoffs
---------
* Wrapping every tool exception into a `ToolResult` means a genuinely
  unexpected bug inside a tool (e.g. a `TypeError` from bad tool code, not
  bad tool *input*) is also swallowed into a graceful failure result rather
  than crashing loudly. We accept this because the whole point of a
  tool-calling layer is resilience to tool failure; Module 8's error
  recovery module is where `error_log` (already present in `GraphState`,
  Module 2) turns these into escalation/retry decisions rather than silent
  failure.

Limitations
------------
* Parameter validation here is shallow (presence + basic Python type
  check), not full JSON Schema validation (no nested object/array shape
  checking). Sufficient for the two tools in this module; flagged for
  Module 4+ if agents need richer tool inputs.

Future improvements
--------------------
* Add async `_arun`/`aexecute` variants once an agent node needs concurrent
  tool calls (not required by any tool in this module).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agentic_research.core.exceptions import ToolExecutionError

#: Maps our provider-agnostic parameter type names to the Python types used
#: for shallow runtime validation in `BaseTool._validate_arguments`.
_PYTHON_TYPE_BY_SCHEMA_TYPE: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


@dataclass(frozen=True, slots=True)
class ToolParameter:
    """Describes a single input parameter a tool accepts.

    Attributes:
        name: Parameter name, as it will appear in ``**kwargs`` passed to
            the tool and in the generated schema's ``properties``.
        type: One of ``"string"``, ``"number"``, ``"integer"``,
            ``"boolean"``, ``"array"``, ``"object"`` — matches JSON Schema
            primitive type names, which is what LLM tool-calling APIs
            (including Anthropic's) expect.
        description: Human/model-readable explanation of what this
            parameter means, used directly in the generated tool schema so
            the LLM knows how to fill it in.
        required: Whether the tool call must include this parameter.
    """

    name: str
    type: str
    description: str
    required: bool = True

    def __post_init__(self) -> None:
        """Validate that ``type`` is one of the supported schema types."""
        if self.type not in _PYTHON_TYPE_BY_SCHEMA_TYPE:
            raise ValueError(
                f"Unsupported ToolParameter.type: {self.type!r}. "
                f"Must be one of {sorted(_PYTHON_TYPE_BY_SCHEMA_TYPE)}."
            )


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The uniform outcome of executing a tool.

    Attributes:
        tool_name: Name of the tool that produced this result.
        success: Whether execution completed without error.
        output: The tool's return value on success; ``None`` on failure.
        error_message: A human-readable failure description; ``None`` on
            success.
        metadata: Free-form structured context (e.g. execution time),
            defaults to an empty dict so callers never need a null check.
    """

    tool_name: str
    success: bool
    output: Any = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Abstract base class every concrete tool must subclass.

    Subclasses provide ``name``, ``description``, and ``parameters`` as
    class-level attributes, and implement ``_run`` with the tool's actual
    logic. Calling code should always use ``execute()``, never ``_run()``
    directly, so that argument validation and error handling are applied
    consistently (Template Method pattern).
    """

    name: str
    description: str
    parameters: list[ToolParameter] = []

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """Execute the tool's actual logic and return a raw result.

        Subclasses implement this. It may raise any exception; ``execute()``
        is responsible for catching it and translating it into a failed
        ``ToolResult`` — subclasses should not do their own top-level
        try/except for that purpose.

        Args:
            **kwargs: Tool-specific arguments, matching ``self.parameters``.

        Returns:
            The tool's raw output. Wrapped into ``ToolResult.output`` by
            ``execute()``.
        """
        raise NotImplementedError

    def _validate_arguments(self, kwargs: dict[str, Any]) -> None:
        """Check required parameters are present and shallowly type-correct.

        Args:
            kwargs: The arguments the caller is about to pass to ``_run``.

        Raises:
            ToolExecutionError: If a required parameter is missing, or a
                provided parameter's Python type doesn't match its declared
                schema type.
        """
        errors: list[str] = []
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                errors.append(f"missing required parameter '{param.name}'")
                continue
            if param.name in kwargs:
                expected_types = _PYTHON_TYPE_BY_SCHEMA_TYPE[param.type]
                value = kwargs[param.name]
                # bool is a subclass of int in Python; only accept bool for
                # "boolean" params, not silently for "integer"/"number" ones.
                if param.type in ("integer", "number") and isinstance(value, bool):
                    errors.append(
                        f"parameter '{param.name}' expected {param.type}, got bool"
                    )
                elif not isinstance(value, expected_types):
                    errors.append(
                        f"parameter '{param.name}' expected type "
                        f"{param.type!r}, got {type(value).__name__!r}"
                    )
        if errors:
            raise ToolExecutionError(
                f"Invalid arguments for tool '{self.name}': " + "; ".join(errors),
                details={"tool_name": self.name, "errors": errors},
            )

    def execute(self, **kwargs: Any) -> ToolResult:
        """Validate arguments, run the tool, and return a uniform result.

        This is the one method calling code (the registry, agent nodes)
        should ever call. It never raises: any exception from validation or
        from ``_run`` is caught and turned into a failed ``ToolResult``.

        Args:
            **kwargs: Tool-specific arguments, matching ``self.parameters``.

        Returns:
            A ``ToolResult`` describing success (with ``output``) or
            failure (with ``error_message``).
        """
        try:
            self._validate_arguments(kwargs)
            output = self._run(**kwargs)
        except Exception as exc:  # noqa: BLE001 - intentional: see module docstring
            return ToolResult(
                tool_name=self.name,
                success=False,
                error_message=f"{type(exc).__name__}: {exc}",
            )
        return ToolResult(tool_name=self.name, success=True, output=output)

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Build this tool's definition in Anthropic Messages API tool format.

        Returns:
            A dict of the shape
            ``{"name": ..., "description": ..., "input_schema": {"type":
            "object", "properties": {...}, "required": [...]}}``, suitable
            to include directly in the ``tools`` parameter of a real
            Anthropic API call in Module 4/5. Structurally validated in this
            module's tests against the documented schema shape; not yet
            validated against a live API call (no network in this sandbox).
        """
        properties = {
            param.name: {"type": param.type, "description": param.description}
            for param in self.parameters
        }
        required = [param.name for param in self.parameters if param.required]
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
