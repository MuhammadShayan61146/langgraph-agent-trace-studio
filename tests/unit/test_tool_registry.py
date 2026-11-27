"""Unit tests for ``agentic_research.tools.registry``."""

from __future__ import annotations

from typing import Any

import pytest

from agentic_research.core.exceptions import ToolNotFoundError
from agentic_research.tools.base import BaseTool, ToolParameter
from agentic_research.tools.registry import ToolRegistry


class _StubTool(BaseTool):
    """Minimal tool for registry tests; behavior is irrelevant here."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"Stub tool named {name}."
        self.parameters: list[ToolParameter] = []

    def _run(self, **kwargs: Any) -> str:
        return "ok"


def test_register_and_get_round_trip() -> None:
    registry = ToolRegistry()
    tool = _StubTool("alpha")
    registry.register(tool)
    assert registry.get("alpha") is tool


def test_register_rejects_duplicate_name() -> None:
    registry = ToolRegistry()
    registry.register(_StubTool("alpha"))
    with pytest.raises(ValueError):
        registry.register(_StubTool("alpha"))


def test_get_unknown_tool_raises_tool_not_found_with_available_list() -> None:
    registry = ToolRegistry()
    registry.register(_StubTool("alpha"))
    registry.register(_StubTool("beta"))
    with pytest.raises(ToolNotFoundError) as exc_info:
        registry.get("gamma")
    assert exc_info.value.details["available_tools"] == ["alpha", "beta"]


def test_list_tool_names_is_sorted() -> None:
    registry = ToolRegistry()
    registry.register(_StubTool("zeta"))
    registry.register(_StubTool("alpha"))
    assert registry.list_tool_names() == ["alpha", "zeta"]


def test_execute_looks_up_and_runs_tool() -> None:
    registry = ToolRegistry()
    registry.register(_StubTool("alpha"))
    result = registry.execute("alpha")
    assert result.success is True
    assert result.output == "ok"


def test_execute_unknown_tool_raises() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        registry.execute("nonexistent")


def test_len_and_contains() -> None:
    registry = ToolRegistry()
    assert len(registry) == 0
    assert "alpha" not in registry
    registry.register(_StubTool("alpha"))
    assert len(registry) == 1
    assert "alpha" in registry


def test_to_anthropic_tool_schemas_returns_one_schema_per_tool() -> None:
    registry = ToolRegistry()
    registry.register(_StubTool("alpha"))
    registry.register(_StubTool("beta"))
    schemas = registry.to_anthropic_tool_schemas()
    assert len(schemas) == 2
    assert {s["name"] for s in schemas} == {"alpha", "beta"}
