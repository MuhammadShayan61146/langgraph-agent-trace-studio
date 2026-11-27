"""Unit tests for ``agentic_research.tools.base``."""

from __future__ import annotations

from typing import Any

import pytest

from agentic_research.tools.base import BaseTool, ToolParameter, ToolResult


class _EchoTool(BaseTool):
    """A minimal concrete tool used only to exercise BaseTool's shared logic."""

    name = "echo"
    description = "Echoes back the given text, optionally repeated."
    parameters = [
        ToolParameter(name="text", type="string", description="Text to echo."),
        ToolParameter(
            name="repeat",
            type="integer",
            description="Number of repetitions.",
            required=False,
        ),
    ]

    def _run(self, **kwargs: Any) -> str:
        return kwargs["text"] * kwargs.get("repeat", 1)


class _AlwaysFailsTool(BaseTool):
    """A tool whose _run always raises, to test execute()'s error wrapping."""

    name = "always_fails"
    description = "Always raises a RuntimeError."
    parameters = []

    def _run(self, **kwargs: Any) -> Any:
        raise RuntimeError("boom")


# --------------------------------------------------------------------------- #
# ToolParameter
# --------------------------------------------------------------------------- #
def test_tool_parameter_accepts_valid_type() -> None:
    param = ToolParameter(name="x", type="string", description="desc")
    assert param.required is True


def test_tool_parameter_rejects_unsupported_type() -> None:
    with pytest.raises(ValueError):
        ToolParameter(name="x", type="not_a_real_type", description="desc")


# --------------------------------------------------------------------------- #
# BaseTool.execute() - success paths
# --------------------------------------------------------------------------- #
def test_execute_returns_successful_result() -> None:
    result = _EchoTool().execute(text="hi", repeat=2)
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.output == "hihi"
    assert result.error_message is None


def test_execute_applies_default_for_missing_optional_param() -> None:
    result = _EchoTool().execute(text="hi")
    assert result.success is True
    assert result.output == "hi"


# --------------------------------------------------------------------------- #
# BaseTool.execute() - validation failures
# --------------------------------------------------------------------------- #
def test_execute_fails_gracefully_on_missing_required_param() -> None:
    result = _EchoTool().execute(repeat=2)
    assert result.success is False
    assert "text" in result.error_message
    assert result.output is None


def test_execute_fails_gracefully_on_wrong_type() -> None:
    result = _EchoTool().execute(text="hi", repeat="not-an-int")
    assert result.success is False
    assert "repeat" in result.error_message


def test_execute_rejects_bool_for_integer_param() -> None:
    """bool is a subclass of int in Python; must not silently pass as integer."""
    result = _EchoTool().execute(text="hi", repeat=True)
    assert result.success is False


# --------------------------------------------------------------------------- #
# BaseTool.execute() - internal exceptions never propagate
# --------------------------------------------------------------------------- #
def test_execute_never_raises_even_when_run_raises() -> None:
    result = _AlwaysFailsTool().execute()
    assert result.success is False
    assert "boom" in result.error_message
    assert result.tool_name == "always_fails"


# --------------------------------------------------------------------------- #
# to_anthropic_schema()
# --------------------------------------------------------------------------- #
def test_to_anthropic_schema_has_expected_shape() -> None:
    schema = _EchoTool().to_anthropic_schema()
    assert schema["name"] == "echo"
    assert schema["description"]
    assert schema["input_schema"]["type"] == "object"
    assert "text" in schema["input_schema"]["properties"]
    assert schema["input_schema"]["required"] == ["text"]


def test_to_anthropic_schema_excludes_optional_params_from_required() -> None:
    schema = _EchoTool().to_anthropic_schema()
    assert "repeat" not in schema["input_schema"]["required"]
    assert "repeat" in schema["input_schema"]["properties"]
