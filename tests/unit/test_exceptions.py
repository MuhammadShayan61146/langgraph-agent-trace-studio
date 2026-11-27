"""Unit tests for ``agentic_research.core.exceptions``."""

from __future__ import annotations

from agentic_research.core.exceptions import (
    AgenticResearchError,
    ConfigurationError,
    MissingEnvironmentVariableError,
    RetryLimitExceededError,
    ToolNotFoundError,
)


def test_configuration_error_is_a_agentic_research_error() -> None:
    assert issubclass(ConfigurationError, AgenticResearchError)


def test_all_leaf_exceptions_inherit_from_root() -> None:
    for exc_type in (
        MissingEnvironmentVariableError,
        RetryLimitExceededError,
        ToolNotFoundError,
    ):
        assert issubclass(exc_type, AgenticResearchError)


def test_missing_environment_variable_error_carries_variable_name() -> None:
    err = MissingEnvironmentVariableError("ANTHROPIC_API_KEY")
    assert err.variable_name == "ANTHROPIC_API_KEY"
    assert "ANTHROPIC_API_KEY" in str(err)


def test_tool_not_found_error_carries_available_tools() -> None:
    err = ToolNotFoundError("search", available_tools=["calculator", "web_fetch"])
    assert err.tool_name == "search"
    assert err.details["available_tools"] == ["calculator", "web_fetch"]


def test_retry_limit_exceeded_error_message_contains_context() -> None:
    err = RetryLimitExceededError(
        node_name="researcher", attempts=3, last_error="timeout"
    )
    message = str(err)
    assert "researcher" in message
    assert "3" in message
    assert "timeout" in message


def test_base_error_without_details_stringifies_to_message_only() -> None:
    err = AgenticResearchError("plain failure")
    assert str(err) == "plain failure"


def test_base_error_with_details_includes_them_in_str() -> None:
    err = AgenticResearchError("failure", details={"key": "value"})
    assert "failure" in str(err)
    assert "key" in str(err)
