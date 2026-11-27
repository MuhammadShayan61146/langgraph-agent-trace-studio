"""Unit tests for ``agentic_research.config.logging_config``."""

from __future__ import annotations

import io
import json
import logging

import pytest

from agentic_research.config.logging_config import configure_logging, get_logger


def _reset_logger() -> None:
    """Test helper: fully reset the shared logger between tests.

    ``configure_logging`` is intentionally idempotent for production use,
    but tests need a clean slate each time to make assertions about handler
    count and output deterministic.
    """
    logger = logging.getLogger("agentic_research")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    if hasattr(logger, "_agentic_research_configured"):
        delattr(logger, "_agentic_research_configured")


def test_configure_logging_attaches_exactly_one_handler() -> None:
    _reset_logger()
    logger = configure_logging(level="DEBUG", log_format="text", stream=io.StringIO())
    assert len(logger.handlers) == 1


def test_configure_logging_is_idempotent() -> None:
    """Calling configure_logging twice must not stack duplicate handlers."""
    _reset_logger()
    stream = io.StringIO()
    configure_logging(level="INFO", log_format="text", stream=stream)
    logger = configure_logging(level="DEBUG", log_format="text", stream=stream)
    assert len(logger.handlers) == 1
    assert logger.level == logging.DEBUG


def test_configure_logging_rejects_unknown_level() -> None:
    _reset_logger()
    with pytest.raises(ValueError):
        configure_logging(level="NOT_REAL", log_format="text", stream=io.StringIO())


def test_text_format_writes_readable_line() -> None:
    _reset_logger()
    stream = io.StringIO()
    logger = configure_logging(level="INFO", log_format="text", stream=stream)
    logger.info("hello world")
    output = stream.getvalue()
    assert "hello world" in output
    assert "INFO" in output


def test_json_format_writes_valid_json_with_expected_keys() -> None:
    _reset_logger()
    stream = io.StringIO()
    logger = configure_logging(level="INFO", log_format="json", stream=stream)
    logger.info("structured message")
    line = stream.getvalue().strip().splitlines()[-1]
    parsed = json.loads(line)  # must not raise
    assert parsed["message"] == "structured message"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "agentic_research"
    assert "timestamp" in parsed


def test_get_logger_returns_namespaced_child_logger() -> None:
    child = get_logger("agents.researcher")
    assert child.name == "agentic_research.agents.researcher"


def test_get_logger_without_name_returns_top_level_logger() -> None:
    top = get_logger()
    assert top.name == "agentic_research"
