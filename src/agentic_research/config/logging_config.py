"""Application logging configuration.


"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

_LOGGER_NAME = "agentic_research"
_CONFIGURED_FLAG_ATTR = "_agentic_research_configured"


class _JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as a single-line JSON string.

        Args:
            record: The stdlib log record to format.

        Returns:
            A JSON-encoded string containing timestamp, level, logger name,
            message, and (if present) exception/context information.
        """
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extra_context = getattr(record, "context", None)
        if extra_context:
            payload["context"] = extra_context
        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO",
    log_format: str = "text",
    *,
    stream: Any = None,
) -> logging.Logger:
    """Configure and return the shared ``agentic_research`` logger.

    Idempotent: calling this multiple times reconfigures the level/format on
    the existing handler rather than stacking duplicate handlers, which
    would otherwise cause every log line to print multiple times.

    Args:
        level: Logging level name (e.g. ``"DEBUG"``, ``"INFO"``).
        log_format: ``"json"`` for structured single-line JSON output,
            ``"text"`` for a human-readable format.
        stream: Output stream for log records. Defaults to ``sys.stdout``.
            Exposed as a parameter primarily so tests can capture output.

    Returns:
        The configured ``logging.Logger`` instance named
        ``"agentic_research"``.

    Raises:
        ValueError: If ``level`` is not a recognized logging level name.
    """
    resolved_level = getattr(logging, level.upper(), None)
    if not isinstance(resolved_level, int):
        raise ValueError(f"Unknown log level: {level!r}")

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(resolved_level)

    formatter: logging.Formatter
    if log_format.lower() == "json":
        formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    already_configured = getattr(logger, _CONFIGURED_FLAG_ATTR, False)
    if already_configured:
        # Reconfigure the existing handler in place instead of adding a new one.
        for handler in logger.handlers:
            handler.setFormatter(formatter)
            handler.setLevel(resolved_level)
        return logger

    handler = logging.StreamHandler(stream=stream or sys.stdout)
    handler.setLevel(resolved_level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    setattr(logger, _CONFIGURED_FLAG_ATTR, True)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the ``agentic_research`` namespace.

    Args:
        name: Optional sub-module name (e.g. ``"agents.researcher"``). If
            omitted, the shared top-level logger is returned.

    Returns:
        A ``logging.Logger`` instance. Callers should prefer this over
        ``logging.getLogger(__name__)`` directly so that all loggers stay
        under the configured ``agentic_research`` namespace and inherit its
        handler/level.
    """
    if name is None:
        return logging.getLogger(_LOGGER_NAME)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
