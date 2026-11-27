"""Application configuration.

"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from agentic_research.core.exceptions import (
    ConfigurationError,
    MissingEnvironmentVariableError,
)

try:  # pragma: no cover - exercised indirectly; dotenv is an optional convenience
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover
    _load_dotenv = None

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_VALID_LOG_FORMATS = frozenset({"json", "text"})


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable, validated application configuration.

    Attributes:
        anthropic_api_key: API key for the Anthropic Claude API. Required for
            any node that actually calls an LLM; not required to construct
            ``Settings`` itself (validated lazily by callers that need it,
            see ``require_anthropic_api_key``) so that non-LLM code paths
            (config tests, graph structure validation) don't need a real key.
        model_name: Anthropic model identifier used by agent nodes.
        temperature: Sampling temperature passed to the LLM. Must be within
            [0.0, 1.0].
        max_retries: Maximum number of retry attempts for a recoverable node
            failure before ``RetryLimitExceededError`` is raised.
        request_timeout_seconds: Per-LLM-call timeout.
        log_level: Python logging level name.
        log_format: Either ``"json"`` (machine-readable, for production) or
            ``"text"`` (human-readable, for local development).
        environment: Deployment environment label (``development``,
            ``staging``, ``production``); influences default log format.
        project_root: Absolute path to the project root, used to resolve
            relative paths (e.g. for a ``.env`` file) consistently.
    """

    model_name: str = "claude-sonnet-4-6"
    temperature: float = 0.2
    max_retries: int = 3
    request_timeout_seconds: int = 60
    log_level: str = "INFO"
    log_format: str = "text"
    environment: str = "development"
    project_root: Path = field(default_factory=Path.cwd)
    anthropic_api_key: str | None = None

    def __post_init__(self) -> None:
        """Validate field values immediately after dataclass construction."""
        self._validate()

    def _validate(self) -> None:
        """Validate field values, raising ``ConfigurationError`` on failure.

        Raises:
            ConfigurationError: If any field holds a value outside its
                documented valid range or set.
        """
        errors: list[str] = []

        if not (0.0 <= self.temperature <= 1.0):
            errors.append(
                f"temperature must be within [0.0, 1.0], got {self.temperature!r}"
            )
        if self.max_retries < 0:
            errors.append(f"max_retries must be >= 0, got {self.max_retries!r}")
        if self.request_timeout_seconds <= 0:
            errors.append(
                "request_timeout_seconds must be > 0, "
                f"got {self.request_timeout_seconds!r}"
            )
        if self.log_level.upper() not in _VALID_LOG_LEVELS:
            errors.append(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, "
                f"got {self.log_level!r}"
            )
        if self.log_format.lower() not in _VALID_LOG_FORMATS:
            errors.append(
                f"log_format must be one of {sorted(_VALID_LOG_FORMATS)}, "
                f"got {self.log_format!r}"
            )
        if not self.model_name:
            errors.append("model_name must be a non-empty string")

        if errors:
            raise ConfigurationError(
                "Invalid settings: " + "; ".join(errors),
                details={"errors": errors},
            )

    def require_anthropic_api_key(self) -> str:
        """Return the Anthropic API key, raising if it is not configured.

        This is deliberately separate from ``__post_init__`` validation so
        that code paths which never call an LLM (graph structure tests,
        routing logic tests) can construct ``Settings`` without needing a
        real API key.

        Returns:
            The configured Anthropic API key.

        Raises:
            MissingEnvironmentVariableError: If no API key was configured.
        """
        if not self.anthropic_api_key:
            raise MissingEnvironmentVariableError("ANTHROPIC_API_KEY")
        return self.anthropic_api_key


def load_settings(
    env_file: str | Path | None = ".env",
    *,
    overrides: dict[str, str] | None = None,
) -> Settings:
    """Load and validate ``Settings`` from environment variables.

    Args:
        env_file: Path to a ``.env`` file to load before reading environment
            variables. Ignored if the file does not exist or if
            ``python-dotenv`` is not installed. Pass ``None`` to skip
            ``.env`` loading entirely (useful in production containers).
        overrides: Optional mapping merged on top of ``os.environ`` for the
            purposes of this call only. Intended for tests, so they can
            inject configuration without mutating global process state.

    Returns:
        A validated ``Settings`` instance.

    Raises:
        ConfigurationError: If any loaded value fails validation.
    """
    if env_file is not None and _load_dotenv is not None:
        env_path = Path(env_file)
        if env_path.exists():
            _load_dotenv(dotenv_path=env_path, override=False)

    source: dict[str, str] = dict(os.environ)
    if overrides:
        source.update(overrides)

    def _get_str(key: str, default: str) -> str:
        return source.get(key, default)

    def _get_float(key: str, default: float) -> float:
        raw = source.get(key)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError as exc:
            raise ConfigurationError(
                f"Environment variable '{key}' must be a float, got {raw!r}"
            ) from exc

    def _get_int(key: str, default: int) -> int:
        raw = source.get(key)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigurationError(
                f"Environment variable '{key}' must be an int, got {raw!r}"
            ) from exc

    return Settings(
        model_name=_get_str("AGENTIC_MODEL_NAME", "claude-sonnet-4-6"),
        temperature=_get_float("AGENTIC_TEMPERATURE", 0.2),
        max_retries=_get_int("AGENTIC_MAX_RETRIES", 3),
        request_timeout_seconds=_get_int("AGENTIC_TIMEOUT_SECONDS", 60),
        log_level=_get_str("AGENTIC_LOG_LEVEL", "INFO"),
        log_format=_get_str("AGENTIC_LOG_FORMAT", "text"),
        environment=_get_str("AGENTIC_ENVIRONMENT", "development"),
        project_root=Path(_get_str("AGENTIC_PROJECT_ROOT", str(Path.cwd()))),
        anthropic_api_key=source.get("ANTHROPIC_API_KEY"),
    )
