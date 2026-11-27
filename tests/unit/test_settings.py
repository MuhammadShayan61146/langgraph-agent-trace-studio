"""Unit tests for ``agentic_research.config.settings``.


"""

from __future__ import annotations

import pytest

from agentic_research.config.settings import Settings, load_settings
from agentic_research.core.exceptions import (
    ConfigurationError,
    MissingEnvironmentVariableError,
)


def test_settings_default_construction_succeeds() -> None:
    """Settings() with no arguments should use valid defaults."""
    settings = Settings()
    assert settings.model_name == "claude-sonnet-4-6"
    assert settings.temperature == 0.2
    assert settings.max_retries == 3
    assert settings.log_level == "INFO"


def test_settings_is_immutable() -> None:
    """Settings is a frozen dataclass; mutation must raise."""
    settings = Settings()
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        settings.temperature = 0.9  # type: ignore[misc]


def test_settings_rejects_out_of_range_temperature() -> None:
    """temperature outside [0.0, 1.0] must raise ConfigurationError."""
    with pytest.raises(ConfigurationError):
        Settings(temperature=1.5)
    with pytest.raises(ConfigurationError):
        Settings(temperature=-0.1)


def test_settings_rejects_negative_max_retries() -> None:
    with pytest.raises(ConfigurationError):
        Settings(max_retries=-1)


def test_settings_rejects_invalid_log_level() -> None:
    with pytest.raises(ConfigurationError):
        Settings(log_level="NOT_A_LEVEL")


def test_settings_rejects_invalid_log_format() -> None:
    with pytest.raises(ConfigurationError):
        Settings(log_format="xml")


def test_settings_rejects_empty_model_name() -> None:
    with pytest.raises(ConfigurationError):
        Settings(model_name="")


def test_require_anthropic_api_key_raises_when_missing() -> None:
    settings = Settings(anthropic_api_key=None)
    with pytest.raises(MissingEnvironmentVariableError):
        settings.require_anthropic_api_key()


def test_require_anthropic_api_key_returns_value_when_present() -> None:
    settings = Settings(anthropic_api_key="sk-test-123")
    assert settings.require_anthropic_api_key() == "sk-test-123"


def test_load_settings_reads_from_overrides_without_touching_environ() -> None:
    """overrides= must not mutate the real process environment."""
    import os

    sentinel_key = "AGENTIC_MODEL_NAME_TEST_SENTINEL"
    assert sentinel_key not in os.environ

    settings = load_settings(
        env_file=None,
        overrides={
            "AGENTIC_MODEL_NAME": "test-model",
            "AGENTIC_TEMPERATURE": "0.75",
            "AGENTIC_MAX_RETRIES": "5",
            "ANTHROPIC_API_KEY": "sk-override",
        },
    )
    assert settings.model_name == "test-model"
    assert settings.temperature == 0.75
    assert settings.max_retries == 5
    assert settings.anthropic_api_key == "sk-override"
    # Confirm no leakage into the real environment.
    assert "AGENTIC_MODEL_NAME" not in os.environ or os.environ.get(
        "AGENTIC_MODEL_NAME"
    ) != "test-model"


def test_load_settings_invalid_numeric_override_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        load_settings(env_file=None, overrides={"AGENTIC_TEMPERATURE": "not-a-float"})


def test_load_settings_defaults_when_no_env_file_and_no_overrides() -> None:
    settings = load_settings(env_file=None, overrides={})
    assert isinstance(settings, Settings)
    assert settings.model_name  # non-empty
