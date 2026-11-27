"""Unit test for ``agentic_research.agents.anthropic_client.AnthropicLLMClient``.


"""

from __future__ import annotations

import pytest

from agentic_research.agents.anthropic_client import AnthropicLLMClient


def test_anthropic_llm_client_raises_not_implemented_with_explanation() -> None:
    with pytest.raises(NotImplementedError) as exc_info:
        AnthropicLLMClient(settings=None)
    message = str(exc_info.value)
    assert "network" in message.lower()
    assert "anthropic" in message.lower()
