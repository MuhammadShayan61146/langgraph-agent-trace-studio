"""Production `LLMClient` backed by the real Anthropic API.


Future implementation sketch (not executed, for a future module/session
with network + package access):

    from anthropic import Anthropic
    from agentic_research.agents.base import LLMResponse, ToolCallRequest

    class AnthropicLLMClient:
        def __init__(self, settings: Settings) -> None:
            self._client = Anthropic(api_key=settings.require_anthropic_api_key())
            self._model = settings.model_name
            self._temperature = settings.temperature

        def complete(self, *, system, messages, tools=None) -> LLMResponse:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                temperature=self._temperature,
                system=system,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                tools=tools or [],
            )
            text = "".join(b.text for b in response.content if b.type == "text")
            tool_calls = [
                ToolCallRequest(tool_name=b.name, arguments=b.input, call_id=b.id)
                for b in response.content
                if b.type == "tool_use"
            ]
            return LLMResponse(
                content=text, tool_calls=tool_calls, stop_reason=response.stop_reason
            )

This sketch has not been executed or syntax-checked against the real
`anthropic` SDK (which is not installed here) and should be treated as a
design draft, not verified code, when a future module implements it for
real.
"""

from __future__ import annotations

from typing import Any


class AnthropicLLMClient:
    """Placeholder for the real Anthropic-backed `LLMClient` implementation.

    Deliberately not implemented — see module docstring.
    """

    def __init__(self, settings: Any) -> None:
        """Raise immediately, explaining what real implementation requires.

        Args:
            settings: Intended to be an `agentic_research.config.Settings`
                instance; typed as ``Any`` here to avoid this file
                importing `agentic_research.config` for a constructor that
                never actually runs, preserving this module's dependency
                footprint until it's genuinely implemented.

        Raises:
            NotImplementedError: Always. This sandbox has no network access
                and no Anthropic SDK installed (see Module 1 report,
                "Environment limitations"); this class is a documented seam
                for a future module/session to fill in, not a stub meant to
                silently succeed.
        """
        del settings  # unused; would be used in the real implementation
        raise NotImplementedError(
            "AnthropicLLMClient is not implemented in this environment: no "
            "network access and no 'anthropic'/'langchain-anthropic' "
            "package installed. See this module's docstring for an "
            "unexecuted implementation sketch, and docs/MODULE_04_REPORT.md "
            "'Known Risks' for details."
        )
