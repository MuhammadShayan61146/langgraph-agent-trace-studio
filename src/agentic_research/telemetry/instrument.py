"""Non-invasive instrumentation: wrappers that emit events around existing parts.

Nothing here edits an existing module. Each wrapper takes the real object
(node function, LLM client, tool registry) and returns something with the
same interface that additionally publishes `TraceEvent`s and, optionally,
injects faults for resilience drills.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from agentic_research.agents.base import LLMClient, LLMResponse
from agentic_research.core.state import GraphState
from agentic_research.telemetry.context import current_attempt, current_node
from agentic_research.telemetry.events import EventType, RunEmitter, preview
from agentic_research.tools.base import ToolResult
from agentic_research.tools.registry import ToolRegistry


class InjectedFault(RuntimeError):
    """Raised by the fault injector to simulate a transient execution failure."""


@dataclass
class FaultRule:
    """One planned failure.

    Attributes:
        kind: ``"node"`` (node raises), ``"llm"`` (LLM call raises) or
            ``"tool"`` (tool returns ``success=False``).
        target: Node name for node/llm rules, tool name for tool rules.
        times: How many consecutive times the fault fires before clearing.
        message: Error text surfaced in events and error records.
    """

    kind: str
    target: str
    times: int = 1
    message: str = "Injected fault"


class FaultInjector:
    """Thread-safe, counted fault plan shared by all instrumented layers."""

    def __init__(self, rules: list[FaultRule] | None = None) -> None:
        self.rules = list(rules or [])
        self._fired: dict[int, int] = {}
        self._lock = threading.Lock()

    def check(self, kind: str, target: str, emitter: RunEmitter | None = None) -> FaultRule | None:
        """Return (and consume one firing of) a matching rule, if any."""
        with self._lock:
            for i, rule in enumerate(self.rules):
                if rule.kind == kind and rule.target == target:
                    if self._fired.get(i, 0) < rule.times:
                        self._fired[i] = self._fired.get(i, 0) + 1
                        fired = self._fired[i]
                        break
            else:
                return None
        if emitter is not None:
            emitter.emit(
                EventType.FAULT_INJECTED,
                node=current_node.get() or (target if kind != "tool" else None),
                kind=kind,
                target=target,
                message=rule.message,
                firing=fired,
                of=rule.times,
            )
        return rule


def instrument_node(
    fn: Callable[[GraphState], dict[str, Any]],
    node_name: str,
    emitter: RunEmitter,
    *,
    faults: FaultInjector | None = None,
    max_retries: int = 0,
) -> Callable[[GraphState], dict[str, Any]]:
    """Wrap a ``state -> dict`` node so every *attempt* is traced.

    Meant to sit **inside** `with_retry` (``with_retry(instrument_node(fn))``)
    so each retry attempt is visible, which `with_retry` alone hides.
    """
    attempts = {"n": 0}

    def wrapped(state: GraphState) -> dict[str, Any]:
        attempts["n"] += 1
        attempt = attempts["n"]
        node_token = current_node.set(node_name)
        attempt_token = current_attempt.set(attempt)
        emitter.emit(
            EventType.NODE_STARTED,
            node=node_name,
            attempt=attempt,
            task_status=_status(state),
        )
        started = time.perf_counter()
        try:
            if faults is not None:
                rule = faults.check("node", node_name, emitter)
                if rule is not None:
                    raise InjectedFault(rule.message)
            update = fn(state)
        except Exception as exc:  # noqa: BLE001 - re-raised below
            will_retry = attempt <= max_retries and type(exc).__name__ != "InvalidStateTransitionError"
            emitter.emit(
                EventType.NODE_ERROR,
                node=node_name,
                attempt=attempt,
                error_type=type(exc).__name__,
                message=str(exc),
                will_retry=will_retry,
                duration_ms=_ms(started),
            )
            if not will_retry:
                attempts["n"] = 0
            raise
        finally:
            current_node.reset(node_token)
            current_attempt.reset(attempt_token)
        attempts["n"] = 0
        status = update.get("task_status")
        emitter.emit(
            EventType.NODE_FINISHED,
            node=node_name,
            attempt=attempt,
            duration_ms=_ms(started),
            update_keys=sorted(update.keys()),
            task_status=getattr(status, "value", status),
        )
        return update

    return wrapped


class TracedLLMClient:
    """`LLMClient` decorator that emits request/response events."""

    def __init__(
        self, inner: LLMClient, emitter: RunEmitter, faults: FaultInjector | None = None
    ) -> None:
        self._inner = inner
        self._emitter = emitter
        self._faults = faults
        self._round: dict[str, int] = {}

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        node = current_node.get()
        key = f"{node}:{current_attempt.get()}"
        self._round[key] = self._round.get(key, 0) + 1
        self._emitter.emit(
            EventType.LLM_REQUEST,
            node=node,
            round=self._round[key],
            message_count=len(messages),
            tools=[t.get("name") for t in (tools or [])],
            last_message=preview(messages[-1]["content"], 400) if messages else "",
        )
        started = time.perf_counter()
        if self._faults is not None and node:
            rule = self._faults.check("llm", node, self._emitter)
            if rule is not None:
                raise InjectedFault(rule.message)
        response = self._inner.complete(system=system, messages=messages, tools=tools)
        self._emitter.emit(
            EventType.LLM_RESPONSE,
            node=node,
            round=self._round[key],
            content=response.content,
            tool_calls=[
                {"tool": c.tool_name, "arguments": c.arguments, "call_id": c.call_id}
                for c in response.tool_calls
            ],
            stop_reason=response.stop_reason,
            duration_ms=_ms(started),
        )
        return response


class TracedToolRegistry(ToolRegistry):
    """`ToolRegistry` subclass that traces calls and can inject tool failures.

    Construct it from an existing registry with `from_registry`; every tool
    instance is shared, only `execute` is decorated.
    """

    def __init__(self, emitter: RunEmitter, faults: FaultInjector | None = None) -> None:
        super().__init__()
        self._emitter = emitter
        self._faults = faults

    @classmethod
    def from_registry(
        cls, registry: ToolRegistry, emitter: RunEmitter, faults: FaultInjector | None = None
    ) -> "TracedToolRegistry":
        traced = cls(emitter, faults)
        for name in registry.list_tool_names():
            traced.register(registry.get(name))
        return traced

    def execute(self, name: str, **kwargs: object) -> ToolResult:
        node = current_node.get()
        self._emitter.emit(EventType.TOOL_CALL, node=node, tool=name, arguments=dict(kwargs))
        started = time.perf_counter()
        try:
            rule = self._faults.check("tool", name, self._emitter) if self._faults else None
            if rule is not None:
                result = ToolResult(tool_name=name, success=False, error_message=rule.message)
            else:
                result = super().execute(name, **kwargs)
        except Exception as exc:  # e.g. ToolNotFoundError
            self._emitter.emit(
                EventType.TOOL_RESULT,
                node=node,
                tool=name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                output=None,
                duration_ms=_ms(started),
            )
            raise
        self._emitter.emit(
            EventType.TOOL_RESULT,
            node=node,
            tool=name,
            success=result.success,
            output=result.output,
            error=result.error_message,
            duration_ms=_ms(started),
        )
        return result


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _status(state: GraphState) -> str | None:
    s = state.get("task_status")
    return getattr(s, "value", s)
