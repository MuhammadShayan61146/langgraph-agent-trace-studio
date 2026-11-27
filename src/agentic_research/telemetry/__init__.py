"""Telemetry layer: trace events, instrumentation wrappers, traced runner, dossiers.

Purely additive -- nothing in the core package imports this package, so the
existing modules, tests and (langgraph-based) `graph.build` are unaffected.
"""

from agentic_research.telemetry.events import (
    EventBus,
    EventType,
    JsonlSink,
    RunEmitter,
    TraceEvent,
    load_jsonl,
)
from agentic_research.telemetry.instrument import (
    FaultInjector,
    FaultRule,
    InjectedFault,
    TracedLLMClient,
    TracedToolRegistry,
    instrument_node,
)
from agentic_research.telemetry.runtime import RunResult, RunStatus, TracedGraphRunner

__all__ = [
    "EventBus", "EventType", "JsonlSink", "RunEmitter", "TraceEvent", "load_jsonl",
    "FaultInjector", "FaultRule", "InjectedFault", "TracedLLMClient",
    "TracedToolRegistry", "instrument_node", "RunResult", "RunStatus", "TracedGraphRunner",
]
