"""Unit tests for instrumentation wrappers and the fault injector."""

from __future__ import annotations

import pytest

from agentic_research.agents.base import LLMResponse, MockLLMClient
from agentic_research.core.state import create_initial_state
from agentic_research.error_recovery import with_retry
from agentic_research.core.exceptions import RetryLimitExceededError
from agentic_research.telemetry.context import current_node
from agentic_research.telemetry.events import EventBus, RunEmitter
from agentic_research.telemetry.instrument import (
    FaultInjector, FaultRule, InjectedFault, TracedLLMClient, TracedToolRegistry, instrument_node,
)
from agentic_research.tools.calculator import CalculatorTool
from agentic_research.tools.registry import ToolRegistry


def _types(bus: EventBus) -> list[str]:
    return [e.type for e in bus.history()]


def test_instrument_node_success_emits_started_finished() -> None:
    bus = EventBus()
    node = instrument_node(lambda s: {"task_status": "x"}, "n", RunEmitter(bus, "r"))
    assert node(create_initial_state("q")) == {"task_status": "x"}
    assert _types(bus) == ["node.started", "node.finished"]
    assert bus.history()[1].data["update_keys"] == ["task_status"]


def test_every_retry_attempt_is_visible_inside_with_retry() -> None:
    bus = EventBus()
    calls = {"n": 0}

    def flaky(state):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("nope")
        return {"retry_count": 0}

    wrapped = with_retry(instrument_node(flaky, "n", RunEmitter(bus, "r"), max_retries=3),
                         max_retries=3, node_name="n")
    wrapped(create_initial_state("q"))
    errs = [e for e in bus.history() if e.type == "node.error"]
    assert [e.data["attempt"] for e in errs] == [1, 2] and all(e.data["will_retry"] for e in errs)
    started = [e.data["attempt"] for e in bus.history() if e.type == "node.started"]
    assert started == [1, 2, 3]


def test_exhausted_retries_mark_last_error_as_not_retried_and_reset_counter() -> None:
    bus = EventBus()

    def boom(state):
        raise ValueError("x")

    inst = instrument_node(boom, "n", RunEmitter(bus, "r"), max_retries=1)
    wrapped = with_retry(inst, max_retries=1, node_name="n")
    with pytest.raises(RetryLimitExceededError):
        wrapped(create_initial_state("q"))
    errs = [e.data for e in bus.history() if e.type == "node.error"]
    assert [e["will_retry"] for e in errs] == [True, False]
    # counter reset: a later call starts again at attempt 1
    with pytest.raises(RetryLimitExceededError):
        wrapped(create_initial_state("q"))
    assert [e.data["attempt"] for e in bus.history() if e.type == "node.started"][-2:] == [1, 2]


def test_fault_injector_fires_exactly_times_then_clears() -> None:
    bus = EventBus()
    em = RunEmitter(bus, "r")
    fi = FaultInjector([FaultRule("node", "a", times=2, message="m")])
    assert fi.check("node", "a", em) and fi.check("node", "a", em)
    assert fi.check("node", "a", em) is None
    assert fi.check("node", "other", em) is None
    assert [e.data["firing"] for e in bus.history()] == [1, 2]


def test_node_fault_raises_injected_fault() -> None:
    bus = EventBus()
    inst = instrument_node(lambda s: {}, "a", RunEmitter(bus, "r"),
                           faults=FaultInjector([FaultRule("node", "a", 1)]), max_retries=0)
    with pytest.raises(InjectedFault):
        inst(create_initial_state("q"))


def test_traced_tool_registry_traces_success_and_injects_failure() -> None:
    bus = EventBus()
    em = RunEmitter(bus, "r")
    base = ToolRegistry()
    base.register(CalculatorTool())
    reg = TracedToolRegistry.from_registry(base, em, FaultInjector([FaultRule("tool", "calculator", 1, "down")]))
    failed = reg.execute("calculator", expression="1+1")
    ok = reg.execute("calculator", expression="1+1")
    assert (failed.success, failed.error_message) == (False, "down")
    assert ok.success and ok.output == 2
    results = [e.data for e in bus.history() if e.type == "tool.result"]
    assert [r["success"] for r in results] == [False, True]


def test_traced_tool_registry_traces_unknown_tool_then_raises() -> None:
    from agentic_research.core.exceptions import ToolNotFoundError

    bus = EventBus()
    reg = TracedToolRegistry(RunEmitter(bus, "r"))
    with pytest.raises(ToolNotFoundError):
        reg.execute("missing")
    assert bus.history()[-1].data["success"] is False


def test_traced_llm_client_reports_active_node_and_tool_calls() -> None:
    bus = EventBus()
    llm = TracedLLMClient(MockLLMClient([LLMResponse(content="hi")]), RunEmitter(bus, "r"))
    token = current_node.set("writer")
    try:
        llm.complete(system="s", messages=[{"role": "user", "content": "q"}], tools=[{"name": "t"}])
    finally:
        current_node.reset(token)
    req, resp = bus.history()
    assert req.node == resp.node == "writer" and req.data["tools"] == ["t"] and resp.data["content"] == "hi"
