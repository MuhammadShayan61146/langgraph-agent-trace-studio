"""Integration tests for `TracedGraphRunner`: parity, pause/resume, failure handling."""

from __future__ import annotations

import pytest

from agentic_research.agents.base import LLMResponse
from agentic_research.config.settings import Settings
from agentic_research.core.enums import TaskStatus
from agentic_research.core.state import create_initial_state
from agentic_research.human.decision import ApprovalDecision
from agentic_research.memory.checkpoint import FileCheckpointStore
from agentic_research.telemetry import EventBus, FaultInjector, FaultRule, RunStatus, TracedGraphRunner
from agentic_research.tools.calculator import CalculatorTool
from agentic_research.tools.registry import ToolRegistry
from agentic_research.tools.search import InMemorySearchBackend, WebSearchTool

from tests.integration.test_full_graph_simulation import _build_wrapped_nodes, _run_graph


class RoleLLM:
    """Minimal stateless LLM: one scripted line per role."""

    def complete(self, *, system, messages, tools=None):
        for role in ("Researcher", "Analyst"):
            if f"{role} agent" in system:
                return LLMResponse(content=f"{role} output.")
        revs = [m["content"] for m in messages if m["content"].startswith("Revision requested")]
        return LLMResponse(content="Draft report." + (f" Revised: {revs[-1]}" if revs else ""))


def _registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(CalculatorTool())
    r.register(WebSearchTool(InMemorySearchBackend()))
    return r


def _runner(bus=None, **kw):
    return TracedGraphRunner(RoleLLM(), _registry(), Settings(max_retries=kw.pop("retries", 2)),
                             bus or EventBus(), **kw)


def _types(bus):
    return [e.type for e in bus.history()]


def test_parity_with_project_simulation_without_approval() -> None:
    """Same nodes/routing as the project's own framework-free simulation."""
    ref = _run_graph(create_initial_state("query"), _build_wrapped_nodes(Settings(max_retries=2), ["Final report."], []))
    runner = TracedGraphRunner(
        type("L", (), {"complete": lambda self, **k: LLMResponse(content="Final report." if "Writer agent" in k["system"] else "x")})(),
        _registry(), Settings(max_retries=2), EventBus())
    res = runner.run(create_initial_state("query"))
    assert res.status == RunStatus.COMPLETED
    assert res.state["task_status"] == ref["task_status"] == TaskStatus.COMPLETED
    assert res.state["final_report"] == ref["final_report"]
    assert [d.to_agent for d in res.state["routing_history"]] == [d.to_agent for d in ref["routing_history"]]


def test_event_stream_covers_the_whole_run_in_order() -> None:
    bus = EventBus()
    res = _runner(bus, run_id="r").run(create_initial_state("q"))
    types = _types(bus)
    assert types[0] == "run.started" and types[-1] == "run.completed" and res.steps == 7
    assert types.count("node.started") == types.count("node.finished") == 7
    assert types.count("route.decision") == 7 and types.count("memory.checkpoint") == 7
    routes = [e.data for e in bus.history() if e.type == "route.decision"]
    assert [r["target"] for r in routes] == ["researcher", "supervisor", "analyst", "supervisor", "writer", "supervisor", "__end__"]
    assert routes[0]["kind"] == "dispatch" and routes[1]["kind"] == "return" and routes[-1]["kind"] == "end"
    assert "has not yet contributed" in routes[0]["reason"]
    seqs = [e.seq for e in bus.history()]
    assert seqs == sorted(seqs)


def test_human_gate_pauses_with_checkpoint_then_resume_approves() -> None:
    bus = EventBus()
    r = _runner(bus)
    paused = r.run(create_initial_state("q", requires_human_approval=True))
    assert paused.status == RunStatus.PAUSED and paused.interrupt_payload["draft_report"] == "Draft report."
    assert _types(bus)[-3:] == ["memory.checkpoint", "human.interrupt", "run.paused"]
    assert r.checkpoints.load(r.thread_id)["task_status"] == TaskStatus.AWAITING_HUMAN_APPROVAL
    done = r.resume(ApprovalDecision.APPROVED)
    assert done.status == RunStatus.COMPLETED and done.state["human_approval_status"] == "approved"
    assert "run.resumed" in _types(bus) and "human.decision" in _types(bus)


def test_revision_loop_pauses_twice_and_feedback_reaches_writer() -> None:
    r = _runner()
    r.run(create_initial_state("q", requires_human_approval=True))
    again = r.resume("revision_requested", "add numbers")
    assert again.status == RunStatus.PAUSED and "Revised: " in again.state["final_report"]
    assert "add numbers" in again.state["final_report"] and again.interrupt_payload["revision_cycle"] == 2
    routes = [e.data["kind"] for e in r.emitter.bus.history() if e.type == "route.decision"]
    assert "revise" in routes
    assert r.resume("approved").status == RunStatus.COMPLETED


def test_reject_ends_failed_with_error_log() -> None:
    r = _runner()
    r.run(create_initial_state("q", requires_human_approval=True))
    res = r.resume("rejected", "not good")
    assert res.status == RunStatus.FAILED and res.state["task_status"] == TaskStatus.FAILED
    assert res.state["error_log"][-1].error_type == "ReportRejected"


def test_resume_works_from_a_fresh_runner_via_file_checkpoint(tmp_path) -> None:
    store = FileCheckpointStore(str(tmp_path))
    first = _runner(thread_id="T", checkpoint_store=store)
    assert first.run(create_initial_state("q", requires_human_approval=True)).status == RunStatus.PAUSED
    second = _runner(thread_id="T", checkpoint_store=FileCheckpointStore(str(tmp_path)))  # "new process"
    res = second.resume("approved")
    assert res.status == RunStatus.COMPLETED and res.state["final_report"] == "Draft report."


def test_resume_without_checkpoint_or_when_not_paused_is_rejected() -> None:
    r = _runner()
    with pytest.raises(LookupError):
        r.resume("approved")
    r2 = _runner()
    r2.run(create_initial_state("q"))  # completes; checkpoint is COMPLETED, not awaiting
    with pytest.raises(ValueError):
        r2.resume("approved")
    with pytest.raises(ValueError):
        _runner().resume("not-a-decision")


def test_transient_node_faults_recover_and_emit_recovery() -> None:
    bus = EventBus()
    r = _runner(bus, retries=3, faults=FaultInjector([FaultRule("node", "analyst", 2)]))
    res = r.run(create_initial_state("q"))
    assert res.status == RunStatus.COMPLETED and res.state["retry_count"] == 2
    rec = [e for e in bus.history() if e.type == "recovery.succeeded"]
    assert len(rec) == 1 and rec[0].data["retries"] == 2
    assert [e.data["attempt"] for e in bus.history() if e.type == "node.started" and e.node == "analyst"] == [1, 2, 3]
    assert sum(1 for e in res.state["error_log"] if e.node_name == "analyst") == 2


def test_exhausted_retries_fail_gracefully_with_events() -> None:
    bus = EventBus()
    r = _runner(bus, retries=1, faults=FaultInjector([FaultRule("node", "researcher", 99)]))
    res = r.run(create_initial_state("q"))
    assert res.status == RunStatus.FAILED and "RetryLimitExceededError" in res.error
    assert res.state["task_status"] == TaskStatus.FAILED
    assert _types(bus)[-1] == "run.failed" and res.state["error_log"][-1].error_type == "RunAborted"


def test_max_steps_guard_stops_runaway_runs() -> None:
    res = _runner(max_steps=2).run(create_initial_state("q"))
    assert res.status == RunStatus.FAILED and "max_steps" in res.error
