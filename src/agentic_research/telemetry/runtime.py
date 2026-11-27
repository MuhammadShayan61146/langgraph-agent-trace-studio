"""`TracedGraphRunner`: runs the project's graph with full telemetry.

Why this exists
---------------
`graph/build.py` registers the project's node functions on a langgraph
`StateGraph`. langgraph is an optional/heavy dependency (and `build.py` has
never been executed in the environment this project was authored in), while
every node function and every routing function is plain Python. This runner
drives **the same node functions, the same routing functions, the same
`with_retry` wrapper and the same edge structure** -- imported, not copied --
so behaviour is identical, while adding:

* a trace event for every node attempt, LLM call, tool call, routing choice,
  state snapshot and memory operation;
* a *real* human-in-the-loop pause: at the approval gate the run checkpoints
  its state and returns `RunStatus.PAUSED`; `resume()` later loads that
  checkpoint (even from a fresh process, with a file-backed store) and
  continues -- the same shape as langgraph's ``interrupt()`` /
  ``Command(resume=...)``;
* optional long-term-memory recall before the run and persistence after it.

Edge structure (kept in sync with build.py by a unit test):
    supervisor --route_from_supervisor--> researcher|analyst|writer|human_approval|END
    human_approval --route_after_supervisor--> writer|END
    researcher|analyst|writer --> supervisor

One documented substitution (same one the project's own
`test_full_graph_simulation.py` makes): `human_approval_node` calls
`langgraph.types.interrupt`, so the runner uses the framework-free
`apply_human_decision` it wraps, applied after a real checkpoint/pause.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from agentic_research.agents.analyst import AnalystAgent
from agentic_research.agents.base import LLMClient
from agentic_research.agents.researcher import ResearcherAgent
from agentic_research.agents.writer import WriterAgent
from agentic_research.config.settings import Settings
from agentic_research.core.contracts import AgentMessage, ErrorRecord, ResearchFinding
from agentic_research.core.enums import AgentRole, MessageRole, TaskStatus
from agentic_research.core.state import GraphState, accumulate_list, validate_transition
from agentic_research.error_recovery import with_retry
from agentic_research.graph.supervisor import (
    END_SENTINEL,
    route_after_supervisor,
    route_from_supervisor,
    supervisor_node,
)
from agentic_research.human.approval import apply_human_decision, build_approval_request
from agentic_research.human.decision import ApprovalDecision
from agentic_research.memory.checkpoint import CheckpointStore, InMemoryCheckpointStore
from agentic_research.memory.long_term import LongTermMemoryStore, save_findings
from agentic_research.memory.serialization import state_to_dict
from agentic_research.telemetry.events import EventBus, EventType, RunEmitter, preview
from agentic_research.telemetry.instrument import (
    FaultInjector,
    TracedLLMClient,
    TracedToolRegistry,
    instrument_node,
)
from agentic_research.tools.registry import ToolRegistry

_LIST_FIELDS = ("messages", "research_findings", "routing_history", "error_log")
_AGENTS = ("researcher", "analyst", "writer")


class RunStatus(str, enum.Enum):
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RunResult:
    """Outcome of `run()` / `resume()`."""

    run_id: str
    thread_id: str
    status: RunStatus
    state: GraphState
    steps: int
    interrupt_payload: dict[str, Any] | None = None
    error: str | None = None


def merge_update(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Apply a node's partial update with the same reducers `GraphState` declares."""
    merged = dict(state)
    for key, value in update.items():
        if key in _LIST_FIELDS:
            merged[key] = accumulate_list(merged.get(key, []), value)
        else:
            merged[key] = value
    return merged


class TracedGraphRunner:
    """Executes the research graph, publishing telemetry on an `EventBus`."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        settings: Settings,
        bus: EventBus,
        *,
        run_id: str | None = None,
        group_id: str | None = None,
        thread_id: str | None = None,
        checkpoint_store: CheckpointStore | None = None,
        memory_store: LongTermMemoryStore | None = None,
        recall_namespace: str | None = None,
        save_namespace: str | None = None,
        faults: FaultInjector | None = None,
        pace: float = 0.0,
        max_steps: int = 40,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        self.thread_id = thread_id or self.run_id
        self.group_id = group_id or self.run_id
        self.emitter = RunEmitter(bus, self.run_id, self.group_id, pace=pace)
        self.checkpoints: CheckpointStore = checkpoint_store or InMemoryCheckpointStore()
        self.memory_store = memory_store
        self.recall_namespace = recall_namespace
        self.save_namespace = save_namespace
        self.max_steps = max_steps
        self.meta = dict(meta or {})
        self.settings = settings
        self._pending_decision: tuple[ApprovalDecision, str | None] | None = None
        self._step = 0

        traced_llm = TracedLLMClient(llm_client, self.emitter, faults)
        traced_tools = TracedToolRegistry.from_registry(tool_registry, self.emitter, faults)
        researcher = ResearcherAgent(llm_client=traced_llm, tool_registry=traced_tools)
        analyst = AnalystAgent(llm_client=traced_llm, tool_registry=traced_tools)
        writer = WriterAgent(llm_client=traced_llm, tool_registry=traced_tools)

        def human_gate_resume(state: GraphState) -> dict[str, Any]:
            assert self._pending_decision is not None
            decision, feedback = self._pending_decision
            return apply_human_decision(state, decision, feedback)

        raw: dict[str, Callable[[GraphState], dict[str, Any]]] = {
            "supervisor": supervisor_node,
            "researcher": researcher.run,
            "analyst": analyst.run,
            "writer": writer.run,
            "human_approval": human_gate_resume,
        }
        # with_retry(instrument_node(fn)): every attempt is traced, and
        # retry/exhaustion semantics are exactly the project's own.
        self.nodes = {
            name: with_retry(
                instrument_node(fn, name, self.emitter, faults=faults,
                                max_retries=settings.max_retries),
                max_retries=settings.max_retries,
                node_name=name,
            )
            for name, fn in raw.items()
        }

    # ------------------------------------------------------------------ API

    def run(self, initial_state: GraphState) -> RunResult:
        """Start a fresh run from ``initial_state``."""
        state: dict[str, Any] = dict(initial_state)
        self.emitter.emit(
            EventType.RUN_STARTED,
            query=state["original_query"],
            thread_id=self.thread_id,
            requires_human_approval=state["requires_human_approval"],
            max_retries=self.settings.max_retries,
            meta=self.meta,
        )
        state = self._recall(state)
        self._snapshot(state, "__start__")
        return self._loop(state, "supervisor")

    def resume(
        self, decision: ApprovalDecision | str, feedback: str | None = None
    ) -> RunResult:
        """Continue a paused run from its checkpoint with a human decision."""
        decision = ApprovalDecision(decision)
        loaded = self.checkpoints.load(self.thread_id)
        if loaded is None:
            raise LookupError(f"No checkpoint found for thread {self.thread_id!r}.")
        state: dict[str, Any] = dict(loaded)
        if state["task_status"] != TaskStatus.AWAITING_HUMAN_APPROVAL:
            raise ValueError(
                f"Thread {self.thread_id!r} is not awaiting approval "
                f"(status={state['task_status']!r})."
            )
        self.emitter.emit(EventType.MEMORY_READ, node="human_approval", source="checkpoint",
                          thread_id=self.thread_id, summary="Loaded paused state from checkpoint")
        self.emitter.emit(EventType.RUN_RESUMED, thread_id=self.thread_id)
        self.emitter.emit(EventType.HUMAN_DECISION, node="human_approval",
                          decision=decision.value, feedback=feedback)
        self._pending_decision = (decision, feedback)
        return self._loop(state, "human_approval")

    # ------------------------------------------------------------- internals

    def _loop(self, state: dict[str, Any], node: str) -> RunResult:
        while True:
            if self._step >= self.max_steps:
                return self._fail(state, node, f"Run exceeded max_steps={self.max_steps}.")

            if node == "human_approval" and self._pending_decision is None:
                return self._pause(state)

            self._step += 1
            try:
                update = self.nodes[node](state)  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001 - graceful, reported failure
                return self._fail(state, node, f"{type(exc).__name__}: {exc}")
            if node == "human_approval":
                self._pending_decision = None

            retries = update.get("retry_count", 0)
            if retries:
                self.emitter.emit(
                    EventType.RECOVERY, node=node, retries=retries,
                    message=f"{node} recovered after {retries} failed attempt(s).",
                )
            state = merge_update(state, update)
            self._snapshot(state, node)
            self._checkpoint(state, node)

            next_node, kind, reason = self._route(node, update, state)
            self.emitter.emit(
                EventType.ROUTE, node=node, source=node,
                target=END_SENTINEL if next_node == END_SENTINEL else next_node,
                kind=kind, reason=reason,
                task_status=_val(state["task_status"]),
            )
            if next_node == END_SENTINEL:
                return self._finish(state)
            node = next_node

    def _route(self, node: str, update: dict[str, Any], state: dict[str, Any]) -> tuple[str, str, str]:
        """Same routing functions as build.py, plus a human-readable 'why'."""
        if node in _AGENTS:
            return "supervisor", "return", f"{node} finished; control returns to the supervisor."
        if node == "supervisor":
            nxt = route_from_supervisor(state)  # type: ignore[arg-type]
            if nxt == "human_approval":
                return nxt, "gate", "Draft report is ready and human approval is required."
            if nxt == END_SENTINEL:
                if state["task_status"] == TaskStatus.COMPLETED:
                    return nxt, "end", "Report ready and approval not required; auto-approved."
                return nxt, "end", "Nothing further to route; ending."
            history = update.get("routing_history") or []
            reason = history[-1].reason if history else f"Task status is {_val(state['task_status'])}."
            return nxt, "dispatch", reason
        # human_approval
        nxt = route_after_supervisor(state)  # type: ignore[arg-type]
        if nxt == "writer":
            return nxt, "revise", "Reviewer requested a revision; Writer runs again with the feedback."
        status = _val(state["task_status"])
        return nxt, "end", f"Human decision recorded ({status}); ending."

    def _recall(self, state: dict[str, Any]) -> dict[str, Any]:
        if not (self.memory_store and self.recall_namespace):
            return state
        records = self.memory_store.search(self.recall_namespace)
        self.emitter.emit(
            EventType.MEMORY_READ, node="supervisor", source="long_term",
            namespace=self.recall_namespace, count=len(records),
            keys=[r.key for r in records],
            summary=f"Recalled {len(records)} record(s) from long-term memory.",
        )
        findings: list[ResearchFinding] = []
        lines = []
        for rec in records:
            v = rec.value
            if "source" in v and "content" in v:
                findings.append(ResearchFinding(
                    source=v["source"], content=v["content"],
                    collected_by=AgentRole(v.get("collected_by", "researcher")),
                    confidence=float(v.get("confidence", 0.7)),
                ))
                lines.append(f"- {v['content']} ({v['source']})")
        if not findings:
            return state
        note = AgentMessage(
            role=MessageRole.SYSTEM,
            content="Recalled from long-term memory (earlier run):\n" + "\n".join(lines),
            metadata={"recalled": len(findings), "namespace": self.recall_namespace},
        )
        return merge_update(state, {"research_findings": findings, "messages": [note]})

    def _checkpoint(self, state: dict[str, Any], node: str) -> None:
        self.checkpoints.save(self.thread_id, state)  # type: ignore[arg-type]
        self.emitter.emit(
            EventType.MEMORY_CHECKPOINT, node=node, thread_id=self.thread_id,
            step=self._step, task_status=_val(state["task_status"]),
        )

    def _snapshot(self, state: dict[str, Any], node: str) -> None:
        snap = state_to_dict(state)  # type: ignore[arg-type]
        self.emitter.emit(
            EventType.STATE_SNAPSHOT, node=node, step=self._step,
            task_status=_val(state["task_status"]),
            counts={
                "messages": len(state["messages"]),
                "findings": len(state["research_findings"]),
                "routes": len(state["routing_history"]),
                "errors": len(state["error_log"]),
            },
            state=snap,
        )

    def _pause(self, state: dict[str, Any]) -> RunResult:
        payload = build_approval_request(state)  # type: ignore[arg-type]
        self._checkpoint(state, "human_approval")
        self.emitter.emit(EventType.HUMAN_INTERRUPT, node="human_approval", payload=payload,
                          thread_id=self.thread_id)
        self.emitter.emit(EventType.RUN_PAUSED, node="human_approval", thread_id=self.thread_id,
                          reason="Awaiting human approval")
        return RunResult(self.run_id, self.thread_id, RunStatus.PAUSED, state,  # type: ignore[arg-type]
                         self._step, interrupt_payload=payload)

    def _finish(self, state: dict[str, Any]) -> RunResult:
        status = state["task_status"]
        if status == TaskStatus.COMPLETED:
            self._persist_memory(state)
            self.emitter.emit(
                EventType.RUN_COMPLETED, final_report=state["final_report"],
                human_approval_status=state["human_approval_status"], steps=self._step,
            )
            return RunResult(self.run_id, self.thread_id, RunStatus.COMPLETED, state, self._step)  # type: ignore[arg-type]
        message = next((e.message for e in reversed(state["error_log"])), "Run ended without completing.")
        self.emitter.emit(EventType.RUN_FAILED, error=message, task_status=_val(status),
                          steps=self._step, final_report=state.get("final_report"))
        return RunResult(self.run_id, self.thread_id, RunStatus.FAILED, state, self._step, error=message)  # type: ignore[arg-type]

    def _fail(self, state: dict[str, Any], node: str, message: str) -> RunResult:
        """Graceful terminal failure: record it, mark FAILED if legal, report."""
        record = ErrorRecord(node_name=node, error_type="RunAborted", message=message, recoverable=False)
        update: dict[str, Any] = {"error_log": [record]}
        try:
            validate_transition(state["task_status"], TaskStatus.FAILED)
            update["task_status"] = TaskStatus.FAILED
        except Exception:  # noqa: BLE001 - keep status if FAILED is not legal from here
            pass
        state = merge_update(state, update)
        self._snapshot(state, node)
        self._checkpoint(state, node)
        self.emitter.emit(EventType.RUN_FAILED, node=node, error=preview(message, 500),
                          task_status=_val(state["task_status"]), steps=self._step,
                          final_report=state.get("final_report"))
        return RunResult(self.run_id, self.thread_id, RunStatus.FAILED, state,  # type: ignore[arg-type]
                         self._step, error=message)

    def _persist_memory(self, state: dict[str, Any]) -> None:
        if not (self.memory_store and self.save_namespace):
            return
        new = [f for f in state["research_findings"] if f.collected_by == AgentRole.RESEARCHER]
        save_findings(self.memory_store, self.save_namespace, new)
        self.memory_store.put(
            self.save_namespace, "final_report",
            {"query": state["original_query"], "report": state["final_report"]},
        )
        self.emitter.emit(
            EventType.MEMORY_WRITE, node="writer", source="long_term",
            namespace=self.save_namespace, count=len(new) + 1,
            summary=f"Saved {len(new)} finding(s) and the final report to long-term memory.",
        )


def _val(x: Any) -> Any:
    return getattr(x, "value", x)
