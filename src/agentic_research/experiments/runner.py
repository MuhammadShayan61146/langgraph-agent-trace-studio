"""`ExperimentRunner`: turns a `Scenario` into traced runs.

A scenario run is a *group* of one or more runs (phases) that share a
``group_id``. Phases execute sequentially; if a phase pauses at the human
gate, the group pauses with it and `resume()` continues it and then runs the
remaining phases. `run_to_completion(decider=...)` drives that loop
synchronously (CLI/tests); the dashboard server calls `start()`/`resume()`
from worker threads.
"""

from __future__ import annotations

import dataclasses
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from agentic_research.config.settings import Settings
from agentic_research.core.contracts import AgentMessage
from agentic_research.core.enums import MessageRole
from agentic_research.core.state import create_initial_state
from agentic_research.experiments.corpora import CorpusSearchBackend
from agentic_research.experiments.llm import ScenarioLLM, split_query
from agentic_research.experiments.scenarios import Phase, Scenario, get_scenario
from agentic_research.human.decision import ApprovalDecision
from agentic_research.memory.checkpoint import (
    CheckpointStore,
    FileCheckpointStore,
    InMemoryCheckpointStore,
)
from agentic_research.memory.long_term import (
    FileLongTermMemoryStore,
    InMemoryLongTermMemoryStore,
    LongTermMemoryStore,
)
from agentic_research.telemetry.events import EventBus
from agentic_research.telemetry.instrument import FaultInjector
from agentic_research.telemetry.runtime import RunResult, RunStatus, TracedGraphRunner
from agentic_research.tools.calculator import CalculatorTool
from agentic_research.tools.registry import ToolRegistry
from agentic_research.tools.search import WebSearchTool

#: Decider signature: (interrupt payload) -> (decision, feedback)
Decider = Callable[[dict], "tuple[ApprovalDecision | str, str | None]"]


@dataclass
class RunSummary:
    run_id: str
    label: str
    status: str
    query: str


@dataclass
class GroupSession:
    group_id: str
    scenario: Scenario
    variant: str | None
    phases: tuple[Phase, ...]
    query_override: str | None
    status: str = "running"  # running | paused | completed | failed
    phase_index: int = 0
    runs: list[RunSummary] = field(default_factory=list)
    results: list[RunResult] = field(default_factory=list)
    current: TracedGraphRunner | None = None
    interrupt_payload: dict | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id, "scenario": self.scenario.id, "variant": self.variant,
            "status": self.status, "phase_index": self.phase_index,
            "runs": [dataclasses.asdict(r) for r in self.runs],
            "interrupt_payload": self.interrupt_payload,
        }


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool(backend=CorpusSearchBackend()))
    return registry


class ExperimentRunner:
    """Runs scenarios on a shared bus with shared (optionally file-backed) memory."""

    def __init__(
        self,
        bus: EventBus,
        *,
        data_dir: str | Path | None = None,
        pace: float = 0.0,
        default_max_retries: int = 3,
    ) -> None:
        self.bus = bus
        self.pace = pace
        self.default_max_retries = default_max_retries
        self.checkpoints: CheckpointStore
        self.memory: LongTermMemoryStore
        if data_dir is not None:
            base = Path(data_dir)
            self.checkpoints = FileCheckpointStore(str(base / "checkpoints"))
            self.memory = FileLongTermMemoryStore(str(base / "memory"))
        else:
            self.checkpoints = InMemoryCheckpointStore()
            self.memory = InMemoryLongTermMemoryStore()
        self.groups: dict[str, GroupSession] = {}

    # ------------------------------------------------------------------ API

    def create(
        self, scenario_id: str, *, variant: str | None = None, query: str | None = None,
        group_id: str | None = None,
    ) -> GroupSession:
        """Register a group without running it (so callers can hand out its id first)."""
        scenario = get_scenario(scenario_id)
        phases = scenario.phases_for(variant)
        group = GroupSession(
            group_id=group_id or f"grp-{uuid.uuid4().hex[:8]}",
            scenario=scenario, variant=variant, phases=phases,
            query_override=(query.strip() if query and scenario.editable_prompt else None) or None,
        )
        self.groups[group.group_id] = group
        return group

    def execute(self, group: GroupSession) -> GroupSession:
        """Run a created group until it pauses, fails or completes."""
        with group.lock:
            self._advance(group)
        return group

    def start(
        self, scenario_id: str, *, variant: str | None = None, query: str | None = None,
        group_id: str | None = None,
    ) -> GroupSession:
        return self.execute(self.create(scenario_id, variant=variant, query=query, group_id=group_id))

    def resume(self, group_id: str, decision: ApprovalDecision | str, feedback: str | None = None) -> GroupSession:
        group = self.groups[group_id]
        with group.lock:
            if group.status != "paused" or group.current is None:
                raise ValueError(f"Group {group_id!r} is not paused (status={group.status}).")
            group.status = "running"
            group.interrupt_payload = None
            result = group.current.resume(decision, feedback)
            self._after_phase(group, result)
            if group.status == "running":
                group.phase_index += 1
                self._advance(group)
        return group

    def run_to_completion(
        self, scenario_id: str, *, variant: str | None = None, query: str | None = None,
        decider: Decider | None = None,
    ) -> GroupSession:
        """Run a scenario synchronously, answering human gates via ``decider``.

        Without a decider, a scenario that reaches a gate is auto-approved.
        """
        group = self.start(scenario_id, variant=variant, query=query)
        while group.status == "paused":
            payload = group.interrupt_payload or {}
            decision, feedback = decider(payload) if decider else (ApprovalDecision.APPROVED, None)
            group = self.resume(group.group_id, decision, feedback)
        return group

    # ------------------------------------------------------------- internals

    def _advance(self, group: GroupSession) -> None:
        """Run phases from ``group.phase_index`` until pause, failure or the end."""
        while group.phase_index < len(group.phases):
            phase = group.phases[group.phase_index]
            runner, state = self._prepare(group, phase)
            group.current = runner
            group.runs.append(RunSummary(runner.run_id, phase.label, "running", state["original_query"]))
            result = runner.run(state)
            self._after_phase(group, result)
            if group.status != "running":
                return
            group.phase_index += 1
        group.status = "completed"
        group.current = None

    def _after_phase(self, group: GroupSession, result: RunResult) -> None:
        group.runs[-1].status = result.status.value
        if result.status == RunStatus.PAUSED:
            group.status = "paused"
            group.interrupt_payload = result.interrupt_payload
            return
        group.results.append(result)
        if result.status == RunStatus.FAILED:
            group.status = "failed"
            group.current = None

    def _prepare(self, group: GroupSession, phase: Phase) -> tuple[TracedGraphRunner, dict]:
        query = phase.query
        if group.query_override and group.phase_index == 0:
            _, lens = split_query(phase.query)
            query = group.query_override + (f"\nOutput lens: {lens}" if lens else "")
        run_id = f"{group.scenario.id}-{group.phase_index + 1}-{uuid.uuid4().hex[:6]}"
        if phase.reset_memory and phase.save_namespace:
            for rec in self.memory.search(phase.save_namespace):
                self.memory.delete(rec.namespace, rec.key)

        previous_report = None
        if phase.recall_previous_report and group.results:
            prev = group.results[-1]
            prev_state = self.checkpoints.load(prev.thread_id)
            previous_report = prev_state["final_report"] if prev_state else None

        settings = Settings(max_retries=phase.max_retries if phase.max_retries is not None
                            else self.default_max_retries)
        runner = TracedGraphRunner(
            ScenarioLLM(), build_registry(), settings, self.bus,
            run_id=run_id, group_id=group.group_id, thread_id=f"thread-{run_id}",
            checkpoint_store=self.checkpoints, memory_store=self.memory,
            recall_namespace=phase.recall_namespace, save_namespace=phase.save_namespace,
            faults=FaultInjector(list(phase.faults)) if phase.faults else None,
            pace=self.pace,
            meta={"scenario": group.scenario.id, "scenario_title": group.scenario.title,
                  "label": phase.label, "variant": group.variant or "default",
                  "phase": group.phase_index + 1, "phases": len(group.phases)},
        )
        state = create_initial_state(query, requires_human_approval=phase.requires_human_approval)
        if previous_report:
            state["messages"] = [AgentMessage(
                role=MessageRole.SYSTEM,
                content="Previous run's final report (loaded from checkpoint):\n" + previous_report,
                metadata={"source": "checkpoint"},
            )]
            runner.emitter.emit("memory.read", node="supervisor", source="checkpoint",
                                thread_id=group.results[-1].thread_id,
                                summary="Loaded previous run's final report from its checkpoint.")
        return runner, state
