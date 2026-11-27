"""Trace events and a small thread-safe event bus.

Purpose
-------
Every observable thing the system does -- a node starting, an LLM call, a
tool result, a routing decision, a checkpoint write, a human interrupt --
is published as one `TraceEvent` on an `EventBus`. The dashboard, the JSONL
sink, the dossier builder and the tests are all just subscribers/consumers
of that one stream, so nothing in the core package knows a UI exists.

Design decisions
-----------------
* Events are plain, JSON-safe dicts under the hood (`TraceEvent.to_dict`),
  so a run can be saved as JSONL and replayed later without this package.
* The bus keeps a bounded in-memory history so late SSE subscribers can
  replay what they missed.
* Subscriber exceptions are swallowed: telemetry must never break a run.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from collections import deque
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque


class EventType:
    """String constants for every event type the system emits."""

    RUN_STARTED = "run.started"
    RUN_PAUSED = "run.paused"
    RUN_RESUMED = "run.resumed"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"

    NODE_STARTED = "node.started"
    NODE_FINISHED = "node.finished"
    NODE_ERROR = "node.error"

    ROUTE = "route.decision"

    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"

    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"

    STATE_SNAPSHOT = "state.snapshot"

    MEMORY_CHECKPOINT = "memory.checkpoint"
    MEMORY_WRITE = "memory.write"
    MEMORY_READ = "memory.read"

    FAULT_INJECTED = "fault.injected"
    RECOVERY = "recovery.succeeded"

    HUMAN_INTERRUPT = "human.interrupt"
    HUMAN_DECISION = "human.decision"


#: Event types after which a run is no longer producing events until resumed.
TERMINAL_TYPES = frozenset({EventType.RUN_COMPLETED, EventType.RUN_FAILED})


def jsonable(value: Any) -> Any:
    """Recursively convert ``value`` into something `json.dumps` accepts."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    return str(value)


def preview(text: Any, limit: int = 240) -> str:
    """Return ``text`` as a single-line string truncated to ``limit`` chars."""
    s = str(text)
    return s if len(s) <= limit else s[: limit - 1] + "…"


@dataclasses.dataclass(frozen=True)
class TraceEvent:
    """One immutable, JSON-serializable trace event."""

    seq: int
    ts: float
    t_ms: int
    run_id: str
    group_id: str
    type: str
    node: str | None
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


Subscriber = Callable[[TraceEvent], None]


class EventBus:
    """Thread-safe publish/subscribe hub with bounded replay history."""

    def __init__(self, max_history: int = 50_000) -> None:
        self._lock = threading.RLock()
        self._seq = 0
        self._history: Deque[TraceEvent] = deque(maxlen=max_history)
        self._subscribers: list[Subscriber] = []

    def subscribe(self, callback: Subscriber) -> Callable[[], None]:
        """Register ``callback``; returns a function that unsubscribes it."""
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def publish(
        self,
        *,
        run_id: str,
        group_id: str,
        type: str,
        node: str | None,
        data: dict[str, Any],
        t_ms: int,
    ) -> TraceEvent:
        with self._lock:
            self._seq += 1
            event = TraceEvent(
                seq=self._seq,
                ts=time.time(),
                t_ms=t_ms,
                run_id=run_id,
                group_id=group_id,
                type=type,
                node=node,
                data=jsonable(data),
            )
            self._history.append(event)
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception:  # noqa: BLE001 - telemetry must never break a run
                pass
        return event

    def history(
        self,
        *,
        run_id: str | None = None,
        group_id: str | None = None,
        since_seq: int = 0,
    ) -> list[TraceEvent]:
        with self._lock:
            return [
                e
                for e in self._history
                if e.seq > since_seq
                and (run_id is None or e.run_id == run_id)
                and (group_id is None or e.group_id == group_id)
            ]


class RunEmitter:
    """Binds a bus to one run so emit sites stay one-liners.

    ``pace`` (seconds) sleeps briefly after "visible" events so a live
    dashboard can show the choreography at a human-readable speed. It is
    ``0`` by default (tests and CLI run at full speed).
    """

    _PACED = frozenset(
        {
            EventType.NODE_STARTED,
            EventType.LLM_RESPONSE,
            EventType.TOOL_RESULT,
            EventType.ROUTE,
            EventType.HUMAN_INTERRUPT,
            EventType.MEMORY_READ,
            EventType.MEMORY_WRITE,
            EventType.FAULT_INJECTED,
            EventType.NODE_ERROR,
            EventType.RECOVERY,
        }
    )

    def __init__(
        self, bus: EventBus, run_id: str, group_id: str | None = None, pace: float = 0.0
    ) -> None:
        self.bus = bus
        self.run_id = run_id
        self.group_id = group_id or run_id
        self.pace = pace
        self._t0 = time.perf_counter()

    def emit(self, type: str, node: str | None = None, **data: Any) -> TraceEvent:
        t_ms = int((time.perf_counter() - self._t0) * 1000)
        event = self.bus.publish(
            run_id=self.run_id,
            group_id=self.group_id,
            type=type,
            node=node,
            data=data,
            t_ms=t_ms,
        )
        if self.pace > 0 and type in self._PACED:
            time.sleep(self.pace)
        return event


class JsonlSink:
    """Subscriber that appends every event to ``<directory>/<run_id>.jsonl``."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def __call__(self, event: TraceEvent) -> None:
        line = json.dumps(event.to_dict(), ensure_ascii=False)
        with self._lock:
            with open(self.directory / f"{event.run_id}.jsonl", "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a saved run's events (as dicts) from a JSONL file."""
    events = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
