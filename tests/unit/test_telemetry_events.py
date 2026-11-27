"""Unit tests for the telemetry event bus, emitter and JSONL sink."""

from __future__ import annotations

import json

from agentic_research.core.enums import AgentRole, TaskStatus
from agentic_research.telemetry.events import (
    EventBus, EventType, JsonlSink, RunEmitter, jsonable, load_jsonl, preview,
)


def test_jsonable_handles_enums_dataclasses_and_nesting() -> None:
    from agentic_research.core.contracts import RoutingDecision

    rd = RoutingDecision(from_agent=AgentRole.SUPERVISOR, to_agent=AgentRole.WRITER, reason="r")
    out = jsonable({"s": TaskStatus.WRITING, "rd": rd, "t": (1, 2), "x": object})
    json.dumps(out)  # must be serializable
    assert out["s"] == "writing" and out["rd"]["to_agent"] == "writer" and out["t"] == [1, 2]


def test_preview_truncates_single_value() -> None:
    assert preview("abc", 10) == "abc"
    assert len(preview("x" * 500, 20)) == 20 and preview("x" * 500, 20).endswith("…")


def test_bus_assigns_monotonic_seq_and_filters_history() -> None:
    bus = EventBus()
    a = RunEmitter(bus, "r1", "g1")
    b = RunEmitter(bus, "r2", "g2")
    a.emit(EventType.RUN_STARTED, query="q")
    b.emit(EventType.RUN_STARTED, query="q2")
    a.emit(EventType.NODE_STARTED, node="supervisor")
    assert [e.seq for e in bus.history()] == [1, 2, 3]
    assert [e.type for e in bus.history(run_id="r1")] == ["run.started", "node.started"]
    assert [e.run_id for e in bus.history(group_id="g2")] == ["r2"]
    assert [e.seq for e in bus.history(since_seq=2)] == [3]


def test_subscribe_unsubscribe_and_failing_subscriber_is_isolated() -> None:
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(lambda e: (_ for _ in ()).throw(RuntimeError("boom")))
    unsub = bus.subscribe(lambda e: seen.append(e.type))
    em = RunEmitter(bus, "r")
    em.emit(EventType.RUN_STARTED)  # must not raise despite the bad subscriber
    unsub()
    em.emit(EventType.RUN_COMPLETED)
    assert seen == ["run.started"]


def test_event_data_is_json_safe_and_group_defaults_to_run() -> None:
    bus = EventBus()
    ev = RunEmitter(bus, "r").emit("x", status=TaskStatus.PENDING)
    assert ev.group_id == "r" and ev.data == {"status": "pending"}
    json.dumps(ev.to_dict())


def test_jsonl_sink_roundtrip(tmp_path) -> None:
    bus = EventBus()
    bus.subscribe(JsonlSink(tmp_path))
    em = RunEmitter(bus, "run-a")
    em.emit(EventType.RUN_STARTED, query="hello")
    em.emit(EventType.RUN_COMPLETED, final_report="done")
    loaded = load_jsonl(tmp_path / "run-a.jsonl")
    assert [e["type"] for e in loaded] == ["run.started", "run.completed"]
    assert loaded[1]["data"]["final_report"] == "done"
