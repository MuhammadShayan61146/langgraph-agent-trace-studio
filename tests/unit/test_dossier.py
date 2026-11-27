"""Unit tests for dossier building and rendering (from synthetic events)."""

from __future__ import annotations

import json

from agentic_research.telemetry.dossier import (
    build_dossier, render_html, render_json, render_markdown, summarize_event,
)
from agentic_research.telemetry.events import EventBus, EventType as T, RunEmitter


def _events() -> list[dict]:
    bus = EventBus()
    em = RunEmitter(bus, "r1", "g1")
    em.emit(T.RUN_STARTED, query="Q <script>", thread_id="t", meta={"label": "L"})
    em.emit(T.NODE_STARTED, node="researcher", attempt=1)
    em.emit(T.NODE_ERROR, node="researcher", attempt=1, error_type="E", message="bad", will_retry=True)
    em.emit(T.TOOL_CALL, node="researcher", tool="web_search", arguments={"query": "x"})
    em.emit(T.TOOL_RESULT, node="researcher", tool="web_search", success=True, output=[1], error=None, duration_ms=1)
    em.emit(T.LLM_REQUEST, node="researcher", round=1, message_count=1)
    em.emit(T.MEMORY_WRITE, node="writer", source="long_term", summary="Saved 2")
    em.emit(T.ROUTE, node="supervisor", source="supervisor", target="researcher", kind="dispatch", reason="why")
    em.emit(T.HUMAN_DECISION, node="human_approval", decision="approved", feedback=None)
    em.emit(T.RUN_COMPLETED, final_report="# Report", human_approval_status="approved", steps=4)
    return [e.to_dict() for e in bus.history()]


def test_build_dossier_collects_each_audit_section() -> None:
    d = build_dossier(_events())
    assert d["status"] == "completed" and d["final_report"] == "# Report" and d["meta"] == {"label": "L"}
    assert d["routes"][0]["reason"] == "why" and d["llm_calls"] == 1
    assert d["tool_audit"][0]["tool"] == "web_search" and d["tool_audit"][0]["success"] is True
    assert d["memory_audit"][0]["op"] == "write"
    assert any(f["type"] == "node.error" for f in d["faults"])  # error also reaches fault list
    assert d["human"][0]["decision"] == "approved"
    assert len(d["events"]) == 10 and "events" not in build_dossier(_events(), include_events=False)


def test_failed_run_dossier_status_and_error() -> None:
    bus = EventBus()
    em = RunEmitter(bus, "r")
    em.emit(T.RUN_STARTED, query="q")
    em.emit(T.RUN_FAILED, error="kaput", steps=2)
    d = build_dossier([e.to_dict() for e in bus.history()])
    assert d["status"] == "failed" and d["error"] == "kaput"
    assert "kaput" in render_markdown(d) and "No report was produced" in render_markdown(d)


def test_renderers_produce_expected_content_and_escape_html() -> None:
    d = build_dossier(_events())
    assert json.loads(render_json(d))["run_id"] == "r1"
    md = render_markdown(d)
    assert "## Final deliverable" in md and "web_search" in md and "| 1 | supervisor | researcher |" in md
    page = render_html(d)
    assert page.startswith("<!doctype html>") and "<script>" not in page and "&lt;script&gt;" in page


def test_summarize_event_falls_back_to_type() -> None:
    assert summarize_event({"type": "weird.type", "data": {}, "node": None}) == "weird.type"
