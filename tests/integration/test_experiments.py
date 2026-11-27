"""Integration tests for the Experiments Suite: every scenario, end to end."""

from __future__ import annotations

import pytest

from agentic_research.experiments import SCENARIOS, ExperimentRunner, get_scenario, scenario_catalog
from agentic_research.experiments.corpora import CorpusSearchBackend
from agentic_research.experiments.llm import ScenarioLLM, split_query
from agentic_research.telemetry import EventBus
from agentic_research.telemetry.dossier import build_dossier


def _run(scenario, **kw):
    bus = EventBus()
    runner = ExperimentRunner(bus, **kw.pop("runner_kw", {}))
    return bus, runner, runner.run_to_completion(scenario, **kw)


def _events(bus, run_id):
    return [e.to_dict() for e in bus.history(run_id=run_id)]


def test_catalog_lists_the_five_required_scenarios() -> None:
    ids = {s["id"] for s in scenario_catalog()}
    assert ids == {"contradiction", "fault-drill", "human-checkpoint", "memory-continuity", "output-lens"}
    assert all(s["default_query"] and s["look_for"] for s in scenario_catalog())
    with pytest.raises(KeyError):
        get_scenario("nope")
    with pytest.raises(KeyError):
        get_scenario("contradiction").phases_for("bogus")


@pytest.mark.parametrize("scenario", [s.id for s in SCENARIOS])
def test_every_scenario_completes_and_dossiers_build(scenario) -> None:
    bus, _, group = _run(scenario)
    assert group.status == "completed"
    for run in group.runs:
        d = build_dossier(_events(bus, run.run_id))
        assert d["status"] == "completed" and d["final_report"]


def test_contradiction_reconciles_both_positions_with_calculator_numbers() -> None:
    bus, _, group = _run("contradiction")
    report = group.results[0].state["final_report"]
    for heading in ("Position A", "Position B", "Reconciliation", "Confidence"):
        assert heading in report
    assert "Mean reported effect +" in report or "Mean reported effect -" in report
    calc = [e for e in bus.history() if e.type == "tool.result" and e.data["tool"] == "calculator"]
    assert len(calc) == 2 and all(e.data["success"] for e in calc)


def test_fault_drill_recovers_from_all_three_fault_layers() -> None:
    bus, _, group = _run("fault-drill")
    types = [e.type for e in bus.history()]
    kinds = [e.data["kind"] for e in bus.history() if e.type == "fault.injected"]
    assert sorted(kinds) == ["llm", "node", "node", "tool"]
    assert types.count("recovery.succeeded") == 2  # analyst + writer
    report = group.results[0].state["final_report"]
    assert group.status == "completed" and "Position A" in report
    researcher = next(e for e in bus.history() if e.type == "llm.response" and "recovered after 1 failed search" in e.data["content"])
    assert researcher.node == "researcher"  # graceful degradation via broader query
    queries = [e.data["arguments"]["query"] for e in bus.history() if e.type == "tool.call" and e.data["tool"] == "web_search"]
    assert len(queries[0].split()) > len(queries[1].split())


def test_fault_drill_exhausted_variant_fails_cleanly() -> None:
    bus, _, group = _run("fault-drill", variant="exhausted")
    assert group.status == "failed" and group.runs[0].status == "failed"
    assert [e.type for e in bus.history()][-1] == "run.failed"
    nodes = [e for e in bus.history() if e.type == "node.started" and e.node == "analyst"]
    assert len(nodes) == 3  # max_retries=2 -> 3 attempts


def test_human_checkpoint_pauses_and_revision_changes_the_report() -> None:
    bus = EventBus()
    runner = ExperimentRunner(bus)
    group = runner.start("human-checkpoint")
    assert group.status == "paused" and group.interrupt_payload["draft_report"]
    first_draft = group.interrupt_payload["draft_report"]
    runner.resume(group.group_id, "revision_requested", "include cost detail")
    assert group.status == "paused" and group.interrupt_payload["draft_report"] != first_draft
    assert "include cost detail" in group.interrupt_payload["draft_report"]
    runner.resume(group.group_id, "approved")
    assert group.status == "completed" and group.results[0].state["human_approval_status"] == "approved"
    with pytest.raises(ValueError):
        runner.resume(group.group_id, "approved")  # no longer paused


def test_human_checkpoint_reject_marks_group_failed() -> None:
    _, _, group = _run("human-checkpoint", decider=lambda p: ("rejected", "no"))
    assert group.status == "failed"


def test_memory_continuity_run2_recalls_run1_findings_and_report(tmp_path) -> None:
    bus, runner, group = _run("memory-continuity", runner_kw={"data_dir": tmp_path})
    r1, r2 = group.results
    assert "Continuity" not in r1.state["final_report"]
    report2 = r2.state["final_report"]
    assert "Continuity" in report2 and "3 finding(s) recalled" in report2
    assert "Prior conclusion carried forward" in report2
    types1 = [e.type for e in bus.history(run_id=r1.run_id)]
    assert "memory.write" in types1
    reads = [e.data["source"] for e in bus.history(run_id=r2.run_id) if e.type == "memory.read"]
    assert sorted(reads) == ["checkpoint", "long_term"]
    # file-backed: a brand-new runner pointed at the same directory sees the memory
    fresh = ExperimentRunner(EventBus(), data_dir=tmp_path)
    assert len(fresh.memory.search("experiment-continuity")) == 4  # 3 findings + final_report


def test_memory_scenario_is_repeatable() -> None:
    runner = ExperimentRunner(EventBus())
    runner.run_to_completion("memory-continuity")
    runner.run_to_completion("memory-continuity")
    assert len(runner.memory.search("experiment-continuity")) == 4  # reset, not accumulated


def test_output_lens_runs_one_question_through_three_distinct_lenses() -> None:
    _, _, group = _run("output-lens")
    reports = [r.state["final_report"] for r in group.results]
    assert [r.label for r in group.runs] == ["Executive Brief", "Deep Analytical Breakdown", "Risk & Caveat Audit"]
    assert reports[0].startswith("# Executive Brief") and "Bottom line" in reports[0]
    assert "Evidence table" in reports[1] and "Risk register" in reports[2]
    assert len(set(reports)) == 3


def test_query_override_applies_to_first_phase_only_and_keeps_lens() -> None:
    _, _, g = _run("contradiction", query="Does remote work raise or lower productivity in hybrid teams?")
    assert "hybrid teams" in g.results[0].state["original_query"]
    _, _, lens = _run("output-lens", query="Is a home battery worth it for solar households?")
    assert lens.results[0].state["original_query"].endswith("Output lens: Executive Brief")
    assert lens.results[1].state["original_query"].count("home battery") == 1


def test_scenario_llm_helpers_and_search_backend() -> None:
    q, lens = split_query("Why?\nOutput lens: Executive Brief")
    assert (q, lens) == ("Why?", "Executive Brief")
    assert CorpusSearchBackend().search("zzz qqq", 5) == []
    assert len(CorpusSearchBackend().search("remote work productivity", 3)) == 3
    resp = ScenarioLLM().complete(system="Researcher agent", messages=[{"role": "user", "content": "remote work"}])
    assert resp.tool_calls[0].tool_name == "web_search"
