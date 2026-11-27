"""Command-line runner for the Experiments Suite.

    python -m agentic_research.experiments --list
    python -m agentic_research.experiments contradiction
    python -m agentic_research.experiments fault-drill --variant exhausted
    python -m agentic_research.experiments human-checkpoint --interactive
    python -m agentic_research.experiments output-lens --export-dir out/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from agentic_research.experiments.runner import ExperimentRunner
from agentic_research.experiments.scenarios import SCENARIOS, get_scenario
from agentic_research.telemetry.dossier import build_dossier, render_html, render_json, render_markdown
from agentic_research.telemetry.events import EventBus


def _interactive_decider(payload: dict) -> tuple[str, str | None]:
    print("\n--- Human gate: draft report ---")
    print(payload.get("draft_report"))
    print(f"\n({payload.get('supporting_findings_count')} findings, revision cycle {payload.get('revision_cycle')})")
    while True:
        choice = input("[a]pprove, [r]eject or [v] request revision? ").strip().lower()[:1]
        if choice == "a":
            return "approved", None
        if choice == "r":
            return "rejected", input("Reason (optional): ").strip() or None
        if choice == "v":
            fb = input("Feedback for the Writer: ").strip()
            if fb:
                return "revision_requested", fb


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentic_research.experiments", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", nargs="?", help="Scenario id (see --list).")
    p.add_argument("--list", action="store_true", help="List scenarios and exit.")
    p.add_argument("--variant", help="Scenario variant (e.g. 'exhausted' for fault-drill).")
    p.add_argument("--query", help="Override the first phase's prompt.")
    p.add_argument("--interactive", action="store_true", help="Answer human-gate prompts yourself (default: auto-approve).")
    p.add_argument("--export-dir", help="Write each run's dossier (md, html, json) here.")
    p.add_argument("--data-dir", help="Persist checkpoints and long-term memory here (default: in-memory).")
    args = p.parse_args(argv)

    if args.list or not args.scenario:
        for s in SCENARIOS:
            variants = f"  variants: {', '.join(s.variants)}" if s.variants else ""
            print(f"{s.id:18} {s.title}{variants}")
        return 0 if args.list else 2
    try:
        get_scenario(args.scenario)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 2

    bus = EventBus()
    runner = ExperimentRunner(bus, data_dir=args.data_dir)
    try:
        group = runner.run_to_completion(
            args.scenario, variant=args.variant, query=args.query,
            decider=_interactive_decider if args.interactive else None,
        )
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 2

    out_dir = Path(args.export_dir) if args.export_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    for run in group.runs:
        dossier = build_dossier([e.to_dict() for e in bus.history(run_id=run.run_id)])
        print(f"\n=== {run.label}  [{dossier['status']}]  {dossier['steps']} steps, {dossier['llm_calls']} LLM calls ===")
        for r in dossier["routes"]:
            print(f"  {r['source']:>14} -> {r['target']:<14} {r['reason']}")
        for f in dossier["faults"]:
            print(f"  ! {f['detail']}")
        print("\n" + (dossier["final_report"] or f"(no report) {dossier['error']}"))
        if out_dir:
            for ext, text in (("md", render_markdown(dossier)), ("html", render_html(dossier)), ("json", render_json(dossier))):
                (out_dir / f"{run.run_id}.dossier.{ext}").write_text(text, encoding="utf-8")
    if out_dir:
        print(f"\nDossiers written to {out_dir}/")
    return 0 if group.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
