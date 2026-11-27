# Visual trace viewer & Experiments Suite

An additive layer on top of the research graph: a live trace dashboard, five
repeatable experiments, and exportable run dossiers. 
## Quick start

```bash
pip install -e .[dev]                 # or just: export PYTHONPATH=src
python scripts/serve_dashboard.py --open      # http://127.0.0.1:8765
python scripts/run_experiment.py --list       # headless, no browser
python scripts/run_experiment.py fault-drill --export-dir out/
pytest -q                                     # 240 original + new tests
```


## Environment variables / flags

| Variable | Flag | Default | Meaning |
|---|---|---|---|
| `AGENTIC_VIZ_HOST` | `--host` | `127.0.0.1` | Bind address (keep loopback; there is no auth). |
| `AGENTIC_VIZ_PORT` | `--port` | `8765` | Port (`0` = any free port). |
| `AGENTIC_VIZ_DATA_DIR` | `--data-dir` | `.agentic_viz` | Traces (`runs/*.jsonl`), checkpoints and long-term memory (`state/`). |
| `AGENTIC_VIZ_PACE` | `--pace` | `0.4` | Seconds of delay after visible events so live runs are legible. `0` = full speed. |

## Architecture

```
src/agentic_research/
  telemetry/            # nothing in core imports this
    events.py           # TraceEvent, EventBus, RunEmitter, JsonlSink
    instrument.py       # instrument_node, TracedLLMClient, TracedToolRegistry, FaultInjector
    runtime.py          # TracedGraphRunner: same nodes/routing/with_retry, + pause/resume
    topology.py         # static graph description for the UI
    dossier.py          # events -> dossier (json / markdown / html)
  experiments/
    scenarios.py        # 5 scenarios as data (Phase, FaultRule, memory wiring)
    runner.py           # ExperimentRunner: groups of runs, pause/resume, shared memory
    llm.py corpora.py   # deterministic stateless LLM stand-in + offline fixture corpora
    cli.py              # python -m agentic_research.experiments
  viz/
    server.py           # stdlib HTTP + SSE server and JSON API
    static/             # index.html, style.css, app.js
```

**One event stream.** Every node attempt, LLM call, tool call, routing choice,
state snapshot, checkpoint, memory read/write, fault, recovery and human
decision is a `TraceEvent` on an `EventBus`. The UI, JSONL files, dossiers and
tests are all consumers. In the UI a single pure function `derive(events)`
produces every panel, so live view, replay and scrubbing share one render path.

**Why a traced runner instead of editing `build.py`.** `build.py` registers the
project's node functions on a langgraph `StateGraph`. `TracedGraphRunner`
imports and drives the *same* node functions, routing functions
(`route_from_supervisor`, `route_after_supervisor`) and `with_retry`, so
behaviour matches (a parity test compares it with the project's own simulation).
A unit test parses `build.py` with `ast` and fails if the dashboard's topology
drifts from the real wiring.

**Human-in-the-loop.** At the gate the runner saves a checkpoint and returns
`PAUSED`. `resume()` reloads that checkpoint and continues; with a file-backed
store this works from a fresh process (tested).

## Experiments

| id | Shows | Look for |
|---|---|---|
| `contradiction` | Competing sources reconciled into a structured report | Stance-tagged findings; two calculator calls; Position A/B/Reconciliation/Confidence |
| `fault-drill` | Tool failure, node crash and LLM failure all recovered. Variant `exhausted` fails cleanly | `fault.injected`, retry attempts 1→2→3, recovery events; error log |
| `human-checkpoint` | Pause, inspect, approve / reject / revise | Checkpoint before pause; revision loop returns to the gate |
| `memory-continuity` | Run 2 recalls run 1's findings and checkpointed report | `memory.write` then `memory.read` (long-term + checkpoint); "Continuity" section |
| `output-lens` | One question through three lenses | Compare tab |

Add a scenario by appending a `Scenario` to `SCENARIOS` in
`experiments/scenarios.py`; it appears in the UI and CLI automatically.

## Dashboard tour

- **Graph:** pulsing node = working, dashed outline = next up, red = failed/retrying, rose = awaiting you; a badge shows revisit counts; a token travels the chosen edge. The line under the graph states *why*.
- **Thought stream / Event log:** per-agent cards with LLM text, tool calls and results; filterable raw events with JSON detail.
- **Player:** replay any saved run, scrub, change speed, or return to live.
- **Inspector:** Report (draft until approved), State (routing history, findings, error log, raw), Audit (tools, memory, faults), Compare (phases side by side), Export.
- **Export:** dossier as Markdown / HTML / JSON, raw events as JSONL. "Open a saved trace…" replays a downloaded `.jsonl` or dossier `.json` entirely client-side.
