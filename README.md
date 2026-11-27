---
title: LangGraph Agent Trace Studio
emoji: 🕸️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Agentic Research System

A production-inspired, LangGraph-based multi-agent research and
report-generation system.

The system uses a Supervisor-controlled workflow to coordinate
specialised Researcher, Analyst, and Writer agents. It demonstrates
stateful workflow orchestration, tool calling, memory, human-in-the-loop
approval, and bounded error recovery.

---

### Automated test suite

```powershell
pytest -q

Real LangGraph demonstration
The compiled LangGraph workflow was also executed directly:
python scripts\demo_multi_agent.py
Current result:
Graph compiled successfully.
Invoking compiled LangGraph...

EXECUTION COMPLETED

Task status: TaskStatus.COMPLETED


Messages:
  - Researched information.
  - Analyzed information.
  - Final demonstration report.

REAL LANGGRAPH DEMONSTRATION FINISHED
This provides runtime evidence that the implemented graph can be constructed, compiled, invoked, and completed successfully with the local demonstration configuration.
Architecture
The system follows a Supervisor-based multi-agent architecture:
                         ┌─────────────────────┐
                         │        START        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     SUPERVISOR      │
                         │  Workflow Control   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     RESEARCHER      │
                         │ Research / Findings │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     SUPERVISOR      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       ANALYST       │
                         │ Analysis / Synthesis│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     SUPERVISOR      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       WRITER        │
                         │   Report Generation │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     SUPERVISOR      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   HUMAN APPROVAL    │
                         │   Review / Decision │
                         └──────────┬──────────┘
                                    │
                              ┌─────┴─────┐
                              │           │
                           APPROVE     REVISION
                              │           │
                              ▼           │
                         ┌─────────┐      │
                         │  END    │◄─────┘
                         └─────────┘
The Supervisor provides deterministic workflow control while the specialised agents perform their respective responsibilities.

Project Structure

agentic-research-system/
│
├── src/
│   └── agentic_research/
│       ├── core/
│       │   ├── contracts.py
│       │   ├── enums.py
│       │   ├── exceptions.py
│       │   └── state.py
│       │
│       ├── config/
│       │   └── settings.py
│       │
│       ├── tools/
│       │   ├── base.py
│       │   ├── calculator.py
│       │   ├── registry.py
│       │   └── search.py
│       │
│       ├── agents/
│       │   ├── base.py
│       │   ├── researcher.py
│       │   ├── analyst.py
│       │   └── writer.py
│       │
│       ├── memory/
│       │   ├── checkpoint.py
│       │   └── long_term.py
│       │
│       ├── human/
│       │   ├── approval.py
│       │   ├── decision.py
│       │   └── interrupt_node.py
│       │
│       ├── graph/
│       │   ├── supervisor.py
│       │   └── build.py
│       │
│       └── error_recovery.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── scripts/
│   ├── run_tests.py
│   └── demo_multi_agent.py
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── TASK4_DEMONSTRATION.md
│   └── module-reports/
│
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── README.md


Core Components:

Supervisor
The Supervisor controls workflow progression and determines which specialised agent should execute next.
The current routing policy is deterministic and follows:
Researcher → Analyst → Writer
The Supervisor is responsible for maintaining workflow control rather than performing the specialised research or writing work itself.

Researcher Agent
The Researcher processes the research stage and contributes findings to the shared graph state.

Analyst Agent
The Analyst processes available research information and contributes analysis to the shared workflow state.

Writer Agent
The Writer uses the accumulated workflow information to produce the final report.

Human Approval
The project contains a human-in-the-loop approval layer supporting:
Approval
Rejection
Revision requests
The framework-free approval logic is independently tested, while the LangGraph integration provides the designated human-approval node and routing seam.

Error Recovery
error_recovery.with_retry provides bounded retry behaviour and is reused across graph nodes.
Memory
The project includes:
Short-term checkpoint/state persistence
Long-term namespaced fact storage

Tools
The tool layer provides:
Calculator tool
Search tool
Tool registry
Dependency-injected search backend
The calculator supports offline evaluation through a restricted AST-based implementation.


Running the Real LangGraph Demonstration
The repository includes a practical demonstration script:
python scripts\demo_multi_agent.py

The demonstration:
Builds the real LangGraph StateGraph.
Compiles the graph.
Creates the initial workflow state.
Invokes the compiled graph.
Routes execution through the Supervisor.
Executes the Researcher.
Executes the Analyst.
Executes the Writer.
Processes the configured approval path.
Reports the resulting workflow state.

---

## Visual trace viewer & Experiments Suite

A live dashboard (graph hand-offs with routing rationale, thought stream,
event log, HITL approve/revise/reject, replay, run dossiers) and five
repeatable experiments (contradiction synthesis, fault injection, human
checkpoint, memory continuity, output lenses) are available with no extra
dependencies:

```bash
python scripts/serve_dashboard.py --open
python scripts/run_experiment.py --list
pytest -q
```

