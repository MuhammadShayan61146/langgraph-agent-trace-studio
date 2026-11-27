"""Scenario catalogue for the Experiments Suite.

A `Scenario` is data only: preset prompt(s), the phases to run, any fault
plan, memory wiring and human-in-the-loop flag. `ExperimentRunner` turns it
into one or more traced runs. Add a scenario by appending to `SCENARIOS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_research.experiments.llm import LENSES
from agentic_research.telemetry.instrument import FaultRule

MEMORY_NAMESPACE = "experiment-continuity"


@dataclass(frozen=True)
class Phase:
    """One traced run inside a scenario."""

    label: str
    query: str
    requires_human_approval: bool = False
    faults: tuple[FaultRule, ...] = ()
    #: Namespace to recall findings from before the run starts.
    recall_namespace: str | None = None
    #: Namespace to persist findings/report into after a completed run.
    save_namespace: str | None = None
    #: Wipe ``save_namespace`` first so the scenario is repeatable.
    reset_memory: bool = False
    #: Inject the previous phase's checkpointed final report as context.
    recall_previous_report: bool = False
    max_retries: int | None = None


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    tagline: str
    description: str
    look_for: tuple[str, ...]
    tags: tuple[str, ...]
    phases: tuple[Phase, ...]
    #: Optional named variants: name -> replacement phases.
    variants: dict[str, tuple[Phase, ...]] = field(default_factory=dict)
    variant_notes: dict[str, str] = field(default_factory=dict)
    #: True if the user-edited prompt should replace phase 0's query.
    editable_prompt: bool = True

    def phases_for(self, variant: str | None) -> tuple[Phase, ...]:
        if variant and variant != "default":
            if variant not in self.variants:
                raise KeyError(f"Unknown variant {variant!r} for scenario {self.id!r}.")
            return self.variants[variant]
        return self.phases


_REMOTE_Q = ("Does remote work raise or lower team productivity? "
             "Reconcile the competing evidence.")
_FOUR_Q = "Summarize the evidence on four day work week trials."
_BATT_Q = "Is a home battery worth it for a household with solar? Summarize the evidence."
_LENS_BASE_Q = "Is a home battery worth it for a household with solar?"


def _lens_phases() -> tuple[Phase, ...]:
    return tuple(
        Phase(label=lens, query=f"{_LENS_BASE_Q}\nOutput lens: {lens}")
        for lens in LENSES[1:]
    )


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="contradiction",
        title="Contradiction & multi-perspective synthesis",
        tagline="Competing evidence in, one structured reconciliation out.",
        description=("The Researcher retrieves sources that disagree. The Analyst quantifies the "
                     "spread with the calculator tool and the Writer reconciles both positions "
                     "into a structured report instead of picking a side."),
        look_for=("Stance-tagged findings in the Researcher's stream",
                  "Two calculator calls (mean effect, spread) in the Analyst's stream",
                  "Report with Position A / Position B / Reconciliation / Confidence"),
        tags=("synthesis", "tools"),
        phases=(Phase(label="Synthesis", query=_REMOTE_Q),),
    ),
    Scenario(
        id="fault-drill",
        title="Fault injection & error-recovery drill",
        tagline="A flaky tool, a crashing node and a failing LLM call — and the run still lands.",
        description=("Injects three failures: the search tool fails once (the Researcher degrades "
                     "gracefully with a broader query), the Analyst node raises twice and the "
                     "Writer's LLM call raises once. `with_retry` recovers each. The 'exhausted' "
                     "variant makes the Analyst fail permanently so you can watch the retry "
                     "budget run out and the run fail cleanly."),
        look_for=("fault.injected events on the timeline",
                  "Retry attempts 1 → 2 → 3 on the Analyst node",
                  "Recovery badge, and error_log entries in the State inspector"),
        tags=("resilience", "retry"),
        phases=(Phase(
            label="Recoverable faults", query=_FOUR_Q, max_retries=3,
            faults=(FaultRule("tool", "web_search", 1, "Search backend timed out (injected)"),
                    FaultRule("node", "analyst", 2, "Transient analyst crash (injected)"),
                    FaultRule("llm", "writer", 1, "LLM rate limit (injected)")),
        ),),
        variants={"exhausted": (Phase(
            label="Retry budget exhausted", query=_FOUR_Q, max_retries=2,
            faults=(FaultRule("node", "analyst", 99, "Analyst unavailable (injected, permanent)"),),
        ),)},
        variant_notes={"exhausted": "Analyst fails on every attempt; the run ends FAILED after the retry budget."},
    ),
    Scenario(
        id="human-checkpoint",
        title="Human-in-the-loop checkpoint & resume",
        tagline="The graph pauses at the gate; you inspect state and decide.",
        description=("The run drafts a report, checkpoints its state and pauses at the human gate. "
                     "Inspect the paused state, then approve, reject, or request a revision — a "
                     "revision sends the Writer around again with your feedback and pauses once more."),
        look_for=("run.paused + human.interrupt with the draft payload",
                  "memory.checkpoint before the pause; checkpoint load on resume",
                  "Revision loop: gate → writer → supervisor → gate"),
        tags=("human-in-the-loop", "checkpoint"),
        phases=(Phase(label="Gated run", query=_BATT_Q, requires_human_approval=True),),
    ),
    Scenario(
        id="memory-continuity",
        title="Persistent memory continuity",
        tagline="Run 2 builds on what run 1 learned.",
        description=("Run 1 researches urban cycling infrastructure and saves findings + report to "
                     "long-term memory (file-backed). Run 2 asks a follow-up about costs in a new "
                     "thread: it recalls run 1's findings and its checkpointed report, then merges "
                     "them with fresh research."),
        look_for=("memory.write at the end of run 1",
                  "memory.read (long_term + checkpoint) at the start of run 2",
                  "'Continuity' section in run 2's report citing recalled findings"),
        tags=("memory", "continuity"),
        editable_prompt=False,
        phases=(
            Phase(label="Run 1 — seed", query="What are the benefits of urban cycling infrastructure for cities?",
                  save_namespace=MEMORY_NAMESPACE, reset_memory=True),
            Phase(label="Run 2 — follow-up", query="Given that, what are the costs and funding concerns?",
                  recall_namespace=MEMORY_NAMESPACE, recall_previous_report=True),
        ),
    ),
    Scenario(
        id="output-lens",
        title="Output lens / style comparison",
        tagline="One question, three presentation lenses, side by side.",
        description=("The same question runs three times with a different `Output lens` — "
                     "Executive Brief, Deep Analytical Breakdown and Risk & Caveat Audit — and "
                     "the dashboard's Compare tab lays the deliverables next to each other."),
        look_for=("Identical research path, different Writer output",
                  "Compare tab: three lenses, one question"),
        tags=("presentation", "compare"),
        phases=_lens_phases(),
    ),
)

_BY_ID = {s.id: s for s in SCENARIOS}


def get_scenario(scenario_id: str) -> Scenario:
    try:
        return _BY_ID[scenario_id]
    except KeyError:
        raise KeyError(f"Unknown scenario {scenario_id!r}. Available: {sorted(_BY_ID)}") from None


def scenario_catalog() -> list[dict[str, object]]:
    """JSON-friendly catalogue for the dashboard."""
    return [
        {
            "id": s.id, "title": s.title, "tagline": s.tagline,
            "description": s.description, "look_for": list(s.look_for),
            "tags": list(s.tags), "editable_prompt": s.editable_prompt,
            "default_query": s.phases[0].query.split("\nOutput lens:")[0],
            "phases": [p.label for p in s.phases],
            "variants": [{"id": k, "note": s.variant_notes.get(k, "")} for k in s.variants],
            "human_in_the_loop": any(p.requires_human_approval for p in s.phases),
        }
        for s in SCENARIOS
    ]
