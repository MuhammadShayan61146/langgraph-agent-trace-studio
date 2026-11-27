"""Human-in-the-loop approval: framework-free request/decision logic.

Purpose
-------
Provides the two pure functions a real interrupt-based graph node
(`interrupt_node.py`, Design-only) will call: one to build what a human
reviewer sees, one to compute the resulting state update from their
decision. Kept framework-free — like `graph/supervisor.py` — so both are
independently testable without `langgraph` installed.

Responsibilities
-----------------
* `build_approval_request` — extract exactly what a human reviewer needs
  from `GraphState` into a plain dict (the payload a real
  `langgraph.types.interrupt(...)` call would surface to the caller
  resuming the graph).
* `apply_human_decision` — given a decision (+ optional feedback text),
  compute the partial state update: approve → `COMPLETED`, reject →
  `FAILED`, request revision → back to `WRITING` with the feedback
  attached as a message for the Writer to see next time.

Design decisions
-----------------
* **All three target transitions were already legal in Module 2's
  `TASK_STATUS_TRANSITIONS`** (`AWAITING_HUMAN_APPROVAL -> {COMPLETED,
  WRITING, FAILED}`) — nothing in `core/enums.py` needed to change for
  this module. `validate_transition` (Module 2) is called before every
  transition here, same discipline as every prior module.
* **`apply_human_decision` guards that the incoming state is actually
  parked at `AWAITING_HUMAN_APPROVAL`**, raising `ValueError` otherwise.
  This is a programmer-error guard (calling this function from the wrong
  state is a wiring bug, not a runtime condition to recover from), so a
  plain `ValueError` was used rather than adding a new exception type to
  `core.exceptions` for a single, narrow precondition check.
* **A revision request feeds back as a `USER`-role `AgentMessage`**, not a
  new contract type. `WriterAgent`'s existing `_messages_from_state`
  (Module 4, untouched) already folds every message into the context it
  sends the LLM — a revision request phrased as a user message needs zero
  changes to `WriterAgent` to be seen on its next turn.
* **No graph wiring in this module.** Per your standing instruction, this
  module does not modify `graph/build.py` or `graph/supervisor.py`.
  `interrupt_node.py` (this package) shows the Design-only real
  integration point; actually attaching it to the compiled graph is
  deferred to a future integration/assembly step and listed as a
  dependency in the Module 7 report, not performed here.

Tradeoffs
---------
* A rejected task moves straight to `FAILED` (terminal) rather than
  offering a further retry path from rejection specifically — a real
  product might want "rejected -> back to Researcher," but that transition
  isn't in Module 2's table and adding it would mean modifying a completed
  module's file without a bug to justify it. Documented here as a
  known scope boundary, not silently worked around.
"""

from __future__ import annotations

from typing import Any

from agentic_research.core.contracts import AgentMessage, ErrorRecord
from agentic_research.core.enums import MessageRole, TaskStatus
from agentic_research.core.state import GraphState, validate_transition
from agentic_research.human.decision import ApprovalDecision


def build_approval_request(state: GraphState) -> dict[str, Any]:
    """Build the payload a human reviewer should see at the approval gate.

    Args:
        state: The current graph state. Expected (not enforced by this
            function) to have ``task_status == TaskStatus.AWAITING_HUMAN_APPROVAL``.

    Returns:
        A plain dict: the original query, the draft report, how many
        findings back it, and how many prior revision cycles have already
        happened (derived from how many `WRITER` messages exist).
    """
    writer_turns = sum(
        1 for msg in state["messages"] if msg.agent is not None and msg.agent.value == "writer"
    )
    return {
        "original_query": state["original_query"],
        "draft_report": state["final_report"],
        "supporting_findings_count": len(state["research_findings"]),
        "revision_cycle": writer_turns,
    }


def apply_human_decision(
    state: GraphState,
    decision: ApprovalDecision,
    feedback: str | None = None,
) -> dict[str, Any]:
    """Compute the state update resulting from a human's approval decision.

    Args:
        state: The current graph state. Must have
            ``task_status == TaskStatus.AWAITING_HUMAN_APPROVAL``.
        decision: The human's decision.
        feedback: Optional free-text feedback, required in practice for
            `ApprovalDecision.REVISION_REQUESTED` (attached as a message
            for the Writer to see on its next turn) and ignored otherwise.

    Returns:
        A partial state update, following the same shape every agent node
        and the supervisor return.

    Raises:
        ValueError: If ``state["task_status"]`` is not
            `TaskStatus.AWAITING_HUMAN_APPROVAL` — calling this function
            outside that state is a wiring error, not a recoverable
            runtime condition.
        InvalidStateTransitionError: Defensive: would only fire if Module
            2's transition table were ever narrowed to no longer permit
            one of these three moves.
    """
    if state["task_status"] != TaskStatus.AWAITING_HUMAN_APPROVAL:
        raise ValueError(
            "apply_human_decision called with task_status="
            f"{state['task_status']!r}; expected AWAITING_HUMAN_APPROVAL."
        )

    if decision == ApprovalDecision.APPROVED:
        validate_transition(TaskStatus.AWAITING_HUMAN_APPROVAL, TaskStatus.COMPLETED)
        return {
            "task_status": TaskStatus.COMPLETED,
            "human_approval_status": ApprovalDecision.APPROVED.value,
        }

    if decision == ApprovalDecision.REJECTED:
        validate_transition(TaskStatus.AWAITING_HUMAN_APPROVAL, TaskStatus.FAILED)
        return {
            "task_status": TaskStatus.FAILED,
            "human_approval_status": ApprovalDecision.REJECTED.value,
            "error_log": [
                ErrorRecord(
                    node_name="human_approval",
                    error_type="ReportRejected",
                    message=feedback or "Report rejected by human reviewer.",
                    recoverable=False,
                )
            ],
        }

    # ApprovalDecision.REVISION_REQUESTED
    validate_transition(TaskStatus.AWAITING_HUMAN_APPROVAL, TaskStatus.WRITING)
    revision_message = AgentMessage(
        role=MessageRole.USER,
        content=f"Revision requested by human reviewer: {feedback or '(no feedback given)'}",
    )
    return {
        "task_status": TaskStatus.WRITING,
        "human_approval_status": None,
        "messages": [revision_message],
    }
