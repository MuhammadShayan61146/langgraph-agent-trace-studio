"""The real LangGraph human-approval node: pauses execution via `interrupt()`.

**Status: Design-only.** `langgraph` is not installed in this sandbox (see
Module 1's environment disclosure, restated in every module that touches
`langgraph` directly). This file is syntax-checked
(`python -m py_compile`) and reviewed against LangGraph's publicly
documented `interrupt()`/`Command(resume=...)` API shape, but has never
been imported or executed. Everything it calls
(`build_approval_request`, `apply_human_decision`, `ApprovalDecision`) is
independently Verified in `approval.py`/`decision.py`; only this file's use
of the real LangGraph pause/resume mechanism is not.

How this would be wired in (not performed in this module)
-----------------------------------------------------------
A future integration step would register this as a node the graph
transitions to instead of ending at `AWAITING_HUMAN_APPROVAL` — the exact
seam Module 5's `supervisor.py` docstring names
(`_handle_awaiting_approval`). That wiring is deliberately not done here,
per the instruction not to modify already-approved module files without a
discovered bug; it is listed as a dependency in the Module 7 report.

Design decisions
-----------------
* **`interrupt()` is called with the dict from `build_approval_request`,
  and its return value is expected to be a dict with a `"decision"` key**
  (and optional `"feedback"`), matching how a caller would resume via
  `graph.invoke(Command(resume={"decision": "approved"}), config)` per
  LangGraph's documented human-in-the-loop pattern. This shape is a design
  choice, not a verified contract, since it was never exercised against a
  real `interrupt()` call.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from agentic_research.core.state import GraphState
from agentic_research.human.approval import apply_human_decision, build_approval_request
from agentic_research.human.decision import ApprovalDecision


def human_approval_node(state: GraphState) -> dict[str, Any]:
    """Pause the graph for human review, then apply the resulting decision.

    Args:
        state: The current graph state, expected to have
            ``task_status == TaskStatus.AWAITING_HUMAN_APPROVAL``.

    Returns:
        The partial state update from `apply_human_decision`, once
        execution resumes with the human's response.

    Note:
        Not executed in this sandbox — see module docstring.
    """
    request_payload = build_approval_request(state)
    response = interrupt(request_payload)
    decision = ApprovalDecision(response["decision"])
    feedback = response.get("feedback")
    return apply_human_decision(state, decision, feedback)
