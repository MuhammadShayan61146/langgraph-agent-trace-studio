"""Human approval decision vocabulary.

Purpose
-------
Defines the closed set of decisions a human reviewer can make at the
`AWAITING_HUMAN_APPROVAL` gate (Module 5's `supervisor.py` parks the task
here when `requires_human_approval=True`, without resolving it further —
see that module's docstring "Design decisions").

Design decisions
-----------------
* **Kept in a new `human` package, not added to `core.enums`.** Per your
  standing instruction not to revisit completed modules absent a genuine
  bug, Module 2's `enums.py` is left untouched. `ApprovalDecision` is a
  narrower, module-specific vocabulary (three human-review outcomes) with
  a different lifecycle than `TaskStatus`/`AgentRole`/`MessageRole`
  (system-wide state machine concepts); giving it its own home in `human/`
  is consistent with Module 2's own OCP framing ("new data contracts can
  be added ... without touching existing ones") applied at the package
  level rather than the file level.
* **Subclasses `str, Enum`**, matching every enum in `core.enums`, for the
  same reason: JSON-serializability (Module 6's persistence layer) and
  safe string comparison.
* **Constructing `ApprovalDecision("not_a_real_value")` raises `ValueError`
  for free** (standard `Enum` behavior) — no custom validation function
  needed to reject unrecognized decision strings coming from, e.g., a
  deserialized human response payload.
"""

from __future__ import annotations

from enum import Enum


class ApprovalDecision(str, Enum):
    """A human reviewer's decision at the approval gate."""

    APPROVED = "approved"
    REJECTED = "rejected"
    REVISION_REQUESTED = "revision_requested"
