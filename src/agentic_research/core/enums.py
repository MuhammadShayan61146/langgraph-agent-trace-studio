"""Core enumerations shared across the entire application.

Purpose
-------
Defines the closed vocabularies used throughout the system: who sent a
message (``MessageRole``), which agent is speaking or acting
(``AgentRole``), and where a task currently sits in its lifecycle
(``TaskStatus``).

Responsibilities
-----------------
Nothing beyond defining these vocabularies and — for ``TaskStatus`` — the
legal transition graph between states. No I/O, no framework imports.

Design decisions
-----------------
* All enums subclass ``str`` in addition to ``Enum`` (i.e. ``class
  MessageRole(str, Enum)``). This makes every member both a proper enum
  *and* a plain string (``MessageRole.USER == "user"`` is ``True``), which
  matters because: (a) it serializes directly to JSON via the logging
  layer's ``_JsonFormatter`` from Module 1 without a custom encoder, and
  (b) LangGraph state dictionaries frequently round-trip through JSON at
  checkpointing boundaries in real deployments, so string-compatible enums
  avoid a whole class of serialization bugs later.
* ``TASK_STATUS_TRANSITIONS`` is defined here (co-located with the enum it
  governs) rather than in ``state.py``, so the "what values exist" and
  "what transitions are legal" facts about ``TaskStatus`` can't drift apart
  across two files.

Tradeoffs
---------
* Str-enums are slightly less strict than plain ``Enum`` (accidental string
  comparisons succeed silently), but the ergonomic and serialization
  benefits outweigh that here, since state is designed to cross
  process/checkpoint boundaries.

Limitations
------------
* The transition graph below is Module 2's best-effort design based on the
  planned agent architecture (Supervisor/Researcher/Analyst/Writer +
  human-in-the-loop). It has not yet been exercised against a real graph
  execution (that happens in Module 5+) and may need refinement then — see
  "Known Risks" in the Module 2 report.

Future improvements
--------------------
* Consider generating a Mermaid state diagram directly from
  ``TASK_STATUS_TRANSITIONS`` for the architecture docs, so the diagram
  can never drift out of sync with the code.
"""

from __future__ import annotations

from enum import Enum


class MessageRole(str, Enum):
    """Who authored a given message in the shared conversation log."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class AgentRole(str, Enum):
    """Identifies a specific agent (or the human) within the multi-agent system."""

    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    HUMAN = "human"


class TaskStatus(str, Enum):
    """Lifecycle stage of the overall research task tracked in graph state."""

    PENDING = "pending"
    ROUTING = "routing"
    RESEARCHING = "researching"
    ANALYZING = "analyzing"
    WRITING = "writing"
    AWAITING_HUMAN_APPROVAL = "awaiting_human_approval"
    COMPLETED = "completed"
    FAILED = "failed"


#: Legal ``TaskStatus`` transitions, expressed as {from_status: {allowed_to_statuses}}.
#:
#: Read as a directed graph: ``PENDING -> ROUTING`` means a task may move from
#: pending to routing, but not the reverse. ``COMPLETED`` has no outgoing edges
#: (terminal state). ``FAILED`` may route back to ``ROUTING`` to support a
#: future retry-from-supervisor recovery path (Module 8); this is a documented
#: design choice, not an oversight — see Module 2 report "Design Decisions".
TASK_STATUS_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset({TaskStatus.ROUTING}),
    TaskStatus.ROUTING: frozenset(
        {
            TaskStatus.RESEARCHING,
            TaskStatus.ANALYZING,
            TaskStatus.WRITING,
            TaskStatus.FAILED,
        }
    ),
    TaskStatus.RESEARCHING: frozenset({TaskStatus.ROUTING, TaskStatus.FAILED}),
    TaskStatus.ANALYZING: frozenset({TaskStatus.ROUTING, TaskStatus.FAILED}),
    TaskStatus.WRITING: frozenset(
        {
            TaskStatus.ROUTING,
            TaskStatus.AWAITING_HUMAN_APPROVAL,
            TaskStatus.FAILED,
        }
    ),
    TaskStatus.AWAITING_HUMAN_APPROVAL: frozenset(
        {TaskStatus.COMPLETED, TaskStatus.WRITING, TaskStatus.FAILED}
    ),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset({TaskStatus.ROUTING}),
}
