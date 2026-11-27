"""LangGraph state schema and supporting pure logic.

Purpose
-------
Defines ``GraphState``: the ``TypedDict`` that will be passed directly to
``langgraph.graph.StateGraph(GraphState)`` in Module 5. Also provides the
reducer functions referenced by that schema, a factory for constructing a
valid initial state, and a pure validator for ``TaskStatus`` transitions.

Responsibilities
-----------------
* Declare the shape of graph state and how its list-typed fields merge
  across node updates (reducers).
* Provide ``create_initial_state`` so every entrypoint (CLI, tests, future
  API layer) constructs state the same validated way instead of hand-
  building dicts.
* Provide ``validate_transition`` so any layer that changes
  ``task_status`` (nodes in Module 4/5) can check legality without
  duplicating the transition table.

Design decisions
-----------------
* **``GraphState`` uses only ``typing.TypedDict``/``typing.Annotated`` —
  zero LangGraph import.** LangGraph's ``StateGraph`` accepts any
  ``TypedDict`` (or pydantic model) as its state type and reads
  ``Annotated[..., reducer]`` metadata to decide how to merge concurrent
  partial updates; none of that requires importing ``langgraph`` to
  *define* the schema, only to *use* it inside a compiled graph. This
  lets Module 2 be 100% verified in this network-disabled sandbox while
  still being the exact, real schema Module 5 will import unchanged.
* **One generic reducer, ``accumulate_list``, shared by every list field**
  (messages, findings, routing_history, error_log) instead of four
  near-identical reducer functions. All four fields have literally the
  same merge semantics (concatenate), so writing four copies would violate
  DRY for no behavioral benefit. If a field ever needs different merge
  semantics (e.g. de-duplicating findings by source), it gets its own
  reducer at that point — not preemptively.
* **Scalar fields (``current_agent``, ``task_status``, ``retry_count``,
  etc.) are declared without ``Annotated``**, relying on LangGraph's
  documented default behavior for un-annotated ``TypedDict`` keys: the
  most recent node's returned value simply overwrites the previous one.
  This matches the intended semantics (there is exactly one "current task
  status" at a time) without needing a trivial "always overwrite" reducer.
* **``validate_transition`` is a pure function, not a method on a class.**
  There is no natural object to hang it off yet (no ``StateMachine`` class
  exists, and adding one purely to hold one function would be needless
  ceremony) — a free function operating on the enum + table already
  defined in ``enums.py`` is the simplest thing that could work, per
  YAGNI. This can be wrapped in a class later if Module 5 needs to bundle
  it with other stateful orchestration concerns.

Tradeoffs
---------
* Because reducers here are plain functions (not classes implementing a
  ``Reducer`` protocol), adding a genuinely different merge strategy later
  means adding a new function, not overriding a method. Given there are
  currently exactly two merge strategies in the whole system (accumulate,
  overwrite), a class hierarchy would be premature abstraction.

Limitations
------------
* This module has never been passed into a real ``StateGraph`` (that
  requires the ``langgraph`` package, unavailable in this sandbox). Its
  compatibility with LangGraph's actual reducer-invocation mechanics is a
  documented risk, not a verified fact — see the Module 2 report's "Known
  Risks" section.

Future improvements
--------------------
* Add a dedicated reducer for ``research_findings`` that de-duplicates by
  ``(source, content)`` if duplicate findings become a real problem once
  the Researcher agent exists (Module 4).
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from agentic_research.core.contracts import (
    AgentMessage,
    ErrorRecord,
    ResearchFinding,
    RoutingDecision,
)
from agentic_research.core.enums import TASK_STATUS_TRANSITIONS, AgentRole, TaskStatus
from agentic_research.core.exceptions import InvalidStateTransitionError


def accumulate_list(existing: list, new: list) -> list:
    """Merge two lists by concatenation, for use as a LangGraph state reducer.

    This is the single reducer used by every list-typed field in
    ``GraphState`` (see "Design decisions" above for why one shared
    function is preferred over one per field).

    Args:
        existing: The list already present in state before this node ran.
        new: The list returned by the node's partial state update.

    Returns:
        A new list containing every element of ``existing`` followed by
        every element of ``new``. Neither input list is mutated.
    """
    return list(existing) + list(new)


class GraphState(TypedDict):
    """The complete shared state threaded through every node in the graph.

    List-typed fields are annotated with ``accumulate_list`` so that when
    two or more nodes contribute to the same field (including in parallel
    branches, once those exist), their contributions are concatenated
    rather than one silently overwriting another. Scalar fields intentionally
    have no reducer annotation: the latest write wins, which is correct for
    fields that represent "the current X" rather than "the history of X".

    Attributes:
        original_query: The user's original research request. Set once at
            graph entry and never modified afterward.
        messages: Full conversation history (accumulates).
        research_findings: Every finding collected so far (accumulates).
        routing_history: Every supervisor routing decision made so far
            (accumulates) — this is what makes routing auditable/debuggable.
        error_log: Every recoverable error encountered so far (accumulates).
        current_agent: Which agent is currently active / was last active.
        task_status: Current lifecycle stage (see ``TaskStatus``).
        retry_count: Number of retry attempts made for the current node.
        requires_human_approval: Whether the workflow is configured to pause
            for human approval before finalizing the report.
        human_approval_status: ``"approved"``, ``"rejected"``, or ``None``
            (no decision yet / not applicable).
        final_report: The finished report text, once the Writer agent has
            produced it and (if required) it has been approved. ``None``
            until then.
    """

    original_query: str
    messages: Annotated[list[AgentMessage], accumulate_list]
    research_findings: Annotated[list[ResearchFinding], accumulate_list]
    routing_history: Annotated[list[RoutingDecision], accumulate_list]
    error_log: Annotated[list[ErrorRecord], accumulate_list]
    current_agent: AgentRole
    task_status: TaskStatus
    retry_count: int
    requires_human_approval: bool
    human_approval_status: str | None
    final_report: str | None


def create_initial_state(
    query: str,
    *,
    requires_human_approval: bool = False,
) -> GraphState:
    """Construct a valid, empty ``GraphState`` for a new graph run.

    Args:
        query: The user's research request. Must be non-empty.
        requires_human_approval: Whether this run should pause for human
            approval before the final report is emitted (Module 7).

    Returns:
        A fresh ``GraphState`` with ``task_status=TaskStatus.PENDING``,
        every accumulating list empty, and ``current_agent`` set to
        ``AgentRole.SUPERVISOR`` (the graph's designated entry point).

    Raises:
        ValueError: If ``query`` is empty or whitespace-only.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string.")

    return GraphState(
        original_query=query,
        messages=[],
        research_findings=[],
        routing_history=[],
        error_log=[],
        current_agent=AgentRole.SUPERVISOR,
        task_status=TaskStatus.PENDING,
        retry_count=0,
        requires_human_approval=requires_human_approval,
        human_approval_status=None,
        final_report=None,
    )


def validate_transition(from_status: TaskStatus, to_status: TaskStatus) -> None:
    """Validate that moving from ``from_status`` to ``to_status`` is legal.

    Args:
        from_status: The task's current status.
        to_status: The status a node is attempting to move the task to.

    Raises:
        InvalidStateTransitionError: If ``to_status`` is not in the set of
            statuses reachable from ``from_status`` per
            ``TASK_STATUS_TRANSITIONS``.
    """
    allowed = TASK_STATUS_TRANSITIONS.get(from_status, frozenset())
    if to_status not in allowed:
        raise InvalidStateTransitionError(
            from_status=from_status.value, to_status=to_status.value
        )
