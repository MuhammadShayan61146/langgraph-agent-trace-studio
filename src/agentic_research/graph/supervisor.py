"""Supervisor: conditional routing logic, kept framework-free like Module 2's state.

Purpose
-------
Decides, at each step, which agent should act next (or whether the run is
done) — the "Conditional Routing" and "Workflow Control" requirements of
this project. Split deliberately into pure functions operating on
`GraphState`, so this entire module is testable without `langgraph`
installed; only `graph/build.py` (this module's companion) actually wires
these functions into a `StateGraph`.

Responsibilities
-----------------
* `decide_next_agent` — pure policy: given state, which `AgentRole` should
  act next (or `AgentRole.HUMAN` if every agent has already contributed).
* `supervisor_node` — the actual supervisor node function (same
  `state -> dict` shape as every `BaseAgent.run`, Module 4). Advances
  `task_status` toward the chosen agent's working status, validates the
  transition, and records a `RoutingDecision`. Also owns the
  `AWAITING_HUMAN_APPROVAL` auto-advance for runs that don't require a
  real human (see "Design decisions").
* `route_after_supervisor` — the conditional-edge function: reads state
  *after* `supervisor_node` ran and returns the LangGraph node name (or an
  end sentinel) to transition to.

Design decisions
-----------------
* **Routing policy is deterministic and rule-based, not LLM-driven.** The
  supervisor decides "who hasn't contributed yet" by inspecting which
  `AgentRole`s appear in `state["messages"]`, rather than asking an LLM to
  choose. This is a conscious tradeoff for this module (see below), not an
  oversight — real multi-agent systems frequently prefer deterministic
  supervisor control flow over LLM-driven routing specifically because it's
  auditable and doesn't require a mocked/real LLM call just to decide
  "researcher happens before writer."
* **`supervisor_node` collapses `PENDING -> ROUTING -> {target}` into one
  call on the graph's first visit.** `TASK_STATUS_TRANSITIONS` (Module 2)
  only allows `PENDING -> ROUTING` directly, but the supervisor's very
  first turn also needs to pick the first working agent in the same call
  (there is no other node that would ever observe the intermediate
  `ROUTING` state in between). Both hops are validated via
  `validate_transition` even though only the final status is persisted —
  documented explicitly rather than silently skipping the intermediate
  check.
* **The `AWAITING_HUMAN_APPROVAL` auto-advance lives here, not in
  `WriterAgent`.** Module 4's `WriterAgent` report flagged this as a
  dependency: since `WRITING` cannot transition directly to `COMPLETED`
  (Module 2's table), *something* has to advance
  `AWAITING_HUMAN_APPROVAL -> COMPLETED` when
  `state["requires_human_approval"]` is `False`. That "something" is the
  supervisor, since it already owns every other `task_status` transition
  decision — keeping all lifecycle transitions in one place (SRP at the
  module level, not just the class level).
* **When `requires_human_approval` is `True`, this module does not
  implement a real pause/interrupt.** `supervisor_node` returns an update
  that leaves `task_status` unchanged (parked at
  `AWAITING_HUMAN_APPROVAL`), and `route_after_supervisor` sends execution
  to the end sentinel regardless. This means, honestly, **the
  human-in-the-loop case currently ends the graph without real human
  input** — a documented limitation, not a hidden one. Module 7 replaces
  this with an actual interrupt.
* **`route_after_supervisor` returns the string `"__end__"`, not
  `langgraph.graph.END`.** Keeping this module free of any `langgraph`
  import means its logic is testable in this sandbox; `graph/build.py`
  (which does import `langgraph`) maps `"__end__"` to the real `END`
  sentinel when wiring the conditional edges.

Tradeoffs
---------
* Because routing is purely presence-based ("has this agent role produced
  a message yet"), the current policy always runs
  Researcher → Analyst → Writer exactly once each, with no ability to loop
  back (e.g. "Analyst says findings are insufficient, route back to
  Researcher") or skip a step (e.g. "query is pure arithmetic, skip
  Researcher"). This is a real limitation of Module 5's scope, not a bug —
  see "Known Risks" in the Module 5 report for what a smarter policy would
  need.

Limitations
------------
* Never executed inside a real `StateGraph` (no `langgraph` in this
  sandbox). Its correctness is proven here via a framework-free simulation
  (see `tests/integration/test_graph_routing_integration.py`) that
  manually drives `supervisor_node` + agent `.run()` + `route_after_supervisor`
  in the same sequence LangGraph would — not a substitute for real
  `StateGraph` execution, but the closest verification available without
  network access.

Future improvements
--------------------
* Replace the fixed Researcher→Analyst→Writer policy with one that can
  loop back or skip steps based on findings quality/query type, likely
  alongside Module 8's error-recovery retry logic.
"""

from __future__ import annotations

from typing import Any

from agentic_research.core.contracts import RoutingDecision
from agentic_research.core.enums import AgentRole, TaskStatus
from agentic_research.core.state import GraphState, validate_transition

#: Maps the agent a task is being routed to, to the task_status that
#: agent's own `next_status`-validation (Module 4) expects to see.
_WORKING_STATUS_BY_AGENT: dict[AgentRole, TaskStatus] = {
    AgentRole.RESEARCHER: TaskStatus.RESEARCHING,
    AgentRole.ANALYST: TaskStatus.ANALYZING,
    AgentRole.WRITER: TaskStatus.WRITING,
}

#: The fixed order agents are expected to contribute in. See module
#: docstring "Tradeoffs" for why this is a linear policy, not a smart one.
_AGENT_ORDER: tuple[AgentRole, ...] = (
    AgentRole.RESEARCHER,
    AgentRole.ANALYST,
    AgentRole.WRITER,
)

#: Maps `task_status` (after supervisor_node has run) to the LangGraph node
#: name `route_after_supervisor` should send execution to.
_NODE_NAME_BY_STATUS: dict[TaskStatus, str] = {
    TaskStatus.RESEARCHING: "researcher",
    TaskStatus.ANALYZING: "analyst",
    TaskStatus.WRITING: "writer",
}

#: Sentinel returned by `route_after_supervisor` meaning "end the graph".
#: `graph/build.py` maps this to the real `langgraph.graph.END`.
END_SENTINEL = "__end__"


def decide_next_agent(state: GraphState) -> AgentRole:
    """Decide which agent should act next, based on who has already spoken.

    Args:
        state: The current graph state.

    Returns:
        The first `AgentRole` in `_AGENT_ORDER` that has not yet produced
        an `AgentMessage` in `state["messages"]`, or `AgentRole.HUMAN` if
        all of them have.
    """
    participated = {msg.agent for msg in state["messages"] if msg.agent is not None}
    for agent_role in _AGENT_ORDER:
        if agent_role not in participated:
            return agent_role
    return AgentRole.HUMAN


def _routing_reason(next_agent: AgentRole) -> str:
    """Build a short, human-readable justification for a routing decision."""
    return f"{next_agent.value} has not yet contributed to this task."


def _handle_awaiting_approval(state: GraphState) -> dict[str, Any]:
    """Handle the supervisor's turn when the task is parked awaiting approval.

    Args:
        state: The current graph state, with
            ``task_status == TaskStatus.AWAITING_HUMAN_APPROVAL``.

    Returns:
        If ``state["requires_human_approval"]`` is ``False``: an update
        auto-advancing to ``TaskStatus.COMPLETED`` with
        ``human_approval_status="approved"``. Otherwise: a minimal update
        that leaves ``task_status`` untouched (see module docstring,
        "Design decisions", for why this does not implement a real pause).
    """
    if state["requires_human_approval"]:
        return {"current_agent": AgentRole.SUPERVISOR}
    validate_transition(TaskStatus.AWAITING_HUMAN_APPROVAL, TaskStatus.COMPLETED)
    return {
        "task_status": TaskStatus.COMPLETED,
        "current_agent": AgentRole.SUPERVISOR,
        "human_approval_status": "approved",
    }


def supervisor_node(state: GraphState) -> dict[str, Any]:
    """The supervisor's turn: decide and record the next routing step.

    This is the real LangGraph node function `graph/build.py` will register
    for the ``"supervisor"`` node — same `state -> dict` shape as every
    `BaseAgent.run` (Module 4).

    Args:
        state: The current graph state.

    Returns:
        A partial state update: at minimum ``current_agent``; when routing
        to a working agent, also ``task_status`` and one new
        ``routing_history`` entry; when auto-completing past the approval
        gate, ``task_status`` and ``human_approval_status`` instead.

    Raises:
        InvalidStateTransitionError: If the computed transition is not
            legal per `TASK_STATUS_TRANSITIONS` (Module 2) — e.g. state was
            corrupted into an unexpected combination.
    """
    current_status = state["task_status"]

    if current_status == TaskStatus.AWAITING_HUMAN_APPROVAL:
        return _handle_awaiting_approval(state)

    next_agent = decide_next_agent(state)
    if next_agent == AgentRole.HUMAN:
        # Safety net: every agent already contributed but task_status isn't
        # AWAITING_HUMAN_APPROVAL yet (should not happen via normal flow,
        # since WriterAgent always sets that status -- see Module 4). No
        # state change; route_after_supervisor will end the graph.
        return {"current_agent": AgentRole.SUPERVISOR}

    target_status = _WORKING_STATUS_BY_AGENT[next_agent]

    if current_status == TaskStatus.PENDING:
        # Collapse PENDING -> ROUTING -> target into this one call; see
        # module docstring "Design decisions" for why both hops are still
        # validated.
        validate_transition(TaskStatus.PENDING, TaskStatus.ROUTING)
        validate_transition(TaskStatus.ROUTING, target_status)
    else:
        validate_transition(current_status, target_status)

    decision = RoutingDecision(
        from_agent=AgentRole.SUPERVISOR,
        to_agent=next_agent,
        reason=_routing_reason(next_agent),
    )
    return {
        "task_status": target_status,
        "current_agent": AgentRole.SUPERVISOR,
        "routing_history": [decision],
    }


def route_after_supervisor(state: GraphState) -> str:
    """Conditional-edge function: where should execution go after the supervisor?

    Args:
        state: The graph state *after* `supervisor_node` has already run
            and its update has been merged in.

    Returns:
        The LangGraph node name to transition to (``"researcher"``,
        ``"analyst"``, or ``"writer"``), or `END_SENTINEL` if there is
        nothing further to route to.
    """
    return _NODE_NAME_BY_STATUS.get(state["task_status"], END_SENTINEL)


def route_from_supervisor(state: GraphState) -> str:
    """Real-graph routing after the supervisor's turn (Module 9 addition).

    Extends `route_after_supervisor` by sending
    `TaskStatus.AWAITING_HUMAN_APPROVAL` to the `"human_approval"` node
    (Module 7's `human_approval_node`) instead of ending the graph — this
    is the concrete replacement for the dead-end `route_after_supervisor`
    left in place (see that function and `_handle_awaiting_approval`'s
    docstrings). Added here, alongside `route_after_supervisor`, rather
    than duplicated inside `graph/build.py`, so `graph/build.py` reuses one
    real, independently-testable routing function instead of an
    unverifiable inline copy.

    Args:
        state: The graph state *after* `supervisor_node` has already run.

    Returns:
        ``"human_approval"`` if the task is parked awaiting human review;
        otherwise whatever `route_after_supervisor` returns.
    """
    if state["task_status"] == TaskStatus.AWAITING_HUMAN_APPROVAL:
        return "human_approval"
    return route_after_supervisor(state)

