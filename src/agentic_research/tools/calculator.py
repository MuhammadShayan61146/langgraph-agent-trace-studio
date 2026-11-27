"""Calculator tool: safe arithmetic expression evaluation.

Purpose
-------
Gives agents a genuine, fully offline, fully verifiable tool — no network,
no LLM, no mocking required. This is the one tool in Module 3 that reaches
100% "Verified" status end-to-end (not just its wiring, but its actual
computed output).

Responsibilities
-----------------
* Parse and evaluate arithmetic expressions (``+ - * / // % **`` and
  parentheses) without using Python's ``eval``, which would execute
  arbitrary code.

Design decisions
-----------------
* **AST-based evaluation, never ``eval()``.** The expression string is
  parsed into a Python AST (`ast.parse(expr, mode="eval")`) and walked by
  hand, accepting only numeric constants and a fixed allow-list of binary/
  unary operators. Any other node type (attribute access, function calls,
  subscripting, names/imports, comprehensions, ...) raises immediately.
  This is a deliberate security boundary: even though this expression
  comes from a user query relayed through an LLM rather than typed
  directly by an attacker, tool inputs must be treated as untrusted, since
  a compromised or manipulated LLM response is a realistic threat model for
  any tool-calling system.
* **The evaluator is a standalone function (`safe_eval`), not a method.**
  This lets it be unit tested directly, independent of the `BaseTool`
  wrapping, and reused if another future tool needs arithmetic evaluation
  without duplicating the AST-walking logic (DRY).

Tradeoffs
---------
* No support for functions like `sqrt`/`sin` or named constants like `pi`
  — deliberately out of scope to keep the allow-list small and the security
  argument easy to audit. Extending it (e.g. adding `math.sqrt`) is
  straightforward but each addition must be a conscious allow-list
  decision, not a blanket "allow function calls."

Limitations
------------
* Integer-only exponents with very large magnitude (e.g. `2 ** 100000`)
  are not separately guarded against and could be slow/memory-heavy. Not a
  security hole (still bounded, deterministic computation) but worth
  flagging; a real production tool would add a magnitude/time guard.

Future improvements
--------------------
* Add an explicit evaluation timeout or exponent/operand magnitude cap if
  this tool is ever exposed to untrusted high-volume input in production.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable

from agentic_research.tools.base import BaseTool, ToolParameter

#: Allow-listed binary operators. Anything not in this map is rejected.
_BINARY_OPERATORS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

#: Allow-listed unary operators (only sign flipping is needed).
_UNARY_OPERATORS: dict[type, Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval(expression: str) -> float:
    """Evaluate a simple arithmetic expression without using ``eval()``.

    Supports ``+ - * / // % **``, unary +/-, parentheses, and int/float
    literals. Anything else (names, calls, attribute access, comparisons,
    etc.) raises ``ValueError``.

    Args:
        expression: The arithmetic expression to evaluate, e.g.
            ``"(2 + 3) * 4"``.

    Returns:
        The numeric result.

    Raises:
        ValueError: If the expression contains disallowed syntax, or is
            not valid Python expression syntax at all.
        ZeroDivisionError: If the expression divides by zero.
    """
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Not a valid arithmetic expression: {expression!r}") from exc
    return _eval_node(parsed.body)


def _eval_node(node: ast.expr) -> float:
    """Recursively evaluate one AST node, rejecting anything not allow-listed.

    Args:
        node: An AST expression node from a ``mode="eval"`` parse tree.

    Returns:
        The numeric value of this subexpression.

    Raises:
        ValueError: If ``node`` (or any child) is not one of: a numeric
            constant, an allow-listed binary op, an allow-listed unary op,
            or a parenthesized grouping thereof.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"Disallowed constant: {node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp):
        op_fn = _BINARY_OPERATORS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Disallowed operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.left), _eval_node(node.right))

    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPERATORS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Disallowed unary operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.operand))

    raise ValueError(f"Disallowed expression element: {type(node).__name__}")


class CalculatorTool(BaseTool):
    """A tool that evaluates arithmetic expressions using `safe_eval`."""

    name = "calculator"
    description = (
        "Evaluate an arithmetic expression and return the numeric result. "
        "Supports +, -, *, /, //, %, **, unary sign, and parentheses."
    )
    parameters = [
        ToolParameter(
            name="expression",
            type="string",
            description="The arithmetic expression to evaluate, e.g. '(2 + 3) * 4'.",
            required=True,
        )
    ]

    def _run(self, **kwargs: Any) -> float:
        """Evaluate the ``expression`` argument via `safe_eval`.

        Args:
            **kwargs: Must contain ``expression`` (validated by
                `BaseTool.execute` before this runs).

        Returns:
            The numeric result of evaluating the expression.
        """
        expression: str = kwargs["expression"]
        return safe_eval(expression)
