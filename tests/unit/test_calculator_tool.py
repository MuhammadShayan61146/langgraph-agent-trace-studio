"""Unit tests for ``agentic_research.tools.calculator``."""

from __future__ import annotations

import pytest

from agentic_research.tools.calculator import CalculatorTool, safe_eval


# --------------------------------------------------------------------------- #
# safe_eval - correctness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2 + 3", 5),
        ("2 - 3", -1),
        ("2 * 3", 6),
        ("7 / 2", 3.5),
        ("7 // 2", 3),
        ("7 % 2", 1),
        ("2 ** 10", 1024),
        ("-5 + 3", -2),
        ("(2 + 3) * 4", 20),
        ("2 + 3 * 4", 14),
        ("-(2 + 3)", -5),
        ("+5", 5),
    ],
)
def test_safe_eval_correct_results(expression: str, expected: float) -> None:
    assert safe_eval(expression) == expected


def test_safe_eval_raises_on_division_by_zero() -> None:
    with pytest.raises(ZeroDivisionError):
        safe_eval("1 / 0")


# --------------------------------------------------------------------------- #
# safe_eval - security boundary (the whole point of this tool)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "malicious_expression",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "[x for x in range(10)]",
        "(1).__class__",
        "1 if True else 2",
        "a + 1",  # names are not allowed
        "print(1)",
    ],
)
def test_safe_eval_rejects_disallowed_syntax(malicious_expression: str) -> None:
    with pytest.raises(ValueError):
        safe_eval(malicious_expression)


def test_safe_eval_rejects_invalid_syntax() -> None:
    with pytest.raises(ValueError):
        safe_eval("2 +* 3")


def test_safe_eval_rejects_non_numeric_constant() -> None:
    with pytest.raises(ValueError):
        safe_eval("'a string'")


# --------------------------------------------------------------------------- #
# CalculatorTool via BaseTool.execute()
# --------------------------------------------------------------------------- #
def test_calculator_tool_success() -> None:
    result = CalculatorTool().execute(expression="2 + 2")
    assert result.success is True
    assert result.output == 4


def test_calculator_tool_reports_failure_without_raising() -> None:
    result = CalculatorTool().execute(expression="__import__('os')")
    assert result.success is False
    assert result.output is None
    assert result.error_message is not None


def test_calculator_tool_missing_argument_reported_as_failure() -> None:
    result = CalculatorTool().execute()
    assert result.success is False


def test_calculator_tool_schema_matches_expected_shape() -> None:
    schema = CalculatorTool().to_anthropic_schema()
    assert schema["name"] == "calculator"
    assert schema["input_schema"]["required"] == ["expression"]
