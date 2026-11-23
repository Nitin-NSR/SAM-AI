import pytest

from backend.tools.calculator import calculate


def test_calculator_observes_operator_precedence() -> None:
    assert calculate("2 + 3 * 4") == 14


def test_calculator_supports_parentheses_and_decimals() -> None:
    assert calculate("(8.5 - 2.5) / 2") == 3


@pytest.mark.parametrize("expression", ["__import__('os').system('whoami')", "2 ** 1000", "1 +"])
def test_calculator_rejects_unsafe_or_invalid_expressions(expression: str) -> None:
    with pytest.raises(ValueError):
        calculate(expression)
