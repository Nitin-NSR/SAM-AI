from datetime import datetime, timezone

from .calculator import calculate


def current_time() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_calculator(expression: str) -> str:
    return str(calculate(expression))
