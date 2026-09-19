"""Game24 benchmark helpers compatible with tree-of-thought-llm's 24.csv."""

import ast
import csv
import re
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
# game countdown:
# https://github.com/holsee/CountdownNumbers

COUNTDOWN_NUMBERS = frozenset((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 75, 100))


def validate_countdown_puzzle(target: int, numbers: Sequence[int]) -> None:
    if type(target) is not int or not 100 <= target <= 999:
        raise ValueError("Countdown target must be an integer from 100 to 999.")
    if len(numbers) != 6 or any(
        type(number) is not int or number not in COUNTDOWN_NUMBERS
        for number in numbers
    ):
        raise ValueError(
            "Countdown requires six numbers from "
            "{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 75, 100}."
        )


def solve_countdown(target: int, numbers: Sequence[int]) -> str:
    """Return an infix expression whose value is closest to ``target``.

    The expression uses each of the six Countdown numbers at most once and uses
    only the four common arithmetic operations over exact rational values.
    """
    validate_countdown_puzzle(target, numbers)

    expressions: dict[int, dict[Fraction, str]] = {
        1 << index: {Fraction(number): str(number)}
        for index, number in enumerate(numbers)
    }
    best_value = Fraction(numbers[0])
    best_expression = str(numbers[0])
    target_value = Fraction(target)

    def consider(value: Fraction, expression: str) -> str | None:
        nonlocal best_value, best_expression
        if abs(value - target_value) < abs(best_value - target_value):
            best_value = value
            best_expression = expression
        return expression if value == target_value else None

    for mask in range(1, 1 << len(numbers)):
        if mask in expressions:
            exact = consider(*next(iter(expressions[mask].items())))
            if exact is not None:
                return exact
            continue
        values: dict[Fraction, str] = {}
        subset = (mask - 1) & mask
        while subset:
            other = mask ^ subset
            if other and subset < other:
                for left_value, left_expression in expressions[subset].items():
                    for right_value, right_expression in expressions[other].items():
                        candidates = (
                            (left_value + right_value, f"({left_expression} + {right_expression})"),
                            (left_value * right_value, f"({left_expression} * {right_expression})"),
                            (left_value - right_value, f"({left_expression} - {right_expression})"),
                            (right_value - left_value, f"({right_expression} - {left_expression})"),
                        )
                        if right_value:
                            candidates += ((
                                left_value / right_value,
                                f"({left_expression} / {right_expression})",
                            ),)
                        if left_value:
                            candidates += ((
                                right_value / left_value,
                                f"({right_expression} / {left_expression})",
                            ),)
                        for value, expression in candidates:
                            values.setdefault(value, expression)
            subset = (subset - 1) & mask
        expressions[mask] = values
        for value, expression in values.items():
            exact = consider(value, expression)
            if exact is not None:
                return exact
    return best_expression


def parse_puzzle(puzzle: str | Sequence[int]) -> tuple[int, int, int, int]:
    values = tuple(map(int, puzzle.split())) if isinstance(puzzle, str) else tuple(puzzle)
    if len(values) != 4 or any(value < 0 for value in values):
        raise ValueError("A Game24 puzzle must contain exactly four nonnegative integers.")
    return values


def load_puzzles(path: Path) -> tuple[tuple[int, int, int, int], ...]:
    with path.open(newline="", encoding="utf-8") as file:
        if file.read(512).lstrip().lower().startswith("<!doctype html"):
            raise ValueError(
                f"{path} is a GitHub HTML page, not a Game24 CSV. Download "
                "https://raw.githubusercontent.com/princeton-nlp/"
                "tree-of-thought-llm/master/src/tot/data/24/24.csv instead."
            )
        file.seek(0)
        rows = csv.DictReader(file)
        if rows.fieldnames is None or "Puzzles" not in rows.fieldnames:
            raise ValueError(f"{path} must be a Game24 CSV with a Puzzles column.")
        puzzles = tuple(parse_puzzle(row["Puzzles"]) for row in rows)
    if not puzzles:
        raise ValueError(f"{path} contains no Game24 puzzles.")
    return puzzles


def load_countdown_puzzles(path: Path) -> tuple[tuple[int, tuple[int, ...]], ...]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        if rows.fieldnames is None or not {"numbers", "target"} <= set(rows.fieldnames):
            raise ValueError(
                f"{path} must be a Countdown CSV with numbers and target columns."
            )
        puzzles = []
        for row in rows:
            try:
                target = int(row["target"])
                numbers = tuple(map(int, row["numbers"].split(",")))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path} contains an invalid Countdown puzzle.") from error
            validate_countdown_puzzle(target, numbers)
            puzzles.append((target, numbers))
    if not puzzles:
        raise ValueError(f"{path} contains no Countdown puzzles.")
    return tuple(puzzles)


def format_prompt(puzzle: str | Sequence[int]) -> str:
    numbers = " ".join(map(str, parse_puzzle(puzzle)))
    return (
        f"Use each of the numbers {numbers} exactly once, together with +, -, *, /, "
        "and parentheses, to make 24. End with exactly one line in this form: "
        "Answer: <expression> = 24."
    )


def format_countdown_prompt(target: int, numbers: Sequence[int]) -> str:
    validate_countdown_puzzle(target, numbers)
    values = " ".join(map(str, numbers))
    return (
        f"Target: {target}. Numbers: {values}. Use each number at most once, "
        "with +, -, *, /, and parentheses, to make the target or get as close "
        "as possible. You may use fewer than six numbers. End with exactly one "
        "line in this form: Answer: <expression> = <value>."
    )


def _answer_expression(output: str) -> str | None:
    for line in reversed(output.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        candidate = re.sub(r"^answer\s*:\s*", "", candidate, flags=re.IGNORECASE)
        if "=" in candidate:
            candidate, result = candidate.rsplit("=", 1)
            if result.strip().rstrip(".") != "24":
                return None
        return candidate.strip()
    return None


def _final_answer_expression(output: str) -> str | None:
    for line in reversed(output.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        candidate = re.sub(r"^answer\s*:\s*", "", candidate, flags=re.IGNORECASE)
        return candidate.rsplit("=", 1)[0].strip()
    return None


def _evaluate_expression(expression: str) -> tuple[Fraction, tuple[int, ...]]:
    tree = ast.parse(expression, mode="eval")
    numbers: list[int] = []

    def evaluate(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            numbers.append(node.value)
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand)
        if not isinstance(node, ast.BinOp):
            raise ValueError("unsupported expression")
        left = evaluate(node.left)
        right = evaluate(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        raise ValueError("unsupported operator")

    return evaluate(tree.body), tuple(numbers)


def is_solution(puzzle: str | Sequence[int], output: str) -> bool:
    expression = _answer_expression(output)
    if expression is None:
        return False
    try:
        value, numbers = _evaluate_expression(expression)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return False
    return sorted(numbers) == sorted(parse_puzzle(puzzle)) and value == 24


def score_countdown_output(
    target: int, numbers: Sequence[int], output: str
) -> Fraction | None:
    """Return a valid model answer's exact value, or ``None`` when invalid."""
    validate_countdown_puzzle(target, numbers)
    expression = _final_answer_expression(output)
    if expression is None:
        return None
    try:
        value, used_numbers = _evaluate_expression(expression)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None
    remaining = list(numbers)
    for number in used_numbers:
        try:
            remaining.remove(number)
        except ValueError:
            return None
    return value