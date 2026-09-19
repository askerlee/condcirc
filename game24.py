"""Game24 benchmark helpers compatible with tree-of-thought-llm's 24.csv."""

import ast
import csv
import re
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path


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


def format_prompt(puzzle: str | Sequence[int]) -> str:
    numbers = " ".join(map(str, parse_puzzle(puzzle)))
    return (
        f"Use each of the numbers {numbers} exactly once, together with +, -, *, /, "
        "and parentheses, to make 24. End with exactly one line in this form: "
        "Answer: <expression> = 24."
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