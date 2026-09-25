"""Sudoku benchmark loading, prompting, and scoring."""
# https://huggingface.co/datasets/sapientinc/sudoku-extreme

import csv
import io
import json
import re
from collections.abc import Sequence
from pathlib import Path
from urllib.request import urlopen


SUDOKU_BOX_SHAPES = {4: (2, 2), 6: (2, 3), 9: (3, 3)}


def parse_puzzle(
    puzzle: Sequence[Sequence[object]], placeholder: object = 0
) -> tuple[tuple[int, ...], ...]:
    rows = tuple(tuple(row) for row in puzzle)
    grid_size = len(rows)
    if grid_size not in SUDOKU_BOX_SHAPES or any(
        len(row) != grid_size for row in rows
    ):
        raise ValueError("A Sudoku puzzle must be a 4x4, 6x6, or 9x9 square grid.")

    normalized_rows = []
    for row in rows:
        normalized_row = []
        for value in row:
            if value == placeholder:
                normalized_row.append(0)
            elif type(value) is int and 1 <= value <= grid_size:
                normalized_row.append(value)
            else:
                raise ValueError(
                    f"Sudoku cells must be the placeholder or integers from 1 to {grid_size}."
                )
        normalized_rows.append(tuple(normalized_row))
    return tuple(normalized_rows)


def load_puzzles(path: str | Path) -> tuple[tuple[tuple[int, ...], ...], ...]:
    puzzles = []
    if str(path).endswith(".csv"):
        if str(path).startswith("https://"):
            source = io.TextIOWrapper(urlopen(str(path)), encoding="utf-8")
        else:
            source = Path(path).open(encoding="utf-8", newline="")
        with source as file:
            for line_number, record in enumerate(csv.DictReader(file), start=2):
                question = record.get("question")
                if question is None or len(question) != 81 or any(
                    cell not in ".123456789" for cell in question
                ):
                    raise ValueError(
                        f"{path} contains an invalid Sudoku CSV record on line {line_number}."
                    )
                puzzles.append(
                    parse_puzzle(
                        [
                            [0 if cell == "." else int(cell) for cell in question[row : row + 9]]
                            for row in range(0, 81, 9)
                        ]
                    )
                )
        if not puzzles:
            raise ValueError(f"{path} contains no Sudoku puzzles.")
        return tuple(puzzles)
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                config = record["config"]
                puzzle = parse_puzzle(
                    record["puzzle"], config.get("placeholder", 0)
                )
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{path} contains an invalid Sudoku4LLM JSONL record on line {line_number}."
                ) from error
            if config.get("grid_size") != len(puzzle):
                raise ValueError(
                    f"{path} Sudoku4LLM record on line {line_number} has a mismatched grid_size."
                )
            puzzles.append(puzzle)
    if not puzzles:
        raise ValueError(f"{path} contains no Sudoku4LLM puzzles.")
    return tuple(puzzles)


def format_prompt(puzzle: Sequence[Sequence[object]]) -> str:
    grid = parse_puzzle(puzzle)
    grid_size = len(grid)
    box_rows, box_columns = SUDOKU_BOX_SHAPES[grid_size]
    formatted_grid = "\n".join(" ".join(map(str, row)) for row in grid)
    return (
        f"Solve this {grid_size}x{grid_size} Sudoku. Zeros are empty cells. "
        f"Each row and column, and each {box_rows}x{box_columns} box, must contain "
        f"every number from 1 to {grid_size} exactly once. Preserve all nonzero clues. "
        "End with `Answer:` followed by the completed grid as exactly one space-separated "
        "row per line.\n\n"
        f"{formatted_grid}"
    )


def _parse_answer_row(line: str, grid_size: int) -> tuple[int, ...] | None:
    candidate = line.strip().strip("|").strip().strip("[]")
    values = re.split(r"\s*(?:,|\s)\s*", candidate)
    if len(values) != grid_size or any(not value.isdecimal() for value in values):
        return None
    row = tuple(map(int, values))
    return row if all(1 <= value <= grid_size for value in row) else None


def _answer_grid(output: str, grid_size: int) -> tuple[tuple[int, ...], ...] | None:
    rows = [_parse_answer_row(line, grid_size) for line in output.splitlines()]
    for start in range(len(rows) - grid_size, -1, -1):
        candidate = rows[start : start + grid_size]
        if all(row is not None for row in candidate):
            return tuple(candidate)  # type: ignore[arg-type]
    return None


def is_solution(puzzle: Sequence[Sequence[object]], output: str) -> bool:
    grid = parse_puzzle(puzzle)
    grid_size = len(grid)
    answer = _answer_grid(output, grid_size)
    if answer is None:
        return False
    required_values = set(range(1, grid_size + 1))
    if any(set(row) != required_values for row in answer):
        return False
    if any(
        {answer[row][column] for row in range(grid_size)} != required_values
        for column in range(grid_size)
    ):
        return False
    box_rows, box_columns = SUDOKU_BOX_SHAPES[grid_size]
    for row_start in range(0, grid_size, box_rows):
        for column_start in range(0, grid_size, box_columns):
            values = {
                answer[row][column]
                for row in range(row_start, row_start + box_rows)
                for column in range(column_start, column_start + box_columns)
            }
            if values != required_values:
                return False
    return all(
        clue == 0 or answer[row][column] == clue
        for row, puzzle_row in enumerate(grid)
        for column, clue in enumerate(puzzle_row)
    )