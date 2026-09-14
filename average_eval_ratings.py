from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class MethodRatings:
    scores: list[float] = field(default_factory=list)
    total_runs: int = 0

    @property
    def average(self) -> float:
        if not self.scores:
            raise ValueError("Cannot average a method with no evaluation scores.")
        return sum(self.scores) / len(self.scores)


def load_method_ratings(path: Path) -> dict[str, MethodRatings]:
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(records, list):
        raise ValueError("Expected the result JSON to contain a top-level list.")

    methods: dict[str, MethodRatings] = {}
    for query_index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or not isinstance(record.get("runs"), list):
            raise ValueError(f"Query {query_index} does not contain a runs list.")
        for run_index, run in enumerate(record["runs"], start=1):
            if not isinstance(run, dict) or not isinstance(run.get("label"), str):
                raise ValueError(
                    f"Run {run_index} of query {query_index} has no method label."
                )
            ratings = methods.setdefault(run["label"], MethodRatings())
            ratings.total_runs += 1
            if "score" not in run:
                continue
            score = run["score"]
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not 0 <= score <= 10
            ):
                raise ValueError(
                    f"Run {run_index} of query {query_index} has an invalid score: "
                    f"{score!r}."
                )
            ratings.scores.append(float(score))

    rated_methods = {
        label: ratings for label, ratings in methods.items() if ratings.scores
    }
    if not rated_methods:
        raise ValueError(f"No evaluation scores found in {path}.")
    return rated_methods


def format_method_ratings(methods: dict[str, MethodRatings]) -> str:
    sections = []
    for label, ratings in methods.items():
        sections.append(
            f"{label}\n"
            f"average_eval_model_rating = {ratings.average:.2f} "
            f"({len(ratings.scores)}/{ratings.total_runs} rated)"
        )
    return "\n\n".join(sections)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the average evaluation-model rating for each method in an "
            "existing result JSON file."
        )
    )
    parser.add_argument("result_json", type=Path, help="Path to the result JSON file.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        methods = load_method_ratings(args.result_json)
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(format_method_ratings(methods))


if __name__ == "__main__":
    main()
