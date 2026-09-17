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
    total_seconds: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def average(self) -> float:
        if not self.scores:
            raise ValueError("Cannot average a method with no evaluation scores.")
        return sum(self.scores) / len(self.scores)

    @property
    def has_stats(self) -> bool:
        return bool(self.stats)


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
            seconds = run.get("seconds")
            if seconds is not None:
                if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
                    raise ValueError(
                        f"Run {run_index} of query {query_index} has invalid seconds."
                    )
                ratings.total_seconds += float(seconds)
            stats = run.get("stats")
            if stats is not None:
                if not isinstance(stats, dict):
                    raise ValueError(
                        f"Run {run_index} of query {query_index} has invalid stats."
                    )
                _add_stats(ratings.stats, stats)
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


def _add_stats(total: dict[str, Any], stats: dict[str, Any]) -> None:
    for key in ("same_top1", "rejected", "adaptive_rejected"):
        if key in stats:
            total[key] = total.get(key, 0) + stats[key]

    for key in ("recirculated_tokens", "adaptive_recirculated_tokens"):
        if key not in stats:
            continue
        values = stats[key]
        aggregate = total.setdefault(key, {"count": 0, "total": 0})
        aggregate["count"] += values["count"]
        aggregate["total"] += values["total"]

    if "rejected_by_gate" in stats:
        gates = total.setdefault("rejected_by_gate", {})
        for gate, count in stats["rejected_by_gate"].items():
            gates[gate] = gates.get(gate, 0) + count

    if "average_adaptive_recirculations" in stats:
        adaptive_count = stats.get("adaptive_recirculated_tokens", {}).get("count", 0)
        total["adaptive_recirculation_sum"] = total.get(
            "adaptive_recirculation_sum", 0.0
        ) + stats["average_adaptive_recirculations"] * adaptive_count
        total["adaptive_recirculation_count"] = total.get(
            "adaptive_recirculation_count", 0
        ) + adaptive_count


def format_method_ratings(methods: dict[str, MethodRatings]) -> str:
    sections = []
    for label, ratings in methods.items():
        section = (
            f"{label}\n"
            f"average_eval_model_rating = {ratings.average:.2f} "
            f"({len(ratings.scores)}/{ratings.total_runs} rated)\n"
            f"total_time = {ratings.total_seconds:.2f}s, "
            f"average_time_per_query = "
            f"{ratings.total_seconds / ratings.total_runs:.2f}s"
        )
        if ratings.has_stats:
            stats = ratings.stats
            recirculated = stats.get("recirculated_tokens", {"count": 0, "total": 0})
            adaptive = stats.get(
                "adaptive_recirculated_tokens", {"count": 0, "total": 0}
            )
            gates = ", ".join(
                f"{gate}={count}"
                for gate, count in stats.get("rejected_by_gate", {}).items()
            )
            adaptive_count = stats.get("adaptive_recirculation_count", 0)
            adaptive_average = (
                stats.get("adaptive_recirculation_sum", 0.0) / adaptive_count
                if adaptive_count
                else 0.0
            )
            section += (
                f"\nrecirculated_tokens = {recirculated['count']}/{recirculated['total']}, "
                f"same_top1 = {stats.get('same_top1', 0)}, "
                f"rejected = {stats.get('rejected', 0)}"
            )
            if gates:
                section += f", by gate: {gates}"
            section += (
                f"\nadaptive_recirculated_tokens = {adaptive['count']}/{adaptive['total']}, "
                f"rejected = {stats.get('adaptive_rejected', 0)}, "
                f"average_adaptive_recirculations = {adaptive_average:.2f}"
            )
        sections.append(section)
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
