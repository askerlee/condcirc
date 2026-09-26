from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from types import MethodType


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _cache_packages_distributions() -> None:
    """Importing transformers calls packages_distributions(), which stats every file in
    site-packages: seconds locally, many minutes on a network filesystem."""
    key = hashlib.sha1(f"{sys.prefix}|{sys.version}".encode()).hexdigest()[:16]
    path = Path.home() / ".cache" / "condcirc" / f"packages_distributions-{key}.json"
    try:
        mapping = json.loads(path.read_text())
    except (OSError, ValueError):
        mapping = importlib.metadata.packages_distributions()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(mapping))
        except OSError:
            pass
    importlib.metadata.packages_distributions = lambda: mapping


_cache_packages_distributions()

import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

from tasks.bbeh import format_prompt as format_bbeh_prompt  # noqa: E402
from tasks.bbeh import is_correct as is_bbeh_correct  # noqa: E402
from tasks.bbeh import load_examples as load_bbeh_examples  # noqa: E402
from tasks.game24 import format_prompt as format_game24_prompt  # noqa: E402
from tasks.game24 import format_countdown_prompt  # noqa: E402
from tasks.game24 import is_solution as is_game24_solution  # noqa: E402
from tasks.game24 import load_countdown_puzzles  # noqa: E402
from tasks.game24 import load_puzzles as load_game24_puzzles  # noqa: E402
from tasks.game24 import score_countdown_output  # noqa: E402
from tasks.knowedit import format_prompt as format_knowedit_prompt  # noqa: E402
from tasks.knowedit import load_examples as load_knowedit_examples  # noqa: E402
from recirculation import (  # noqa: E402
    AdjacentLayerSimilarityStats,
    RecirculationConfig,
    SimilarityStats,
    recirculate,
)
from tasks.sudoku import format_prompt as format_sudoku_prompt  # noqa: E402
from tasks.sudoku import is_solution as is_sudoku_solution  # noqa: E402
from tasks.sudoku import load_puzzles as load_sudoku_puzzles  # noqa: E402


EXAMPLE_QUERIES = (
    "Explain why the daytime sky is blue but sunsets often appear red. Connect the explanation to scattering, wavelength, and the distance sunlight travels through the atmosphere.",
    "A city wants to reduce downtown traffic without making commuting harder for low-income workers. Compare congestion pricing, improved public transit, and parking restrictions, then recommend a phased policy with safeguards and measurable success criteria.",
    "Maya manages a project originally due Friday. The client moves it to Wednesday, an engineer reports a two-day blocker, and a required reviewer is unavailable Tuesday. Develop a realistic recovery plan, identify assumptions, and explain what Maya should communicate to each stakeholder.",
    "A company claims that productivity increased after employees returned to the office, so remote work must reduce productivity. Critique this inference, propose plausible confounders, and design a stronger evaluation that could support a causal conclusion.",
    "Design a fair procedure for allocating five emergency shelter beds among twelve eligible people when needs differ and information is incomplete. Explain the values behind your procedure and how appeals or new evidence should be handled.",
    "Fred took his fishing pole to the bank of a river. Later, a friend texted that she would meet him at the bank to discuss a loan. Analyze the ambiguity, explain which interpretation each person may hold, and propose a message that prevents a costly misunderstanding.",
    "All roses are flowers, some flowers fade quickly, and no quickly fading plant survives a frost. Explain exactly what can and cannot be inferred about roses, then give two additional premises that would support different conclusions.",
    "A small software team must choose between shipping a fragile feature this week or delaying it for testing while a competitor is launching a similar product. Build a decision framework, evaluate the main risks, and recommend a course of action under clearly stated assumptions.",
    "Plan how to prepare tea for six guests when there is one kettle, four clean mugs, two dirty mugs, and one guest avoids caffeine. Include ordering, resource constraints, and a contingency if the kettle stops working.",
    "A store is considering a 25% discount followed by a loyalty reward, but margins are thin and customers respond differently to promotions. Explain how the store should evaluate profitability, customer behavior, and long-term effects before choosing a promotion design.",
    "A coastal town must decide whether to rebuild a storm-damaged seawall, restore wetlands, or relocate the most exposed homes. Compare the options across cost, resilience, fairness, and uncertainty, then propose a decision process that can adapt as conditions change.",
    "A hospital has fewer intensive-care beds than patients likely to need them during an outbreak. Design a transparent allocation policy, explain how it handles changing prognoses and ties, and identify safeguards against bias and avoidable harm.",
    "Two departments report conflicting results from the same customer survey: one says satisfaction improved, while the other says complaints became more severe. Explain how both claims could be true and outline an analysis that would reconcile the evidence.",
    "A teacher discovers that students are using generative AI for homework, but the school has no clear policy. Develop a response that supports learning, treats students fairly, and distinguishes acceptable assistance from work that misrepresents understanding.",
    "An old bridge is still considered safe but requires increasingly frequent repairs. Compare continued maintenance, major rehabilitation, and replacement while accounting for disruption, uncertain future demand, public safety, and budget constraints.",
    "A neighborhood wants more housing but disagrees about building height, affordability requirements, parking, and preservation of local businesses. Propose a negotiation framework and a compromise plan, including who bears each cost and how outcomes should be measured.",
    "A research team finds a statistically significant effect that is much smaller than expected and disappears under one reasonable analysis choice. Interpret the result, identify what should be reported, and recommend the next study without reducing the decision to a single p-value.",
    "A family must choose between caring for an aging relative at home, hiring in-home support, or moving them to assisted living. Build a respectful decision process that considers autonomy, safety, finances, caregiver capacity, and how the plan should be revisited over time.",
    "A news platform wants to reduce misinformation without suppressing legitimate disagreement or breaking-news updates that later change. Design a moderation approach that combines labels, distribution rules, appeals, and evidence standards, then explain its likely failure modes.",
    "A manufacturer can lower emissions by replacing equipment now, purchasing cleaner electricity, or waiting for a promising technology still under development. Recommend a staged strategy using plausible assumptions about cost, risk, and regulation, and specify signals that would trigger a change in course.",
    # r"> Using each of the numbers $2,3,4,6$ exactly once, together with $+,-,\times,\div$ and parentheses, make 24. Give exactly one solution and explain your reasoning.",
)

REJECTION_GATES = (
    "margin-narrowed",
    "post-margin-min",
    "post-margin-ratio",
    "post-margin-max",
    "cosine",
    "rank",
)
GPU_MEMORY_RESERVE_BYTES = 4 * 1024**3


def summarize_recirculation_stats(
    recirculated_flags: Sequence[bool],
    rejected_flags: Sequence[bool],
    adaptive_recirculated_flags: Sequence[bool],
    adaptive_rejected_flags: Sequence[bool],
    adaptive_recirculation_counts: Sequence[int],
    final_pass_same_top1_flags: Sequence[bool],
    rejection_reasons: Sequence[Sequence[str]],
) -> dict[str, Any]:
    total_tokens = len(recirculated_flags)
    adaptive_counts = [count for count in adaptive_recirculation_counts if count > 0]
    average_adaptive_recirculations = (
        sum(adaptive_counts) / len(adaptive_counts) if adaptive_counts else 0.0
    )
    return {
        "recirculated_tokens": {
            "count": sum(recirculated_flags),
            "total": total_tokens,
        },
        "same_top1": sum(final_pass_same_top1_flags),
        "rejected": sum(rejected_flags),
        "rejected_by_gate": {
            gate: sum(gate in reasons for reasons in rejection_reasons)
            for gate in REJECTION_GATES
        },
        "adaptive_recirculated_tokens": {
            "count": sum(adaptive_recirculated_flags),
            "total": total_tokens,
        },
        "adaptive_rejected": sum(adaptive_rejected_flags),
        "average_adaptive_recirculations": round(
            average_adaptive_recirculations, 2
        ),
    }


def format_run_stats(stats: dict[str, Any]) -> tuple[str, ...]:
    recirculated = stats["recirculated_tokens"]
    rejected_by_gate = stats["rejected_by_gate"]
    adaptive = stats["adaptive_recirculated_tokens"]
    gates = ", ".join(
        f"{gate}={rejected_by_gate[gate]}" for gate in REJECTION_GATES
    )
    return (
        f"recirculated_tokens = {recirculated['count']}/{recirculated['total']}, "
        f"same_top1 = {stats['same_top1']}, rejected = {stats['rejected']}, "
        f"by gate: {gates}",
        f"adaptive_recirculated_tokens = {adaptive['count']}/{adaptive['total']}, "
        f"rejected = {stats['adaptive_rejected']}, "
        "average_adaptive_recirculations = "
        f"{stats['average_adaptive_recirculations']:.2f}",
    )


def format_average_eval_rating(scores: Sequence[float]) -> str:
    if not scores:
        raise ValueError("Cannot average an empty collection of evaluation scores.")
    return f"average_eval_model_rating = {sum(scores) / len(scores):.2f}"


def teacher_forced_token_accuracy(
    prompt_ids: Tensor,
    target_ids: Tensor,
    step: Callable[[Tensor], Tensor],
    on_top_two: Callable[[int, tuple[tuple[int, float], ...]], None] | None = None,
) -> float:
    if prompt_ids.shape[0] != 1 or target_ids.shape[0] != 1 or target_ids.shape[1] == 0:
        raise ValueError("KnowEdit scoring requires one prompt and a nonempty target.")
    logits = step(prompt_ids)
    correct = 0
    for token_index in range(target_ids.shape[1]):
        target_token = target_ids[:, token_index:token_index + 1]
        next_logits = logits[0, -1, :].float()
        if on_top_two is not None:
            top_two = torch.topk(next_logits, k=2)
            probabilities = torch.softmax(next_logits, dim=-1)
            on_top_two(
                token_index,
                tuple(
                    (int(token_id), float(probabilities[token_id]))
                    for token_id in top_two.indices.tolist()
                ),
            )
        correct += int(next_logits.argmax().item() == target_token.item())
        if token_index + 1 < target_ids.shape[1]:
            logits = step(target_token)
    return correct / target_ids.shape[1]


def aggregate_recirculation_stats(
    stats_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    adaptive_count = sum(
        stats["adaptive_recirculated_tokens"]["count"] for stats in stats_records
    )
    adaptive_recirculation_total = sum(
        stats["average_adaptive_recirculations"]
        * stats["adaptive_recirculated_tokens"]["count"]
        for stats in stats_records
    )
    return {
        "recirculated_tokens": {
            "count": sum(
                stats["recirculated_tokens"]["count"] for stats in stats_records
            ),
            "total": sum(
                stats["recirculated_tokens"]["total"] for stats in stats_records
            ),
        },
        "same_top1": sum(stats["same_top1"] for stats in stats_records),
        "rejected": sum(stats["rejected"] for stats in stats_records),
        "rejected_by_gate": {
            gate: sum(
                stats["rejected_by_gate"].get(gate, 0)
                for stats in stats_records
            )
            for gate in REJECTION_GATES
        },
        "adaptive_recirculated_tokens": {
            "count": adaptive_count,
            "total": sum(
                stats["adaptive_recirculated_tokens"]["total"]
                for stats in stats_records
            ),
        },
        "adaptive_rejected": sum(
            stats["adaptive_rejected"] for stats in stats_records
        ),
        "average_adaptive_recirculations": round(
            adaptive_recirculation_total / adaptive_count if adaptive_count else 0.0,
            2,
        ),
    }


def parse_results(path: Path) -> list[tuple[str, list[tuple[str, str]]]]:
    text = path.read_text(encoding="utf-8")
    try:
        records = json.loads(text)
    except json.JSONDecodeError:
        records = None
    if records is not None:
        return [
            (
                record["prompt"],
                [(run["label"], run["output"]) for run in record["runs"]],
            )
            for record in records
        ]

    sections = re.split(r"\n=== Query(?: \d+)? ===\n", text)[1:]
    parsed: list[tuple[str, list[tuple[str, str]]]] = []
    header = re.compile(r"\n=== (Baseline|Ablation): ([^\n]+) \([^\n]+\) ===\n")
    for section in sections:
        query, *run_parts = header.split(section)
        runs = [
            (f"{run_parts[index]}: {run_parts[index + 1]}", run_parts[index + 2].strip())
            for index in range(0, len(run_parts), 3)
        ]
        if not runs:
            raise ValueError("A query has no parseable method results.")
        parsed.append((query.strip(), runs))
    if not parsed:
        raise ValueError(f"No query sections found in {path}.")
    return parsed


def build_evaluation_prompt(
    query: str, answers: Sequence[tuple[str, str]]
) -> str:
    formatted_answers = "\n\n".join(
        f"METHOD {index}: {label}\n{answer}"
        for index, (label, answer) in enumerate(answers, start=1)
    )
    return f"""Evaluate the following excerpts from answers to the query.
These excerpts may end abruptly because the source answer was truncated. Treat
each excerpt as the complete evidence available for grading, not as an incomplete
submission. Never deduct points because a later section, recommendation, caveat,
or requested item is absent after the visible ending. Do not infer that the answer
would have addressed anything beyond the excerpt. Score only the quality of what
is visible: factual correctness, clarity, internal reasoning, and usefulness of
the visible material. Assess coverage only of the claims or subtopics actually
present in the excerpt. Do not reward length or formatting.
Return JSON only, as an array in the same order, with objects containing exactly
the integer field method, the floating-point field score, and a brief string field
rationale. Scores may use one decimal place, such as 8.5.

QUERY:
{query}

ANSWERS:
{formatted_answers}
"""


def parse_evaluation(text: str, method_count: int) -> list[dict[str, Any]]:
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Evaluator did not return a JSON array: {text!r}")
    evaluations = json.loads(match.group(0))
    if not isinstance(evaluations, list) or len(evaluations) != method_count:
        raise ValueError("Evaluator returned the wrong number of scores.")
    for expected_method, evaluation in enumerate(evaluations, start=1):
        if evaluation.get("method") != expected_method:
            raise ValueError("Evaluator returned methods out of order.")
        score = evaluation.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0 <= score <= 10
        ):
            raise ValueError("Evaluator scores must be numbers from 0 to 10.")
        evaluation["score"] = float(score)
    return evaluations


def openai_evaluate(prompt: str, model: str, api_key: str, base_url: str) -> str:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API request failed ({error.code}): {detail}") from error
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("OpenAI API response did not contain message content.") from error
    print(f"OpenAI evaluator response:\n{content}\n", flush=True)
    return content


def evaluate_single_answer(
    prompt: str,
    label: str,
    answer: str,
    *,
    eval_provider: str,
    model: nn.Module,
    tokenizer: Any,
    input_device: Any,
    api_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    evaluation_model: str = "gpt-5.6-sol",
) -> dict[str, Any]:
    evaluation_prompt = build_evaluation_prompt(prompt, [(label, answer)])
    if eval_provider == "openai":
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY before using --eval-provider openai.")
        evaluation_text = openai_evaluate(
            evaluation_prompt,
            evaluation_model,
            api_key,
            base_url,
        )
    else:
        evaluation_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": evaluation_prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(input_device)
        with torch.inference_mode():
            evaluation_ids = model.generate(
                input_ids=evaluation_input,
                max_new_tokens=500,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        evaluation_text = tokenizer.decode(
            evaluation_ids[0, evaluation_input.shape[1] :],
            skip_special_tokens=True,
        )
    evaluations = parse_evaluation(evaluation_text, 1)
    return evaluations[0]


def write_evaluation_report(
    results: list[tuple[str, list[tuple[str, str]]]],
    output_path: Path,
    evaluator: Callable[[str, int], str],
) -> None:
    score_totals: dict[str, list[float]] = {}
    report_lines = [
        "# Partial-answer ratings",
        "",
        "Scores use a 0-10 scale and reflect only the visible answer text.",
        "",
    ]
    table_headers = ["Query"] + [label for label, _ in results[0][1]]
    report_lines.append("| " + " | ".join(table_headers) + " |")
    report_lines.append("| " + " | ".join("---" for _ in table_headers) + " |")

    for query_index, (query, answers) in enumerate(results, start=1):
        print(f"Evaluating query {query_index}/{len(results)}...", flush=True)
        evaluations = parse_evaluation(
            evaluator(build_evaluation_prompt(query, answers), len(answers)),
            len(answers),
        )
        scores = []
        for (label, _), evaluation in zip(answers, evaluations):
            score = evaluation["score"]
            score_totals.setdefault(label, []).append(score)
            scores.append(f"{score:.2f}")
        report_lines.append("| " + " | ".join([str(query_index), *scores]) + " |")

    report_lines.extend(("", "## Method averages", ""))
    report_lines.append("| Method | Average rating |")
    report_lines.append("| --- | ---: |")
    for label, scores in score_totals.items():
        report_lines.append(f"| {label} | {sum(scores) / len(scores):.2f} |")
    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def parse_query_indices(value: str) -> tuple[int, ...]:
    indices: list[int] = []
    for part in value.split(","):
        bounds = part.split("-", maxsplit=1)
        try:
            start = int(bounds[0])
            end = int(bounds[-1])
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid query index or range: {part!r}"
            ) from error
        if start > end:
            raise argparse.ArgumentTypeError(
                f"query index range must be ascending: {part!r}"
            )
        if start < 1 or end > len(EXAMPLE_QUERIES):
            raise argparse.ArgumentTypeError(
                f"query indices must be between 1 and {len(EXAMPLE_QUERIES)}"
            )
        indices.extend(range(start, end + 1))
    return tuple(dict.fromkeys(indices))


def parse_benchmark_indices(value: str, task_name: str) -> tuple[int, ...]:
    indices: list[int] = []
    for part in value.split(","):
        match = re.fullmatch(r"(-?\d+)(?:-(-?\d+))?", part)
        if match is None:
            raise argparse.ArgumentTypeError(
                f"invalid {task_name} index or range: {part!r}"
            )
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start > end:
            raise argparse.ArgumentTypeError(
                f"{task_name} index range must be ascending: {part!r}"
            )
        if start == 0:
            raise argparse.ArgumentTypeError(
                f"{task_name} indices cannot be zero."
            )
        indices.extend(range(start, end + 1))
    return tuple(dict.fromkeys(indices))


def parse_game24_indices(value: str) -> tuple[int, ...]:
    return parse_benchmark_indices(value, "Game24")


def parse_sudoku_indices(value: str) -> tuple[int, ...]:
    return parse_benchmark_indices(value, "Sudoku")


def parse_sudoku_file(value: str) -> str | Path:
    return value if value.startswith("https://") else Path(value)


def parse_bbeh_indices(value: str) -> tuple[int, ...]:
    return parse_benchmark_indices(value, "BBEH")


def parse_knowedit_indices(value: str) -> tuple[int, ...]:
    return parse_benchmark_indices(value, "KnowEdit")


def resolve_benchmark_indices(
    indices: Sequence[int], puzzle_count: int, task_name: str
) -> tuple[int, ...]:
    resolved = tuple(
        index if index > 0 else puzzle_count + index + 1 for index in indices
    )
    if any(not 1 <= index <= puzzle_count for index in resolved):
        raise ValueError(
            f"{task_name} indices must be between 1 and {puzzle_count}, or between "
            f"-{puzzle_count} and -1."
        )
    return resolved


def resolve_game24_indices(
    indices: Sequence[int], puzzle_count: int
) -> tuple[int, ...]:
    return resolve_benchmark_indices(indices, puzzle_count, "Game24")


def format_index_ranges(indices: Sequence[int]) -> str:
    if not indices:
        return ""
    ranges: list[str] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = index
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def query_index_signature(argv: Sequence[str]) -> str:
    signature = "all"
    for index, argument in enumerate(argv):
        if argument == "--query-index":
            signature = argv[index + 1]
        elif argument.startswith("--query-index="):
            signature = argument.partition("=")[2]
    return signature


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    normalized_argv: list[str] = []
    argument_index = 0
    while argument_index < len(argv):
        argument = argv[argument_index]
        if (
            argument
            in ("--game24-index", "--countdown-index", "--sudoku-index", "--bbeh-index", "--knowedit-index")
            and argument_index + 1 < len(argv)
            and argv[argument_index + 1].startswith("-")
        ):
            normalized_argv.append(f"{argument}={argv[argument_index + 1]}")
            argument_index += 2
            continue
        normalized_argv.append(argument)
        argument_index += 1
    argv = normalized_argv
    parser = argparse.ArgumentParser(
        description="Generate text with multi-pass residual-stream recirculation."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Free-form prompt. Overrides --query-index when provided.",
    )
    parser.add_argument(
        "--query-index",
        dest="query_indices",
        type=parse_query_indices,
        default=tuple(range(1, len(EXAMPLE_QUERIES) + 1)),
        metavar="INDEX[-INDEX][,...]",
        help=(
            "Select built-in queries by 1-based index or inclusive range "
            "(default: all queries)."
        ),
    )
    parser.add_argument(
        "--list-queries",
        action="store_true",
        help="Print the built-in queries and exit.",
    )
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--game24-puzzle",
        type=int,
        nargs=4,
        metavar=("NUMBER", "NUMBER", "NUMBER", "NUMBER"),
        help="Run one Game24 puzzle using the four supplied numbers.",
    )
    task_group.add_argument(
        "--game24-file",
        type=Path,
        help="Run puzzles from a tree-of-thought-llm-compatible Game24 CSV.",
    )
    task_group.add_argument(
        "--countdown-puzzle",
        type=int,
        nargs=7,
        metavar=("TARGET", "N1", "N2", "N3", "N4", "N5", "N6"),
        help="Run one Countdown target followed by its six allowed numbers.",
    )
    task_group.add_argument(
        "--countdown-file",
        type=Path,
        help="Run puzzles from a Countdown CSV with numbers and target columns.",
    )
    task_group.add_argument(
        "--sudoku-file",
        type=parse_sudoku_file,
        default="https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/test.csv",
        help="Run puzzles from the Sudoku Extreme test CSV URL (default), a local CSV, or a Sudoku4LLM JSONL file.",
    )
    task_group.add_argument(
        "--bbeh-file",
        type=Path,
        help=(
            "Run examples from a BBEH benchmark_tasks directory, task.json, "
            "or mini/data.json file."
        ),
    )
    task_group.add_argument(
        "--knowedit-file",
        type=Path,
        help="Run target-blind questions from a KnowEdit fact or WikiBio JSON file.",
    )
    parser.add_argument(
        "--do-sudoku",
        action="store_true",
        help="Run the Sudoku evaluation task using --sudoku-file.",
    )
    parser.add_argument(
        "--game24-index",
        type=parse_game24_indices,
        default=(),
        metavar="INDEX[-INDEX][,...]",
        help=(
            "1-based Game24 CSV indices; negative values count from the end "
            "(default: all)."
        ),
    )
    parser.add_argument(
        "--countdown-index",
        type=parse_game24_indices,
        default=(),
        metavar="INDEX[-INDEX][,...]",
        help=(
            "1-based Countdown CSV indices; negative values count from the end "
            "(default: all)."
        ),
    )
    parser.add_argument(
        "--sudoku-index",
        type=parse_sudoku_indices,
        default=(),
        metavar="INDEX[-INDEX][,...]",
        help=(
            "1-based Sudoku puzzle indices; negative values count from the "
            "end (default: all)."
        ),
    )
    parser.add_argument(
        "--bbeh-index",
        type=parse_bbeh_indices,
        default=(),
        metavar="INDEX[-INDEX][,...]",
        help=(
            "1-based BBEH JSON indices; negative values count from the end "
            "(default: all)."
        ),
    )
    parser.add_argument(
        "--knowedit-index",
        type=parse_knowedit_indices,
        default=(),
        metavar="INDEX[-INDEX][,...]",
        help="1-based KnowEdit JSON indices; negative values count from the end (default: all).",
    )
    parser.add_argument(
        "--eval-provider",
        choices=("local", "openai"),
        default="openai",
        help="Use a local Transformers model or the OpenAI API for evaluation.",
    )
    parser.add_argument(
        "--do-eval",
        action="store_true",
        help="Evaluate each generated run with the configured evaluation model.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Compare unmodified and recirculated passes of --model and record "
            "distribution similarities."
        ),
    )
    parser.add_argument(
        "--evaluation-model",
        default="gpt-5.6-sol",
        help="Model used by the OpenAI evaluator (default: gpt-5.6-sol).",
    )
    parser.add_argument(
        "--openai-base-url",
        default="https://api.openai.com/v1",
        help="OpenAI API base URL, also usable with compatible APIs.",
    )
    parser.add_argument(
        "--pair",
        dest="pairs",
        action="append",
        type=int,
        nargs=2,
        metavar=("SOURCE", "DESTINATION"),
        default=None,
        help=(
            "Source/destination pair to recirculate. Repeat for multiple pairs "
            "(default: -5 5, or 25 19 for gemma-4-26b-a4b)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("source", "layerwise"),
        default="source",
        help="Recirculate source features or repeat each selected layer in place.",
    )
    parser.add_argument(
        "--override-to-global-attn",
        action="store_true",
        help=(
            "Snap every --pair layer to the nearest full/global-attention layer, "
            "for models that interleave sliding and global attention."
        ),
    )
    # alpha: weight of the source residual stream in the convex combination. 
    parser.add_argument("--alpha", type=float, default=0.5)
    # beta: weight of the destination residual stream in the convex combination. 
    # If None, defaults to 1 - alpha.
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Defaults to 1 - alpha.",
    )
    parser.add_argument(
        "--noise-level-range",
        type=float,
        nargs=2,
        default=(0.1, 0.2),
        metavar=("MIN", "MAX"),
        help=(
            "Magnitude-matched Gaussian noise range. Each pass maps its "
            "normalized preceding margin from MIN to MAX "
            "(0 <= MIN <= MAX <= 0.5; default: 0.1 0.2)."
        ),
    )
    parser.add_argument(
        "--noise-decay-per-pass",
        type=float,
        default=0.7,
        metavar="COEFFICIENT",
        help=(
            "Multiply the noise weight by this coefficient after each "
            "recirculation pass (0 to 1; default: 0.7)."
        ),
    )
    parser.add_argument(
        "--perturb-pre-margin-thres",
        type=float,
        default=0.1,
        metavar="THRESHOLD",
        help=(
            "Apply noise only when the pre-pass "
            "top-1/top-2 probability margin is at least this threshold "
            "(default: 0.05)."
        ),
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Total model passes per token, including the initial pass (default: 1).",
    )
    parser.add_argument(
        "--cond-recirculate",
        action="store_true",
        help=(
            "Recirculate only when the inference model's source/destination "
            "activation similarity reaches --act-sim-thres."
        ),
    )
    parser.add_argument(
        "--act-sim-thres",
        type=float,
        nargs="+",
        default=None,
        metavar="THRESHOLD",
        help=(
            "Conditional thresholds in --pair order. Provide one value for all "
            "pairs or one per pair; each pair must meet its own threshold to "
            "inject (default: None, disabled if not set)."
        ),
    )
    parser.add_argument(
        "--pre-margin-thres",
        type=float,
        default=0.2,
        metavar="THRESHOLD",
        help=(
            "Primary conditional gate: recirculate only when the top-1 versus "
            "top-2 probability margin is at most this value."
        ),
    )
    parser.add_argument(
        "--post-margin-thres",
        type=float,
        nargs=2,
        default=(0.1, 0.2),
        metavar=("MIN", "MAX"),
        help=(
            "Final-pass margin bands: reject below MIN, require "
            "--post-margin-ratio-thres from MIN (inclusive) to MAX "
            "(exclusive), and accept at or above MAX (default: 0.1 0.2)."
        ),
    )
    parser.add_argument(
        "--post-margin-ratio-thres",
        type=float,
        default=1.2,
        metavar="RATIO",
        help=(
            "Minimum post-margin/P1-margin ratio required when the post margin "
            "is between --post-margin-thres MIN and MAX."
        ),
    )
    parser.add_argument(
        "--ada-recirculate",
        type=int,
        default=2,
        metavar="X",
        help=(
            "Run at most X additional passes; with a post-margin gate, stop "
            "early once either configured threshold is met."
        ),
    )
    parser.add_argument(
        "--cosine-reject",
        type=float,
        default=0.8,
        metavar="THRESHOLD",
        help=(
            "Discard the final pass when its distribution has cosine similarity "
            "below this threshold relative to P1."
        ),
    )
    parser.add_argument(
        "--repetition-recovery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Force recirculation without eligibility or rejection gates when a "
            "generated line occurs for the third time (default: enabled)."
        ),
    )
    parser.add_argument(
        "--perturb-every-n-tokens",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Force one recirculation pass every N generated tokens, regardless "
            "of repetition or eligibility gates (default: 0, disabled)."
        ),
    )
    parser.add_argument(
        "--perturb-mode",
        choices=("repel-history", "towards-target"),
        default=None,
        help=(
            "Select periodic noise candidates by repelling history or maximizing "
            "the first KnowEdit target token's probability (default: towards-target "
            "with --knowedit-file, repel-history otherwise)."
        ),
    )
    parser.add_argument(
        "--perturb-for-k-tokens",
        type=int,
        default=30,
        metavar="K",
        help=(
            "Force recirculation for K consecutive generated tokens after each "
            "--perturb-every-n-tokens interval (default: 30)."
        ),
    )
    parser.add_argument(
        "--perturb-recent-m-tokens",
        type=int,
        default=32,
        metavar="M",
        help=(
            "Average the most recent M token embeddings to form the negative "
            "direction injected during periodic perturbation (default: 32)."
        ),
    )
    parser.add_argument(
        "--perturb-history-decay",
        type=float,
        default=0,
        metavar="LAMBDA",
        help=(
            "Decay for earlier periodic-perturbation centroids in the injected "
            "repulsion direction (0 to 1; default: 0, disable history)."
        ),
    )
    parser.add_argument(
        "--perturb-noise-level-range",
        type=float,
        nargs=2,
        default=(0.1, 0.2),
        metavar=("MIN", "MAX"),
        help=(
            "Override the noise range during periodic perturbation "
            "(default: 0.1 0.2)."
        ),
    )
    parser.add_argument(
        "--periodic-perturbation-candidate-count",
        type=int,
        default=16,
        metavar="COUNT",
        help="Number of periodic perturbation candidates to probe per step (default: 16).",
    )
    parser.add_argument(
        "--periodic-perturbation-steps",
        type=int,
        default=1,
        metavar="STEPS",
        help="Number of sequential periodic perturbation steps (default: 1).",
    )
    parser.add_argument(
        "--periodic-perturbation-step-decay",
        type=float,
        default=0.8,
        metavar="S",
        help="Scale each step's weighted perturbation by S^step (default: 0.8).",
    )
    parser.add_argument(
        "--cosine-top-k",
        type=int,
        default=5,
        metavar="K",
        help=(
            "Compute P1/final-pass cosine over the union of their top-K "
            "tokens and log the final student's top-K tokens (default: 5)."
        ),
    )
    parser.add_argument(
        "--noise-injected-source-top-k",
        type=int,
        default=4,
        metavar="K",
        help="Log the top K tokens decoded from each injected source (default: 4).",
    )
    parser.add_argument(
        "--gating-pair-index",
        type=int,
        default=0,
        metavar="INDEX",
        help=(
            "When multiple (source, destination) pairs are present, this is the "
            "zero-based --pair index that must meet its threshold before any "
            "conditional recirculation occurs (default: 0)."
        ),
    )
    parser.add_argument(
        "--debug-layer-sim",
        action="store_true",
        help=(
            "Collect per-token source/destination activation similarity and "
            "print summary stats after each query, even without recirculation."
        ),
    )
    parser.add_argument(
        "--debug-adj-layer-sim",
        action="store_true",
        help=(
            "Collect per-token cosine similarity between every pair of "
            "adjacent decoder blocks and print summary stats after each query."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument(
        "--no-recirculate-after-N-tokens",
        dest="no_recirculate_after_tokens",
        type=int,
        default=None,
        metavar="N",
        help="Disable recirculation after N generated tokens.",
    )
    parser.add_argument(
        "--ablation",
        dest="ablations",
        action="store_true",
        help=(
            "Arguments before the first --ablation form the baseline. Each "
            "--ablation and its following arguments form one ablation run."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Use greedy decoding at 0, sampling above 0.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        metavar="PENALTY",
        help=(
            "Override the model repetition penalty; values above 1 discourage "
            "previously generated tokens."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="Single-device fallback used when --device-map is none.",
    )
    parser.add_argument(
        "--device-map",
        choices=("balanced", "auto", "balanced_low_0", "sequential", "none"),
        default="balanced",
        help="Transformers device map. 'balanced' shards layers across all GPUs.\n",
    )
    parser.add_argument(
        "--gpu-memory",
        default="auto",
        help=(
            "Maximum model memory per GPU when sharding, or 'auto' to use free "
            "memory minus 4 GiB per GPU (default: auto)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save queries and generated outputs as pretty JSON.",
    )
    parser.add_argument(
        "--similarities-output",
        type=Path,
        default=None,
        help="Write pretty-printed debug-run records with per-token similarities.",
    )
    parser.add_argument(
        "--evaluate-results-json",
        dest="evaluate_results_json",
        type=Path,
        metavar="PATH",
        help="Evaluate partial answers in an existing JSON results report and exit.",
    )
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        default=Path("ratings.md"),
        help="Save the evaluation table to this Markdown file.",
    )
    if "--ablation" not in argv:
        args = parser.parse_args(argv)
        if args.pairs is None:
            model_name = args.model.rsplit("/", 1)[-1].lower()
            args.pairs = [(25, 19)] if model_name.startswith("gemma-4-26b-a4b") else [(-5, 5)]
        args.ablations = False
        args.query_index_signature = query_index_signature(argv)
        args.perturb_mode = args.perturb_mode or (
            "towards-target" if args.knowedit_file is not None else "repel-history"
        )
        return args

    ablations_index = argv.index("--ablation")
    baseline_argv = argv[:ablations_index]
    args = parser.parse_args(baseline_argv)
    if args.pairs is None:
        model_name = args.model.rsplit("/", 1)[-1].lower()
        args.pairs = [(25, 19)] if model_name.startswith("gemma-4-26b-a4b") else [(-5, 5)]
    args.query_index_signature = query_index_signature(argv)

    ablation_groups: list[list[str]] = []
    current_group: list[str] | None = None
    for argument in argv[ablations_index:]:
        if argument == "--ablation":
            if current_group is not None:
                ablation_groups.append(current_group)
            current_group = []
        else:
            assert current_group is not None
            current_group.append(argument)
    assert current_group is not None
    ablation_groups.append(current_group)

    if len(ablation_groups) == 1 and not ablation_groups[0]:
        args.ablations = None
        args.perturb_mode = args.perturb_mode or (
            "towards-target" if args.knowedit_file is not None else "repel-history"
        )
        return args
    if any(not group for group in ablation_groups):
        parser.error("each --ablation must be followed by at least one argument")

    global_dests = {
        "ablations",
        "debug",
        "debug_layer_sim",
        "debug_adj_layer_sim",
        "device",
        "device_map",
        "do_eval",
        "eval_provider",
        "evaluate_results_json",
        "evaluation_model",
        "evaluation_output",
        "gpu_memory",
        "game24_file",
        "game24_index",
        "game24_puzzle",
        "countdown_puzzle",
        "countdown_file",
        "countdown_index",
        "sudoku_file",
        "sudoku_index",
        "do_sudoku",
        "bbeh_file",
        "bbeh_index",
        "knowedit_file",
        "knowedit_index",
        "list_queries",
        "max_new_tokens",
        "model",
        "perturb_pre_margin_thres",
        "noise_injected_source_top_k",
        "no_recirculate_after_tokens",
        "openai_base_url",
        "output",
        "query_indices",
        "repetition_penalty",
        "similarities_output",
    }

    ablations: list[tuple[tuple[str, Any], ...]] = []
    for argument_group in ablation_groups:
        option_names = [
            argument.partition("=")[0]
            for argument in argument_group
            if argument.partition("=")[0] in parser._option_string_actions
        ]
        variation = parser.parse_args(argument_group)
        if not option_names or variation.prompt is not None:
            parser.error("ablation arguments must be options")
        for option_name in dict.fromkeys(option_names):
            dest = parser._option_string_actions[option_name].dest
            if dest in global_dests:
                val = (
                    getattr(variation, dest)
                    if parser._option_string_actions[option_name].nargs == 0
                    and isinstance(
                        parser._option_string_actions[option_name].const, bool
                    )
                    else getattr(variation, dest)
                )
                setattr(args, dest, val)
        overrides = tuple(
            (
                parser._option_string_actions[option_name].dest,
                (
                    getattr(variation, parser._option_string_actions[option_name].dest)
                    if parser._option_string_actions[option_name].nargs == 0
                    and isinstance(
                        parser._option_string_actions[option_name].const, bool
                    )
                    else getattr(
                        variation, parser._option_string_actions[option_name].dest
                    )
                ),
            )
            for option_name in dict.fromkeys(option_names)
            if parser._option_string_actions[option_name].dest not in global_dests
        )
        ablations.append(overrides)

    args.ablations = tuple(ablations)
    args.perturb_mode = args.perturb_mode or (
        "towards-target" if args.knowedit_file is not None else "repel-history"
    )
    return args


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_gpu_memory_limits(requested: str) -> dict[int, str]:
    if requested != "auto":
        return {
            gpu: requested for gpu in range(torch.cuda.device_count())
        }

    limits: dict[int, str] = {}
    for gpu in range(torch.cuda.device_count()):
        free_memory, _ = torch.cuda.mem_get_info(gpu)
        usable_memory = free_memory - GPU_MEMORY_RESERVE_BYTES
        if usable_memory <= 0:
            raise RuntimeError(
                f"GPU {gpu} has less than 4 GiB free; specify --gpu-memory "
                "explicitly or free GPU memory."
            )
        limits[gpu] = f"{usable_memory // 1024**2}MiB"
    return limits


def enable_fp32_output_projection(model: nn.Module) -> None:
    output_embeddings = model.get_output_embeddings()
    if not isinstance(output_embeddings, nn.Linear):
        raise TypeError("FP32 logits require a linear output embedding module.")

    def fp32_forward(module: nn.Linear, hidden_states: Tensor) -> Tensor:
        flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
        if (
            flattened.device.type == "cuda"
            and flattened.dtype in (torch.float16, torch.bfloat16)
            and flattened.dtype == module.weight.dtype
        ):
            logits = torch.mm(
                flattened,
                module.weight.t(),
                out_dtype=torch.float32,
            )
        else:
            logits = torch.nn.functional.linear(
                flattened.float(),
                module.weight.float(),
            )
        if module.bias is not None:
            logits += module.bias.float()
        return logits.reshape(*hidden_states.shape[:-1], module.out_features)

    replacement = MethodType(fp32_forward, output_embeddings)
    if hasattr(output_embeddings, "_old_forward"):
        output_embeddings._old_forward = replacement
    else:
        output_embeddings.forward = replacement


def configure_model_generation(
    model: nn.Module, model_name: str, repetition_penalty: float | None = None
) -> None:
    if repetition_penalty is not None:
        model.generation_config.repetition_penalty = repetition_penalty
    elif model_name.rsplit("/", 1)[-1].lower().startswith("gpt-oss"):
        model.generation_config.repetition_penalty = 1.1
    elif model.generation_config.repetition_penalty is None:
        model.generation_config.repetition_penalty = 1.0


def resolve_source(source: int, num_blocks: int) -> int:
    return num_blocks + source if source < 0 else source


def resolve_recirculation_pairs(
    args: argparse.Namespace, num_blocks: int, global_attention_layers: Sequence[int] | None
) -> tuple[tuple[int, int], ...]:
    requested_pairs = args.pairs
    pairs = tuple(
        (resolve_source(source, num_blocks), resolve_source(destination, num_blocks))
        for source, destination in requested_pairs
    )
    if args.override_to_global_attn and global_attention_layers:
        pairs = tuple(
            (
                nearest_global_layer(global_attention_layers, source),
                nearest_global_layer(global_attention_layers, destination),
            )
            for source, destination in pairs
        )
    return pairs


def output_recirculation_pairs(args: argparse.Namespace) -> Sequence[Sequence[int]]:
    if args.cond_recirculate or not args.ablations:
        return args.pairs
    for overrides in args.ablations:
        values = dict(overrides)
        if values.get("cond_recirculate") and "pairs" in values:
            return values["pairs"]
    return args.pairs


def find_global_attention_layer_indices(
    model: nn.Module, num_blocks: int
) -> tuple[int, ...] | None:
    # Models that interleave sliding-window and full/global attention layers
    # (e.g. Gemma) expose the pattern via layer_types or sliding_window_pattern.
    text_config = getattr(model.config, "text_config", model.config)
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is not None and len(layer_types) == num_blocks:
        return tuple(
            index
            for index, layer_type in enumerate(layer_types)
            if layer_type in ("full_attention", "global_attention")
        )
    pattern = getattr(text_config, "sliding_window_pattern", None)
    if isinstance(pattern, int) and pattern > 0:
        return tuple(range(pattern - 1, num_blocks, pattern))
    return None


def nearest_global_layer(candidates: Sequence[int], target: int) -> int:
    return min(candidates, key=lambda candidate: (abs(candidate - target), candidate))


def find_decoder_blocks(model: nn.Module) -> Sequence[nn.Module]:
    candidate_paths = (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("transformer", "blocks"),
        # Multimodal wrappers (e.g. Gemma 4) nest the text decoder under language_model.
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
    )
    for path in candidate_paths:
        value: Any = model
        for attribute in path:
            value = getattr(value, attribute, None)
            if value is None:
                break
        if isinstance(value, (nn.ModuleList, list, tuple)) and all(
            isinstance(block, nn.Module) for block in value
        ):
            return value

    raise ValueError(
        "Could not locate the decoder blocks. Add the model's ModuleList path "
        "to find_decoder_blocks()."
    )


def rewind_dynamic_cache(cache: DynamicCache) -> DynamicCache:
    crop_parameter = next(iter(inspect.signature(cache.crop).parameters.values()))
    if crop_parameter.name == "tokens_to_remove":
        cache.crop(-1)
    else:
        cache.crop(cache.get_seq_length() - 1)
    return cache


def finalize_dynamic_cache_token(cache: DynamicCache) -> DynamicCache:
    crop_parameter = next(iter(inspect.signature(cache.crop).parameters.values()))
    if crop_parameter.name == "tokens_to_remove":
        cache.crop(0)
    return cache


def rewind_dynamic_cache_layer(
    cache: DynamicCache, layer_index: int
) -> DynamicCache:
    layer = cache.layers[layer_index]
    crop_parameter = next(iter(inspect.signature(layer.crop).parameters.values()))
    if crop_parameter.name == "tokens_to_remove":
        layer.crop(-1)
    else:
        layer.crop(layer.get_seq_length() - 1)
    return cache


def clone_tensor_states(states: dict[int, Tensor | None]) -> dict[int, Tensor]:
    return {
        index: tensor.detach().clone()
        for index, tensor in states.items()
        if tensor is not None
    }


def restore_tensor_states(
    states: dict[int, Tensor | None], saved: dict[int, Tensor]
) -> None:
    for index, tensor in saved.items():
        current = states.get(index)
        if current is None:
            states[index] = tensor.clone()
        else:
            current.copy_(tensor)


def capture_dynamic_cache_rewind_state(
    cache: DynamicCache,
) -> tuple[tuple[dict[int, Tensor], dict[int, bool]] | None, ...]:
    return tuple(
        (
            clone_tensor_states(layer.recurrent_states),
            dict(layer.has_previous_state),
        )
        if isinstance(getattr(layer, "recurrent_states", None), dict)
        else None
        for layer in cache.layers
    )


def restore_dynamic_cache_rewind_state(
    cache: DynamicCache,
    saved_states: tuple[
        tuple[dict[int, Tensor], dict[int, bool]] | None, ...
    ],
) -> None:
    for layer, saved in zip(cache.layers, saved_states):
        if saved is None:
            continue
        recurrent_states, has_previous_state = saved
        restore_tensor_states(layer.recurrent_states, recurrent_states)
        layer.has_previous_state.clear()
        layer.has_previous_state.update(has_previous_state)


def capture_dynamic_cache_token(cache: DynamicCache) -> tuple[dict[str, Any], ...]:
    captured = []
    for layer in cache.layers:
        state: dict[str, Any] = {}
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if isinstance(keys, Tensor) and isinstance(values, Tensor) and keys.numel() > 0:
            state["keys"] = keys[..., -1:, :].detach().clone()
            state["values"] = values[..., -1:, :].detach().clone()
        conv_states = getattr(layer, "conv_states", None)
        if isinstance(conv_states, dict):
            state["conv_states"] = clone_tensor_states(conv_states)
        recurrent_states = getattr(layer, "recurrent_states", None)
        if isinstance(recurrent_states, dict):
            state["recurrent_states"] = clone_tensor_states(recurrent_states)
        captured.append(state)
    return tuple(captured)


def restore_dynamic_cache_token(
    cache: DynamicCache, cached_token: tuple[dict[str, Any], ...]
) -> None:
    for layer, state in zip(cache.layers, cached_token):
        if "keys" in state:
            layer.keys[..., -1:, :].copy_(state["keys"])
            layer.values[..., -1:, :].copy_(state["values"])
        if "conv_states" in state:
            layer.conv_states.update(
                {
                    index: tensor.clone()
                    for index, tensor in state["conv_states"].items()
                }
            )
        if "recurrent_states" in state:
            restore_tensor_states(layer.recurrent_states, state["recurrent_states"])


def sample_token(logits: Tensor, temperature: float) -> Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def top_k_decoded_tokens(logits: Tensor, tokenizer: Any, count: int) -> list[str]:
    token_ids = torch.topk(
        logits.float(), k=min(count, logits.shape[-1]), dim=-1
    ).indices[0]
    return [tokenizer.decode(token_id) for token_id in token_ids]


def apply_repetition_penalty(
    logits: Tensor,
    previous_token_ids: Tensor,
    penalty: float,
    recirculated: bool = False,
) -> Tensor:
    if recirculated or penalty == 1.0:
        return logits
    previous_scores = logits.gather(1, previous_token_ids)
    previous_scores = torch.where(
        previous_scores < 0, previous_scores * penalty, previous_scores / penalty
    )
    return logits.scatter(1, previous_token_ids, previous_scores)


REPETITION_RECOVERY_TOKEN_COUNT = 4
REPETITION_TEXT_WINDOW_WORD_COUNT = 1024


def repetition_recovery_penalty(recovery_pending: bool) -> float:
    return 1.1 if recovery_pending else 1.0


def periodic_perturbation_active(
    perturb_every_n_tokens: int,
    perturb_for_k_tokens: int,
    generated_token_count: int,
) -> bool:
    if perturb_every_n_tokens <= 0 or perturb_for_k_tokens <= 0:
        return False
    position_in_interval = generated_token_count % perturb_every_n_tokens
    return generated_token_count >= perturb_every_n_tokens and (
        position_in_interval == 0 or position_in_interval < perturb_for_k_tokens
    )


def periodic_perturbation_direction_scale(
    perturb_every_n_tokens: int, generated_token_count: int
) -> float:
    """Exponentially decay the direction within a periodic perturbation window."""
    position_in_interval = generated_token_count % perturb_every_n_tokens
    return 0.9**position_in_interval


def periodic_perturbation_history_direction(
    centroids: Sequence[Tensor], history_decay: float
) -> Tensor:
    weighted_centroid = torch.zeros_like(centroids[-1])
    for age, centroid in enumerate(reversed(centroids)):
        weighted_centroid += history_decay**age * centroid
    return -weighted_centroid


def recent_token_centroid(
    embeddings: nn.Module, token_ids: Tensor, token_count: int
) -> Tensor:
    with torch.inference_mode():
        return embeddings(token_ids[:, -token_count:]).mean(dim=1).detach()


def repetition_text_window(
    text: str, maximum_word_count: int = REPETITION_TEXT_WINDOW_WORD_COUNT
) -> str:
    """Return the recent word-bounded portion of decoded text for detection."""
    word_matches = list(re.finditer(r"\S+", text))
    if len(word_matches) <= maximum_word_count:
        return text
    return text[word_matches[-maximum_word_count].start() :]


def effective_adaptive_recirculation(
    configured_passes: int,
    conditional_recirculation: bool,
    forced_recovery: bool = False,
) -> int:
    return configured_passes if conditional_recirculation or forced_recovery else 0


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False


def format_run_arguments(args: argparse.Namespace, options: Sequence[str]) -> str:
    values = {
        option: (
            getattr(args, option).rsplit("/", 1)[-1]
            if option == "model"
            else getattr(args, option)
        )
        for option in options
    }
    return ", ".join(
        f"{option.replace('_', '-')}={value}" for option, value in values.items()
    )


def periodic_perturbation_noise_level_range(
    recovery_noise_level_range: tuple[float, float],
    override_noise_level_range: tuple[float, float] | None,
) -> tuple[float, float]:
    if override_noise_level_range is not None:
        return override_noise_level_range
    return (
        recovery_noise_level_range
        if recovery_noise_level_range[1] > 0
        else (0.1, 0.2)
    )


def has_third_repeated_suffix(
    token_ids: Sequence[int],
    minimum_sequence_length: int = 8,
    maximum_sequence_length: int = 32,
) -> bool:
    """Whether the current suffix is the third occurrence of a token sequence."""
    maximum_length = min(maximum_sequence_length, len(token_ids) // 3)
    for sequence_length in range(minimum_sequence_length, maximum_length + 1):
        suffix = token_ids[-sequence_length:]
        occurrences = sum(
            token_ids[start : start + sequence_length] == suffix
            for start in range(len(token_ids) - sequence_length + 1)
        )
        if occurrences >= 3:
            return True
    return False


def has_third_repeated_text_suffix(
    text: str,
    minimum_word_count: int = 8,
    maximum_word_count: int = 32,
) -> bool:
    """Whether a substantial normalized word span appears at least three times."""
    return bool(
        repeated_text_signature_counts(
            text, minimum_word_count, maximum_word_count
        )
    )


def repeated_text_signature_counts(
    text: str,
    minimum_word_count: int = 8,
    maximum_word_count: int = 32,
) -> dict[tuple[str, ...], int]:
    """Count substantial normalized patterns and blank-line loops."""
    signature_counts: dict[tuple[str, ...], int] = {}
    blank_line_run = 0
    previous_symbol_line: str | None = None
    symbol_line_run = 0
    for line in text.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            previous_symbol_line = None
            symbol_line_run = 0
            blank_line_run += 1
            if blank_line_run >= 3:
                signature_counts[("blank-lines",)] = blank_line_run
            continue
        blank_line_run = 0
        if re.fullmatch(r"[^\w\s]{1,3}", stripped_line):
            symbol_line_run = (
                symbol_line_run + 1
                if stripped_line == previous_symbol_line
                else 1
            )
            previous_symbol_line = stripped_line
            if symbol_line_run >= 3:
                signature_counts[("symbol-line", stripped_line)] = symbol_line_run
        else:
            previous_symbol_line = None
            symbol_line_run = 0
    lines = [
        re.sub(r"^let's try:\s*", "", re.sub(r"^\d+[.)]\s*", "", line.strip()).lower())
        for line in text.splitlines()
        if line.strip()
    ]
    for line in lines:
        words = re.findall(r"\S+", line)
        line_count = lines.count(line)
        if len(words) >= minimum_word_count and line_count >= 3:
            signature_counts[("line", line)] = line_count
        maximum_length = min(maximum_word_count, len(words) // 3)
        for word_count in range(minimum_word_count, maximum_length + 1):
            spans: dict[tuple[str, ...], list[int]] = {}
            for start in range(len(words) - word_count + 1):
                span = tuple(words[start : start + word_count])
                starts = spans.setdefault(span, [])
                if not starts or start - starts[-1] >= word_count:
                    starts.append(start)
                    if len(starts) >= 3:
                        signature = ("span", *span)
                        signature_counts[signature] = max(
                            signature_counts.get(signature, 0), len(starts)
                        )
    return signature_counts


@dataclasses.dataclass(frozen=True)
class RepetitionRecoverySettings:
    noise_level_range: tuple[float, float]
    cosine_reject: float | None
    pre_margin_threshold: float | None
    post_margin_threshold: tuple[float, float] | None
    post_margin_ratio_threshold: float | None
    condition_thresholds: Sequence[float] | None
    recirculation_allowed: bool
    force_recirculation: bool


def repetition_recovery_settings(
    noise_level_range: tuple[float, float],
    cosine_reject: float | None,
    pre_margin_threshold: float,
    post_margin_threshold: tuple[float, float] | None,
    post_margin_ratio_threshold: float | None,
    condition_thresholds: Sequence[float] | None,
    recirculation_allowed: bool,
    token_ids: Sequence[int],
    enabled: bool = True,
    decoded_text: str | None = None,
    consecutive_repetition_count: int = 1,
    repetition_detected: bool | None = None,
) -> RepetitionRecoverySettings:
    if repetition_detected is None:
        repetition_detected = (
            has_third_repeated_text_suffix(decoded_text)
            if decoded_text is not None
            else has_third_repeated_suffix(token_ids)
        )
    if not enabled or not repetition_detected:
        return RepetitionRecoverySettings(
            noise_level_range,
            cosine_reject,
            pre_margin_threshold,
            post_margin_threshold,
            post_margin_ratio_threshold,
            condition_thresholds,
            recirculation_allowed,
            False,
        )
    recovery_noise_level_range = (
        (0.1, 0.2) if noise_level_range == (0.0, 0.0) else noise_level_range
    )
    recovery_noise_level_range = tuple(
        min(0.4, level * 2 ** (consecutive_repetition_count - 1))
        for level in recovery_noise_level_range
    )
    return RepetitionRecoverySettings(
        recovery_noise_level_range,
        min(cosine_reject, 0.3) if cosine_reject is not None else None,
        None,
        None,
        None,
        None,
        True,
        True,
    )


class StreamingSimilarityWriter:
    def __init__(self, path: Path) -> None:
        self.file = path.open("w+", encoding="utf-8")
        self.file.write("[]\n")
        self.file.flush()
        self.top_close_position = 1
        self.run_count = 0
        self.similarity_count = 0
        self.similarities_close_position = 0

    def start_run(self, prompt: str, run: str, seed: int) -> None:
        self.file.seek(self.top_close_position)
        self.file.write("\n" if self.run_count == 0 else ",\n")
        self.file.write("  {\n    \"prompt\": ")
        json.dump(prompt, self.file, ensure_ascii=False)
        self.file.write(",\n    \"run\": ")
        json.dump(run, self.file, ensure_ascii=False)
        self.file.write(f",\n    \"seed\": {seed},\n    \"similarities\": [\n")
        self.similarities_close_position = self.file.tell()
        self.similarity_count = 0
        self._write_closing_delimiters()
        self.run_count += 1

    def append(self, similarity: dict[str, Any]) -> None:
        self.file.seek(self.similarities_close_position)
        if self.similarity_count:
            self.file.write(",\n")
        self.file.write("      ")
        json.dump(similarity, self.file, ensure_ascii=False, indent=6)
        self.file.write("\n")
        self.similarities_close_position = self.file.tell()
        self.similarity_count += 1
        self._write_closing_delimiters()

    def _write_closing_delimiters(self) -> None:
        self.file.write("    ]\n  }\n")
        self.top_close_position = self.file.tell()
        self.file.write("]\n")
        self.file.truncate()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def validate_run_arguments(args: argparse.Namespace) -> None:
    if args.repetition_penalty is not None and args.repetition_penalty <= 0:
        raise ValueError("--repetition-penalty must be positive.")
    if args.perturb_mode == "towards-target" and args.knowedit_file is None:
        raise ValueError("--perturb-mode towards-target requires --knowedit-file.")
    if args.perturb_every_n_tokens < 0:
        raise ValueError("--perturb-every-n-tokens must be nonnegative.")
    if args.perturb_for_k_tokens < 1:
        raise ValueError("--perturb-for-k-tokens must be at least 1.")
    if args.perturb_recent_m_tokens < 1:
        raise ValueError("--perturb-recent-m-tokens must be at least 1.")
    if args.periodic_perturbation_candidate_count < 1:
        raise ValueError("--periodic-perturbation-candidate-count must be at least 1.")
    if args.periodic_perturbation_steps < 1:
        raise ValueError("--periodic-perturbation-steps must be at least 1.")
    if not 0.0 <= args.periodic_perturbation_step_decay <= 1.0:
        raise ValueError("--periodic-perturbation-step-decay must be between 0 and 1.")
    if not 0.0 <= args.perturb_history_decay <= 1.0:
        raise ValueError("--perturb-history-decay must be between 0 and 1.")
    noise_min, noise_max = args.noise_level_range
    if not 0.0 <= noise_min <= noise_max <= 0.5:
        raise ValueError(
            "--noise-level-range requires 0 <= MIN <= MAX <= 0.5."
        )
    if args.perturb_noise_level_range is not None:
        perturb_noise_min, perturb_noise_max = args.perturb_noise_level_range
        if not 0.0 <= perturb_noise_min <= perturb_noise_max <= 0.5:
            raise ValueError(
                "--perturb-noise-level-range requires 0 <= MIN <= MAX <= 0.5."
            )
    if not 0.0 <= args.noise_decay_per_pass <= 1.0:
        raise ValueError("--noise-decay-per-pass must be between 0 and 1.")
    if args.perturb_pre_margin_thres < 0:
        raise ValueError("--perturb-pre-margin-thres must be nonnegative.")
    if noise_max > 0 and args.mode != "source":
        raise ValueError("noise ranges require --mode source.")
    if noise_max > 0 and (
        args.post_margin_thres is None
        or args.post_margin_thres[0] == args.post_margin_thres[1]
    ):
        raise ValueError(
            "noise ranges require --post-margin-thres with MIN < MAX."
        )
    if args.ada_recirculate < 0:
        raise ValueError("--ada-recirculate must be nonnegative.")
    if args.ada_recirculate:
        if args.mode != "source":
            raise ValueError("--ada-recirculate requires --mode source.")
    if args.passes == 1 and not args.ada_recirculate:
        return
    if args.cosine_reject is not None:
        if not 0.0 <= args.cosine_reject <= 1.0:
            raise ValueError("--cosine-reject must be between 0 and 1.")
        if args.passes < 2 and not args.ada_recirculate:
            raise ValueError("--cosine-reject requires --passes of at least 2.")
        if args.mode != "source":
            raise ValueError("--cosine-reject requires --mode source.")
    if args.post_margin_thres is not None:
        post_margin_min, post_margin_max = args.post_margin_thres
        if post_margin_min < 0 or post_margin_max < 0:
            raise ValueError("--post-margin-thres values must be nonnegative.")
        if post_margin_min > post_margin_max:
            raise ValueError("--post-margin-thres requires MIN <= MAX.")
        if args.passes < 2 and not args.ada_recirculate:
            raise ValueError("--post-margin-thres requires --passes of at least 2.")
        if args.mode != "source":
            raise ValueError("--post-margin-thres requires --mode source.")
    if args.post_margin_ratio_thres is not None:
        if args.post_margin_ratio_thres < 0:
            raise ValueError("--post-margin-ratio-thres must be nonnegative.")
        if args.passes < 2 and not args.ada_recirculate:
            raise ValueError(
                "--post-margin-ratio-thres requires --passes of at least 2."
            )
        if args.mode != "source":
            raise ValueError("--post-margin-ratio-thres requires --mode source.")
    if args.cosine_top_k < 1:
        raise ValueError("--cosine-top-k must be at least 1.")
    if args.noise_injected_source_top_k < 1:
        raise ValueError("--noise-injected-source-top-k must be at least 1.")


def main() -> None:
    args = parse_args()
    print(f"GPUs used: {torch.cuda.device_count()}", flush=True)
    if args.list_queries:
        for index, query in enumerate(EXAMPLE_QUERIES, start=1):
            print(f"{index:2}. {query}")
        return 

    if args.prompt is not None and (
        args.game24_puzzle is not None
        or args.game24_file is not None
        or args.countdown_puzzle is not None
        or args.countdown_file is not None
        or args.do_sudoku
        or args.bbeh_file is not None
        or args.knowedit_file is not None
    ):
        raise ValueError("A free-form prompt cannot be combined with a benchmark task.")
    if args.do_sudoku and (
        args.game24_puzzle is not None
        or args.game24_file is not None
        or args.countdown_puzzle is not None
        or args.countdown_file is not None
        or args.bbeh_file is not None
        or args.knowedit_file is not None
    ):
        raise ValueError("--do-sudoku cannot be combined with another benchmark task.")
    if args.game24_puzzle is not None and args.game24_index:
        raise ValueError("--game24-index requires --game24-file.")
    if args.countdown_puzzle is not None and args.countdown_index:
        raise ValueError("--countdown-index requires --countdown-file.")
    if args.sudoku_index and not args.do_sudoku:
        raise ValueError("--sudoku-index requires --do-sudoku.")
    if args.bbeh_index and args.bbeh_file is None:
        raise ValueError("--bbeh-index requires --bbeh-file.")
    if args.knowedit_index and args.knowedit_file is None:
        raise ValueError("--knowedit-index requires --knowedit-file.")

    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be nonnegative.")
    if (
        args.no_recirculate_after_tokens is not None
        and args.no_recirculate_after_tokens < 0
    ):
        raise ValueError("--no-recirculate-after-N-tokens must be nonnegative.")
    if args.passes < 1:
        raise ValueError("--passes must be at least 1.")
    validate_run_arguments(args)
    if args.temperature < 0:
        raise ValueError("--temperature must be nonnegative.")
    if args.eval_provider == "openai" and args.evaluate_results_json is not None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY before using --eval-provider openai.")
        write_evaluation_report(
            parse_results(args.evaluate_results_json),
            args.evaluation_output,
            lambda prompt, _method_count: openai_evaluate(
                prompt, args.evaluation_model, api_key, args.openai_base_url
            ),
        )
        print(args.evaluation_output.read_text(encoding="utf-8"), end="")
        return

    set_random_seed(args.seed)
    device = choose_device(args.device)
    use_device_map = args.device_map != "none"
    if use_device_map and not torch.cuda.is_available():
        raise RuntimeError("--device-map requires CUDA; use --device-map none on CPU/MPS.")
    dtype = torch.bfloat16 if use_device_map or device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    load_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "attn_implementation": "sdpa",
    }
    if use_device_map:
        load_kwargs.update(
            device_map=args.device_map,
            max_memory=resolve_gpu_memory_limits(args.gpu_memory),
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if not use_device_map:
        model.to(device)
    model.eval()
    configure_model_generation(model, args.model, args.repetition_penalty)
    enable_fp32_output_projection(model)
    input_device = model.get_input_embeddings().weight.device

    if args.evaluate_results_json is not None:
        results = parse_results(args.evaluate_results_json)
        score_totals: dict[str, list[int]] = {}
        report_lines = [
            "# Partial-answer ratings",
            "",
            "Scores use a 0-10 scale and reflect only the visible answer text.",
            "",
        ]
        table_headers = ["Query"] + [label for label, _ in results[0][1]]
        report_lines.append("| " + " | ".join(table_headers) + " |")
        report_lines.append("| " + " | ".join("---" for _ in table_headers) + " |")

        for query_index, (query, answers) in enumerate(results, start=1):
            evaluation_input = tokenizer.apply_chat_template(
                [{"role": "user", "content": build_evaluation_prompt(query, answers)}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            ).to(input_device)
            with torch.inference_mode():
                evaluation_ids = model.generate(
                    input_ids=evaluation_input,
                    max_new_tokens=500,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            evaluation_text = tokenizer.decode(
                evaluation_ids[0, evaluation_input.shape[1] :],
                skip_special_tokens=True,
            )
            evaluations = parse_evaluation(evaluation_text, len(answers))
            scores = []
            for (label, _), evaluation in zip(answers, evaluations):
                score = evaluation["score"]
                score_totals.setdefault(label, []).append(score)
                scores.append(str(score))
            report_lines.append(
                "| " + " | ".join([str(query_index), *scores]) + " |"
            )

        report_lines.extend(("", "## Method averages", ""))
        report_lines.append("| Method | Average rating |")
        report_lines.append("| --- | ---: |")
        for label, scores in score_totals.items():
            report_lines.append(f"| {label} | {sum(scores) / len(scores):.2f} |")
        args.evaluation_output.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        print(args.evaluation_output.read_text(encoding="utf-8"), end="")
        return

    blocks = find_decoder_blocks(model)
    global_attention_layers = find_global_attention_layer_indices(model, len(blocks))
    gate_signature = "".join(
        signature
        for threshold, signature in (
            (args.pre_margin_thres, f"-pre{args.pre_margin_thres}"),
            (
                args.post_margin_thres,
                f"-post{args.post_margin_thres[0]},{args.post_margin_thres[1]}"
                if args.post_margin_thres is not None
                else "",
            ),
            (
                args.cosine_reject,
                f"-cos{args.cosine_reject}-k{args.cosine_top_k}",
            ),
            (
                args.ada_recirculate if args.ada_recirculate else None,
                f"-ada{args.ada_recirculate}",
            ),
            (
                args.post_margin_ratio_thres,
                f"-postr{args.post_margin_ratio_thres}"
                if args.post_margin_ratio_thres is not None
                else "",
            ),
            (
                args.noise_level_range if args.noise_level_range[1] > 0 else None,
                f"-noise{args.noise_level_range[0]},{args.noise_level_range[1]}"
                f"-ndecay{args.noise_decay_per_pass}",
            ),
        )
        if threshold is not None
    )
    cutoff_signature = (
        f"-cutoff{args.no_recirculate_after_tokens}"
        if args.no_recirculate_after_tokens is not None
        else ""
    )
    repetition_penalty_signature = (
        f"-rpen{args.repetition_penalty}"
        if args.repetition_penalty is not None
        else ""
    )
    if args.output is None:
        output_args = argparse.Namespace(
            **{**vars(args), "pairs": output_recirculation_pairs(args)}
        )
        pairs = resolve_recirculation_pairs(
            output_args, len(blocks), global_attention_layers
        )
        model_slug = args.model.rsplit("/", 1)[-1].lower()
        pair_slug = "_".join(f"{source}-{destination}" for source, destination in pairs)
        query_signature = "" if args.query_index_signature == "all" else f"-{args.query_index_signature}"
        game24_signature = (
            f"-game24-{format_index_ranges(args.game24_index) or 'all'}"
            if args.game24_file is not None
            else "-game24"
            if args.game24_puzzle is not None
            else ""
        )
        countdown_signature = (
            f"-countdown-{format_index_ranges(args.countdown_index) or 'all'}"
            if args.countdown_file is not None
            else (
                f"-countdown-{args.countdown_puzzle[0]}-"
                f"{','.join(map(str, args.countdown_puzzle[1:]))}"
                if args.countdown_puzzle is not None
                else ""
            )
        )
        sudoku_signature = (
            f"-sudoku-{format_index_ranges(args.sudoku_index) or 'all'}"
            if args.do_sudoku
            else ""
        )
        bbeh_signature = (
            f"-bbeh-{format_index_ranges(args.bbeh_index) or 'all'}"
            if args.bbeh_file is not None
            else ""
        )
        knowedit_signature = (
            f"-knowedit-{args.knowedit_file.stem}-{format_index_ranges(args.knowedit_index) or 'all'}"
            if args.knowedit_file is not None
            else ""
        )
        args.output = Path(
            f"{model_slug}-{pair_slug}-passes{args.passes}"
            f"-tokens{args.max_new_tokens}"
            f"{gate_signature}{cutoff_signature}{repetition_penalty_signature}"
            f"{game24_signature}{countdown_signature}{sudoku_signature}{bbeh_signature}{knowedit_signature}"
            f"{query_signature}.json"
        )
    if args.similarities_output is None:
        args.similarities_output = args.output.with_name(
            f"{args.output.stem}-debug{args.output.suffix}"
        )

    def model_step(
        active_model: nn.Module, token: Tensor, cache: DynamicCache
    ) -> tuple[Tensor, DynamicCache]:
        past_length = cache.get_seq_length()
        token_length = token.shape[1]
        cache_position = torch.arange(
            past_length,
            past_length + token_length,
            device=token.device,
        )
        attention_mask = torch.ones(
            token.shape[0],
            past_length + token_length,
            dtype=torch.long,
            device=token.device,
        )
        with torch.inference_mode():
            outputs = active_model(
                input_ids=token,
                attention_mask=attention_mask,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
        return outputs.logits, outputs.past_key_values

    def student_step(token: Tensor, cache: DynamicCache) -> tuple[Tensor, DynamicCache]:
        return model_step(model, token, cache)

    eos_token_ids = model.generation_config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    eos_token_ids = set(eos_token_ids or [])
    def decoded_latent_stats(
        token: Tensor,
        source_index: int,
        latent: Tensor,
        cache: DynamicCache,
        rewind_state: Any,
    ) -> dict[str, Any]:
        replay_cache = rewind_dynamic_cache(copy.deepcopy(cache))
        restore_dynamic_cache_rewind_state(replay_cache, rewind_state)
        output_embeddings = model.get_output_embeddings()
        projection_inputs: list[Tensor] = []
        projection_outputs: list[Tensor] = []

        def inject_source(_module: nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            return (
                latent.to(device=inputs[0].device, dtype=inputs[0].dtype),
                *inputs[1:],
            )

        def capture_projection(
            _module: nn.Module, inputs: tuple[Any, ...], output: Tensor
        ) -> None:
            projection_inputs.append(inputs[0][:, -1, :].detach())
            projection_outputs.append(output[:, -1, :].detach())

        handle = blocks[source_index + 1].register_forward_pre_hook(inject_source)
        projection_handle = output_embeddings.register_forward_hook(capture_projection)
        try:
            logits, _ = student_step(token, replay_cache)
        finally:
            handle.remove()
            projection_handle.remove()
        probabilities = torch.softmax(logits[:, -1, :].float(), dim=-1)
        top_two = torch.topk(probabilities, k=2, dim=-1)
        margin = float((top_two.values[..., 0] - top_two.values[..., 1]).item())
        stats: dict[str, Any] = {
            "margin": margin,
            "topk_tokens": top_k_decoded_tokens(
                logits[:, -1, :], tokenizer, args.noise_injected_source_top_k
            ),
        }
        if margin == 0.0 and projection_inputs and projection_outputs:
            token_indices = top_two.indices[0]
            projection_device = output_embeddings.weight.device
            projection_input = projection_inputs[-1][0].to(
                device=projection_device, dtype=torch.float32
            )
            weight_indices = token_indices.to(projection_device)
            selected_weight = output_embeddings.weight[weight_indices].float()
            fp32_logits = torch.mv(selected_weight, projection_input)
            bias = getattr(output_embeddings, "bias", None)
            if bias is not None:
                fp32_logits += bias[weight_indices].float()
            output_indices = token_indices.to(projection_outputs[-1].device)
            bf16_logits = projection_outputs[-1][0, output_indices].float()
            stats["zero_gap_bf16_diagnostic"] = {
                "fp32_before_bf16": {
                    "z1": float(fp32_logits[0].item()),
                    "z2": float(fp32_logits[1].item()),
                    "gap": float((fp32_logits[0] - fp32_logits[1]).item()),
                },
                "after_bf16": {
                    "dtype": str(projection_outputs[-1].dtype),
                    "z1": float(bf16_logits[0].item()),
                    "z2": float(bf16_logits[1].item()),
                    "gap": float((bf16_logits[0] - bf16_logits[1]).item()),
                },
            }
        return stats

    def probe_periodic_perturbation(
        token: Tensor,
        source_index: int,
        destination_index: int,
        destination: Tensor,
        candidate_source: Tensor,
        cache: DynamicCache,
        rewind_state: Any,
        config: RecirculationConfig,
    ) -> Tensor:
        replay_cache = rewind_dynamic_cache(copy.deepcopy(cache))
        restore_dynamic_cache_rewind_state(replay_cache, rewind_state)
        captured_outputs: list[Tensor] = []

        def inject_candidate(
            _module: nn.Module, inputs: tuple[Any, ...]
        ) -> tuple[Any, ...]:
            beta = 1.0 - config.alpha if config.beta is None else config.beta
            mixed = (
                beta * destination.to(device=inputs[0].device, dtype=torch.float32)
                + config.alpha
                * candidate_source.to(device=inputs[0].device, dtype=torch.float32)
            ).to(inputs[0].dtype)
            return (mixed, *inputs[1:])

        def capture_output(
            _module: nn.Module, _inputs: tuple[Any, ...], output: Any
        ) -> None:
            hidden_states = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden_states, Tensor):
                raise TypeError("A transformer block must return hidden states.")
            captured_outputs.append(hidden_states[:, -1, :].detach())

        injection_handle = blocks[destination_index + 1].register_forward_pre_hook(
            inject_candidate
        )
        output_handle = blocks[source_index].register_forward_hook(capture_output)
        try:
            logits, _ = student_step(token, replay_cache)
        finally:
            injection_handle.remove()
            output_handle.remove()
        if config.perturbation_target_token_id is not None:
            return logits[:, -1, :]
        if not captured_outputs:
            raise RuntimeError("Periodic perturbation probe did not capture a source residual.")
        return captured_outputs[-1]

    def distribution_similarity(
        teacher_logits: Tensor, student_logits: Tensor
    ) -> dict[str, float | int | str]:
        if teacher_logits.shape[-1] != student_logits.shape[-1]:
            raise ValueError(
                "Teacher and student tokenizers must have the same vocabulary size."
            )
        student_prob = torch.softmax(student_logits.float(), dim=-1)
        teacher_prob = torch.softmax(
            teacher_logits.to(student_logits.device).float(), dim=-1
        )
        midpoint = 0.5 * (teacher_prob + student_prob)
        teacher_kl = torch.sum(
            teacher_prob
            * (torch.log(teacher_prob.clamp_min(1e-12))
               - torch.log(midpoint.clamp_min(1e-12))),
            dim=-1,
        )
        student_kl = torch.sum(
            student_prob
            * (torch.log(student_prob.clamp_min(1e-12))
               - torch.log(midpoint.clamp_min(1e-12))),
            dim=-1,
        )
        js_divergence = 0.5 * (teacher_kl + student_kl)
        cosine = torch.nn.functional.cosine_similarity(
            teacher_prob, student_prob, dim=-1
        )
        teacher_top_two = torch.topk(teacher_prob, k=2, dim=-1).indices
        student_top_two = torch.topk(student_prob, k=2, dim=-1).indices
        teacher_top = teacher_top_two[..., 0]
        student_top = student_top_two[..., 0]
        return {
            "cosine_similarity": round(float(cosine.item()), 3),
            "js_similarity": round(float((1.0 - js_divergence / torch.log(
                torch.tensor(2.0, device=js_divergence.device)
            )).clamp(0.0, 1.0).item()), 3),
            "teacher_top_token": tokenizer.decode(teacher_top),
            "teacher_top2_token": tokenizer.decode(teacher_top_two[..., 1]),
            "student_top_token": tokenizer.decode(student_top),
            "student_top2_token": tokenizer.decode(student_top_two[..., 1]),
            "top1_agreement": int((teacher_top == student_top).item()),
        }

    def generate(
        use_recirculation: bool,
        run_args: argparse.Namespace,
        run_config: RecirculationConfig,
        target_token_id: int | None = None,
        on_generated_token: Callable[[Tensor], None] | None = None,
        on_debug_comparison: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[Tensor, list[dict[str, Any]], dict[str, Any]]:
        set_random_seed(run_args.seed)
        similarity_stats = (
            SimilarityStats() if run_args.debug_layer_sim else None
        )
        adjacent_layer_stats = (
            AdjacentLayerSimilarityStats() if run_args.debug_adj_layer_sim else None
        )
        condition_thresholds = (
            run_args.act_sim_thres
            if run_args.cond_recirculate
            else None
        )

        student_cache = DynamicCache(config=model.config)
        if use_recirculation:
            student_cache.activate_past_recording()
        # Probe the residual streams with a single pass so the similarity stats
        # are also collected for runs without recirculation.
        def probe_step(
            tokens: Tensor, cache: DynamicCache
        ) -> tuple[Tensor, DynamicCache]:
            return recirculate(
                tokens,
                blocks=blocks,
                cache=cache,
                step=student_step,
                rewind_one=rewind_dynamic_cache,
                config=dataclasses.replace(run_config, mode="source"),
                passes=1,
                rewind_layer=rewind_dynamic_cache_layer,
                similarity_stats=similarity_stats,
                adjacent_layer_stats=adjacent_layer_stats,
            )

        plain_step = (
            student_step
            if similarity_stats is None and adjacent_layer_stats is None
            else probe_step
        )

        periodic_perturbation_centroids: list[Tensor] = []

        def recirculate_prompt(**kwargs: Any) -> tuple[Tensor, DynamicCache]:
            if (
                not use_recirculation
                or run_args.perturb_every_n_tokens != 1
                or run_args.max_new_tokens == 0
            ):
                return recirculate(input_ids, **kwargs)
            if input_ids.shape[1] > 1:
                _, kwargs["cache"] = recirculate(input_ids[:, :-1], **kwargs)
            centroid = (
                recent_token_centroid(
                    model.get_input_embeddings(), input_ids, run_args.perturb_recent_m_tokens
                )
                if run_args.perturb_mode == "repel-history"
                else None
            )
            if centroid is not None:
                periodic_perturbation_centroids.append(centroid)
            kwargs["config"] = dataclasses.replace(
                run_config,
                noise_level_range=periodic_perturbation_noise_level_range(
                    run_config.noise_level_range,
                    tuple(run_args.perturb_noise_level_range)
                    if run_args.perturb_noise_level_range is not None
                    else None,
                ),
                perturbation_direction=(
                    periodic_perturbation_history_direction(
                        periodic_perturbation_centroids, run_args.perturb_history_decay
                    )
                    if centroid is not None
                    else None
                ),
                perturbation_target_token_id=(
                    target_token_id if run_args.perturb_mode == "towards-target" else None
                ),
            )
            kwargs["force_recirculation"] = True
            kwargs["adaptive_recirculation"] = effective_adaptive_recirculation(
                run_args.ada_recirculate, run_args.cond_recirculate, True
            )
            kwargs["perturbation_probe"] = probe_periodic_perturbation
            return recirculate(input_ids[:, -1:], **kwargs)

        recirculated_flags: list[bool] = []
        rejected_flags: list[bool] = []
        adaptive_recirculated_flags: list[bool] = []
        adaptive_rejected_flags: list[bool] = []
        adaptive_recirculation_counts: list[int] = []
        final_pass_same_top1_flags: list[bool] = []
        rejection_reasons: list[tuple[str, ...]] = []
        generated_recirculated_flags: list[bool] = []
        generated_rejected_flags: list[bool] = []
        generated_adaptive_recirculated_flags: list[bool] = []
        generated_adaptive_rejected_flags: list[bool] = []
        generated_adaptive_recirculation_counts: list[int] = []
        generated_final_pass_same_top1_flags: list[bool] = []
        generated_rejection_reasons: list[tuple[str, ...]] = []
        def record_generated_token_stats() -> None:
            generated_recirculated_flags.append(
                recirculated_flags[-1] if recirculated_flags else False
            )
            generated_rejected_flags.append(
                rejected_flags[-1] if rejected_flags else False
            )
            generated_adaptive_recirculated_flags.append(
                adaptive_recirculated_flags[-1]
                if adaptive_recirculated_flags
                else False
            )
            generated_adaptive_rejected_flags.append(
                adaptive_rejected_flags[-1] if adaptive_rejected_flags else False
            )
            generated_adaptive_recirculation_counts.append(
                adaptive_recirculation_counts[-1]
                if adaptive_recirculation_counts
                else 0
            )
            generated_final_pass_same_top1_flags.append(
                final_pass_same_top1_flags[-1]
                if final_pass_same_top1_flags
                else False
            )
            generated_rejection_reasons.append(
                rejection_reasons[-1] if rejection_reasons else ()
            )

        def generated_stats() -> dict[str, Any]:
            return summarize_recirculation_stats(
                generated_recirculated_flags,
                generated_rejected_flags,
                generated_adaptive_recirculated_flags,
                generated_adaptive_rejected_flags,
                generated_adaptive_recirculation_counts,
                generated_final_pass_same_top1_flags,
                generated_rejection_reasons,
            )

        def allow_generated_recirculation(generated_token_count: int) -> bool:
            limit = run_args.no_recirculate_after_tokens
            return limit is None or generated_token_count < limit

        def report_stats(
            recirculated_flags: list[bool] | None = None,
            rejected_flags: list[bool] | None = None,
            adaptive_recirculated_flags: list[bool] | None = None,
            adaptive_rejected_flags: list[bool] | None = None,
            adaptive_recirculation_counts: list[int] | None = None,
            final_pass_same_top1_flags: list[bool] | None = None,
            rejection_reasons: list[tuple[str, ...]] | None = None,
        ) -> None:
            if recirculated_flags is not None:
                rejection_breakdown = ""
                if rejection_reasons is not None:
                    enabled_reasons = tuple(
                        reason
                        for enabled, reason in (
                            (
                                run_args.ada_recirculate
                                and run_args.post_margin_thres is not None,
                                "margin-narrowed",
                            ),
                            (run_args.post_margin_thres, "post-margin-min"),
                            (
                                run_args.post_margin_ratio_thres,
                                "post-margin-ratio",
                            ),
                            (run_args.cosine_reject, "cosine"),
                        )
                        if enabled
                    )
                    if enabled_reasons:
                        rejection_breakdown = ", by gate: " + ", ".join(
                            f"{reason}={sum(reason in reasons for reasons in rejection_reasons)}"
                            for reason in enabled_reasons
                        )
                print(
                    f"recirculated_tokens = {sum(recirculated_flags)}/"
                    f"{len(recirculated_flags)}, same_top1 = "
                    f"{sum(final_pass_same_top1_flags or [])}, rejected = "
                    f"{sum(rejected_flags or [])}{rejection_breakdown}"
                )
            if adaptive_recirculated_flags is not None:
                counts = [
                    count
                    for count in adaptive_recirculation_counts or []
                    if count > 0
                ]
                average_count = sum(counts) / len(counts) if counts else 0.0
                print(
                    "adaptive_recirculated_tokens = "
                    f"{sum(adaptive_recirculated_flags)}/"
                    f"{len(adaptive_recirculated_flags)}, rejected = "
                    f"{sum(adaptive_rejected_flags or [])}, "
                    f"average_adaptive_recirculations = {average_count:.2f}"
                )
            summary = similarity_stats.summary() if similarity_stats else None
            if summary is not None:
                formatted = ", ".join(
                    f"{name}={value:.3f}" if name != "count" else f"{name}={value:.0f}"
                    for name, value in summary.items()
                )
                print(
                    f"src_dst_similarity {list(run_config.pairs)}: {formatted}"
                )
            if adjacent_layer_stats is not None:
                for layer_index, layer_summary in enumerate(
                    adjacent_layer_stats.summaries()
                ):
                    if layer_summary is None:
                        continue
                    formatted = ", ".join(
                        f"{name}={value:.3f}" if name != "count" else f"{name}={value:.0f}"
                        for name, value in layer_summary.items()
                    )
                    print(
                        f"adjacent_layer_similarity {layer_index}-"
                        f"{layer_index + 1}: {formatted}"
                    )

        if not args.debug:
            if use_recirculation:
                # When not debug, this recirculate() processes input_ids and produces logits for the first generated token, 
                # using the configured recirculation gates.
                prompt_logits, student_cache = recirculate_prompt(
                    blocks=blocks,
                    cache=student_cache,
                    step=student_step,
                    rewind_one=rewind_dynamic_cache,
                    config=run_config,
                    similarity_stats=similarity_stats,
                    adjacent_layer_stats=adjacent_layer_stats,
                    passes=run_args.passes,
                    rewind_layer=rewind_dynamic_cache_layer,
                    condition_thresholds=condition_thresholds,
                    pre_margin_threshold=run_args.pre_margin_thres,
                    post_margin_threshold=run_args.post_margin_thres,
                    post_margin_ratio_threshold=run_args.post_margin_ratio_thres,
                    adaptive_recirculation=effective_adaptive_recirculation(
                        run_args.ada_recirculate,
                        run_args.cond_recirculate,
                    ),
                    recirculation_allowed=True,
                    cosine_reject=run_args.cosine_reject,
                    cosine_top_k=run_args.cosine_top_k,
                    gating_pair_index=run_args.gating_pair_index,
                    recirculated_flags=recirculated_flags,
                    rejected_flags=rejected_flags,
                    adaptive_recirculated_flags=adaptive_recirculated_flags,
                    adaptive_rejected_flags=adaptive_rejected_flags,
                    adaptive_recirculation_counts=adaptive_recirculation_counts,
                    final_pass_same_top1_flags=final_pass_same_top1_flags,
                    rejection_reasons=rejection_reasons,
                    decode_injected_source_latent=(
                        decoded_latent_stats
                        if run_config.noise_level_range[1] > 0
                        else None
                    ),
                    capture_cached_token=capture_dynamic_cache_token,
                    restore_cached_token=restore_dynamic_cache_token,
                    capture_rewind_state=capture_dynamic_cache_rewind_state,
                    restore_rewind_state=restore_dynamic_cache_rewind_state,
                    finalize_token_cache=finalize_dynamic_cache_token,
                )
                next_logits = prompt_logits[:, -1, :]

            generated_ids = input_ids.clone()
            active_repetition_counts: dict[tuple[str, ...], int] = {}
            repetition_recovery_tokens_remaining = 0
            repetition_recovery_penalty_tokens_remaining = 0
            consecutive_repetition_count = 0
            for _ in range(run_args.max_new_tokens):
                effective_repetition_penalty = repetition_recovery_penalty(
                    repetition_recovery_penalty_tokens_remaining > 0
                )
                next_token = sample_token(
                    apply_repetition_penalty(
                        next_logits,
                        generated_ids,
                        effective_repetition_penalty,
                        recirculated=bool(
                            recirculated_flags and recirculated_flags[-1]
                        ) and repetition_recovery_penalty_tokens_remaining == 0,
                    ),
                    run_args.temperature,
                )
                repetition_recovery_penalty_tokens_remaining = max(
                    0, repetition_recovery_penalty_tokens_remaining - 1
                )
                generated_ids = torch.cat((generated_ids, next_token), dim=1)
                if on_generated_token is not None:
                    on_generated_token(next_token)
                record_generated_token_stats()
                generated_token_ids = generated_ids[0, input_ids.shape[1] :]
                generated_text = tokenizer.decode(
                    generated_token_ids, skip_special_tokens=True
                )
                repetition_counts = repeated_text_signature_counts(
                    repetition_text_window(generated_text)
                )
                repetition_present = bool(repetition_counts)
                repetition_detected = any(
                    count > active_repetition_counts.get(signature, 0)
                    for signature, count in repetition_counts.items()
                )
                if repetition_detected:
                    print("\nrepetition detected", flush=True)
                active_repetition_counts = repetition_counts
                consecutive_repetition_count += int(repetition_detected)
                if repetition_detected and run_args.repetition_recovery:
                    repetition_recovery_tokens_remaining = (
                        REPETITION_RECOVERY_TOKEN_COUNT
                    )
                    repetition_recovery_penalty_tokens_remaining = (
                        REPETITION_RECOVERY_TOKEN_COUNT
                    )
                if next_token.item() in eos_token_ids:
                    break
                repetition_recovery_active = repetition_recovery_tokens_remaining > 0
                if use_recirculation:
                    generated_token_count = generated_ids.shape[1] - input_ids.shape[1]
                    periodic_perturbation = periodic_perturbation_active(
                        run_args.perturb_every_n_tokens,
                        run_args.perturb_for_k_tokens,
                        generated_token_count,
                    )
                    periodic_perturbation_direction = None
                    repetition_recovery_direction = None
                    if (
                        periodic_perturbation
                        and generated_token_count % run_args.perturb_every_n_tokens == 0
                    ):
                        print(f"(perturb at {generated_token_count})", flush=True)
                        if run_args.perturb_mode == "repel-history":
                            centroid = recent_token_centroid(
                                model.get_input_embeddings(),
                                generated_ids,
                                run_args.perturb_recent_m_tokens,
                            )
                            periodic_perturbation_centroids.append(centroid)
                    if periodic_perturbation:
                        if run_args.perturb_mode == "repel-history":
                            periodic_perturbation_direction = (
                                periodic_perturbation_history_direction(
                                    periodic_perturbation_centroids,
                                    run_args.perturb_history_decay,
                                )
                            )
                    elif repetition_recovery_active:
                        repetition_recovery_direction = -recent_token_centroid(
                            model.get_input_embeddings(),
                            generated_ids,
                            run_args.perturb_recent_m_tokens,
                        )
                    recovery_settings = repetition_recovery_settings(
                        run_config.noise_level_range,
                        run_args.cosine_reject,
                        run_args.pre_margin_thres,
                        tuple(run_args.post_margin_thres)
                        if run_args.post_margin_thres is not None
                        else None,
                        run_args.post_margin_ratio_thres,
                        condition_thresholds,
                        allow_generated_recirculation(generated_token_count),
                        generated_ids[0, -generated_token_count:].tolist(),
                        enabled=run_args.repetition_recovery,
                        decoded_text=generated_text,
                        consecutive_repetition_count=consecutive_repetition_count,
                        repetition_detected=repetition_recovery_active,
                    )
                    token_run_config = dataclasses.replace(
                        run_config,
                        noise_level_range=(
                            periodic_perturbation_noise_level_range(
                                recovery_settings.noise_level_range,
                                (
                                    tuple(run_args.perturb_noise_level_range)
                                    if run_args.perturb_noise_level_range is not None
                                    else None
                                ),
                            )
                            if periodic_perturbation
                            else recovery_settings.noise_level_range
                        ),
                        perturbation_direction=(
                            periodic_perturbation_direction
                            if periodic_perturbation
                            else repetition_recovery_direction
                        ),
                        perturbation_target_token_id=(
                            target_token_id
                            if periodic_perturbation
                            and run_args.perturb_mode == "towards-target"
                            else None
                        ),
                        perturbation_direction_scale=(
                            periodic_perturbation_direction_scale(
                                run_args.perturb_every_n_tokens,
                                generated_token_count,
                            )
                            if periodic_perturbation
                            else 1.0
                        ),
                    )
                    # When not debug, this recirculate() processes each newly selected token to produce logits for the next one. 
                    # It can adjust or force recirculation for repetition recovery and periodic perturbation.
                    token_logits, student_cache = recirculate(
                        next_token,
                        blocks=blocks,
                        cache=student_cache,
                        step=student_step,
                        rewind_one=rewind_dynamic_cache,
                        config=token_run_config,
                        similarity_stats=similarity_stats,
                        adjacent_layer_stats=adjacent_layer_stats,
                        passes=run_args.passes,
                        rewind_layer=rewind_dynamic_cache_layer,
                        condition_thresholds=recovery_settings.condition_thresholds,
                        pre_margin_threshold=recovery_settings.pre_margin_threshold,
                        post_margin_threshold=recovery_settings.post_margin_threshold,
                        post_margin_ratio_threshold=(
                            recovery_settings.post_margin_ratio_threshold
                        ),
                        adaptive_recirculation=effective_adaptive_recirculation(
                            run_args.ada_recirculate,
                            run_args.cond_recirculate,
                            repetition_recovery_active
                            or periodic_perturbation,
                        ),
                        recirculation_allowed=(
                            recovery_settings.recirculation_allowed
                            or periodic_perturbation
                        ),
                        force_recirculation=(
                            recovery_settings.force_recirculation
                            or periodic_perturbation
                        ),
                        cosine_reject=recovery_settings.cosine_reject,
                        cosine_top_k=run_args.cosine_top_k,
                        gating_pair_index=run_args.gating_pair_index,
                        recirculated_flags=recirculated_flags,
                        rejected_flags=rejected_flags,
                        adaptive_recirculated_flags=adaptive_recirculated_flags,
                        adaptive_rejected_flags=adaptive_rejected_flags,
                        adaptive_recirculation_counts=adaptive_recirculation_counts,
                        final_pass_same_top1_flags=final_pass_same_top1_flags,
                        rejection_reasons=rejection_reasons,
                        decode_injected_source_latent=(
                            decoded_latent_stats
                            if token_run_config.noise_level_range[1] > 0
                            else None
                        ),
                        perturbation_probe=probe_periodic_perturbation,
                        capture_cached_token=capture_dynamic_cache_token,
                        restore_cached_token=restore_dynamic_cache_token,
                        capture_rewind_state=capture_dynamic_cache_rewind_state,
                        restore_rewind_state=restore_dynamic_cache_rewind_state,
                        finalize_token_cache=finalize_dynamic_cache_token,
                    )
                else:
                    token_logits, student_cache = plain_step(
                        next_token, student_cache
                    )
                repetition_recovery_tokens_remaining = max(
                    0, repetition_recovery_tokens_remaining - 1
                )
                next_logits = token_logits[:, -1, :]

            print()
            report_stats(
                generated_recirculated_flags,
                generated_rejected_flags,
                generated_adaptive_recirculated_flags,
                generated_adaptive_rejected_flags,
                generated_adaptive_recirculation_counts,
                generated_final_pass_same_top1_flags,
                generated_rejection_reasons,
            )
            return generated_ids, [], generated_stats()

        similarities: list[dict[str, Any]] = []
        generated_ids = input_ids.clone()
        first_pass_logits: list[Tensor] = []
        first_pass_similarities: list[tuple[float, ...]] = []
        pass_probability_margins: list[list[float]] = []
        final_pass_cosine_similarities: list[float | None] = []
        pass_cosine_similarities: list[list[float]] = []
        pass_top_k_token_ids: list[list[list[int]]] = []
        pass_target_token_probabilities: list[list[float]] = []
        candidate_target_token_probabilities: list[list[float]] = []
        aggregate_target_token_probabilities: list[list[float]] = []
        cosine_reject_thresholds: list[float | None] = []
        injected_noise_levels: list[list[float]] = []
        initial_decoded_noise_source_latents: list[list[dict[str, Any]]] = []
        decoded_injected_source_latents: list[list[dict[str, Any]]] = []
        # When debug, this recirculate() does the prompt work while also collecting first-pass logits, margins, 
        # similarities, and noise diagnostics for comparison.
        student_logits, student_cache = recirculate_prompt(
            blocks=blocks,
            cache=student_cache,
            step=student_step,
            rewind_one=rewind_dynamic_cache,
            config=run_config,
            similarity_stats=similarity_stats,
            adjacent_layer_stats=adjacent_layer_stats,
            passes=run_args.passes if use_recirculation else 1,
            rewind_layer=rewind_dynamic_cache_layer,
            condition_thresholds=condition_thresholds,
            pre_margin_threshold=run_args.pre_margin_thres,
            post_margin_threshold=run_args.post_margin_thres,
            post_margin_ratio_threshold=run_args.post_margin_ratio_thres,
            adaptive_recirculation=effective_adaptive_recirculation(
                run_args.ada_recirculate,
                run_args.cond_recirculate,
            ),
            recirculation_allowed=True,
            cosine_reject=run_args.cosine_reject,
            cosine_top_k=run_args.cosine_top_k,
            gating_pair_index=run_args.gating_pair_index,
            first_pass_logits=first_pass_logits,
            first_pass_similarities=first_pass_similarities,
            pass_probability_margins=pass_probability_margins,
            recirculated_flags=recirculated_flags,
            rejected_flags=rejected_flags,
            adaptive_recirculated_flags=adaptive_recirculated_flags,
            adaptive_rejected_flags=adaptive_rejected_flags,
            adaptive_recirculation_counts=adaptive_recirculation_counts,
            final_pass_same_top1_flags=final_pass_same_top1_flags,
            rejection_reasons=rejection_reasons,
            final_pass_cosine_similarities=final_pass_cosine_similarities,
            pass_cosine_similarities=pass_cosine_similarities,
            pass_top_k_token_ids=pass_top_k_token_ids,
            target_token_id=target_token_id,
            pass_target_token_probabilities=pass_target_token_probabilities,
            candidate_target_token_probabilities=candidate_target_token_probabilities,
            aggregate_target_token_probabilities=aggregate_target_token_probabilities,
            cosine_reject_thresholds=cosine_reject_thresholds,
            injected_noise_levels=injected_noise_levels,
            decode_injected_source_latent=decoded_latent_stats,
            initial_decoded_noise_source_latents=(
                initial_decoded_noise_source_latents
            ),
            decoded_injected_source_latents=decoded_injected_source_latents,
            capture_cached_token=capture_dynamic_cache_token,
            restore_cached_token=restore_dynamic_cache_token,
            capture_rewind_state=capture_dynamic_cache_rewind_state,
            restore_rewind_state=restore_dynamic_cache_rewind_state,
            finalize_token_cache=finalize_dynamic_cache_token,
        )
        teacher_logits = first_pass_logits[-1]
        teacher_src_dst_sim = sum(first_pass_similarities[-1]) / len(run_config.pairs)
        teacher_next_logits = teacher_logits[:, -1, :]
        student_next_logits = student_logits[:, -1, :]
        current_token = input_ids[:, -1:]
        student_run_config = run_config
        active_repetition_counts: dict[tuple[str, ...], int] = {}
        repetition_recovery_active = False
        repetition_recovery_tokens_remaining = 0
        repetition_recovery_penalty_tokens_remaining = 0
        consecutive_repetition_count = 0

        for token_index in range(run_args.max_new_tokens):
            comparison = distribution_similarity(teacher_next_logits, student_next_logits)
            record_generated_token_stats()
            comparison["teacher_src_dst_sim"] = round(teacher_src_dst_sim, 3)
            comparison["top1_top2_margin"] = [
                round(margin, 3) for margin in pass_probability_margins[-1]
            ]
            comparison["top1_prob"] = round(
                float(torch.softmax(teacher_next_logits.float(), dim=-1).max().item()), 3
            )
            comparison["recirculated"] = generated_recirculated_flags[-1]
            comparison["rejected"] = generated_rejected_flags[-1]
            comparison["adaptive_recirculated"] = (
                generated_adaptive_recirculated_flags[-1]
            )
            comparison["adaptive_rejected"] = generated_adaptive_rejected_flags[-1]
            comparison["adaptive_recirculation_count"] = (
                generated_adaptive_recirculation_counts[-1]
            )
            comparison["final_pass_same_top1"] = (
                generated_final_pass_same_top1_flags[-1]
            )
            comparison["rejection_reasons"] = generated_rejection_reasons[-1]
            comparison["final_pass_top_k_cosine_similarity"] = (
                round(final_pass_cosine_similarities[-1], 6)
                if final_pass_cosine_similarities[-1] is not None
                else None
            )
            comparison["pass_top_k_cosine_similarities"] = [
                round(cosine, 6) for cosine in pass_cosine_similarities[-1]
            ]
            comparison["pass_recirculation_topk_tokens"] = [
                [tokenizer.decode(token_id) for token_id in attempt]
                for attempt in pass_top_k_token_ids[-1]
            ]
            if target_token_id is not None:
                comparison["target_token"] = tokenizer.decode(target_token_id)
                comparison["pre_recirculation_target_token_probability"] = float(
                    torch.softmax(teacher_next_logits[0].float(), dim=-1)[target_token_id].item()
                )
                candidates_per_step = student_run_config.periodic_perturbation_candidate_count
                probabilities = candidate_target_token_probabilities[-1]
                comparison["candidate_target_token_probabilities"] = [
                    [f"{probability:.2e}" for probability in probabilities[start:start + candidates_per_step]]
                    for start in range(0, len(probabilities), candidates_per_step)
                ]
                comparison["aggregate_target_token_probabilities"] = [
                    f"{probability:.2e}"
                    for probability in aggregate_target_token_probabilities[-1]
                ]
                comparison["final_pass_target_token_probability"] = (
                    f"{pass_target_token_probabilities[-1][-1]:.2e}"
                    if pass_target_token_probabilities[-1]
                    else None
                )
                comparison["post_recirculation_target_token_probability"] = (
                    f"{torch.softmax(student_next_logits[0].float(), dim=-1)[target_token_id].item():.2e}"
                )
            comparison["cosine_reject_threshold"] = cosine_reject_thresholds[-1]
            comparison["cosine_top_k"] = run_args.cosine_top_k
            comparison["post_recirculation_topk_tokens"] = top_k_decoded_tokens(
                student_next_logits, tokenizer, run_args.cosine_top_k
            )
            decoded_noise_latents = decoded_injected_source_latents[-1]
            zero_gap_diagnostics = [
                stats["zero_gap_bf16_diagnostic"]
                for stats in decoded_noise_latents
                if "zero_gap_bf16_diagnostic" in stats
            ]
            if zero_gap_diagnostics:
                comparison["injected_source_zero_gap_bf16_diagnostics"] = (
                    zero_gap_diagnostics
                )
            if student_run_config.noise_level_range[1] > 0:
                initial_decoded_noise_latents = (
                    initial_decoded_noise_source_latents[-1]
                )
                comparison["injected_noise_levels"] = [
                    round(level, 6) for level in injected_noise_levels[-1]
                ]
                comparison["initial_noise_injected_source_top1_top2_margin"] = [
                    f"{float(stats['margin']):.6e}"
                    for stats in initial_decoded_noise_latents
                ]
                comparison["noise_injected_source_top1_top2_margin"] = [
                    f"{float(stats['margin']):.6e}"
                    for stats in decoded_noise_latents
                ]
                comparison["noise_injected_source_topk_tokens"] = [
                    stats["topk_tokens"] for stats in decoded_noise_latents
                ]
            effective_repetition_penalty = repetition_recovery_penalty(
                repetition_recovery_penalty_tokens_remaining > 0
            )
            next_token = sample_token(
                apply_repetition_penalty(
                    student_next_logits,
                    generated_ids,
                    effective_repetition_penalty,
                    recirculated=bool(
                        recirculated_flags and recirculated_flags[-1]
                    ) and repetition_recovery_penalty_tokens_remaining == 0,
                ),
                run_args.temperature,
            )
            repetition_recovery_penalty_tokens_remaining = max(
                0, repetition_recovery_penalty_tokens_remaining - 1
            )
            comparison.update(
                token_index=token_index,
                selected_token=tokenizer.decode(next_token[0]),
                repetition_penalty=effective_repetition_penalty,
            )
            generated_ids = torch.cat((generated_ids, next_token), dim=1)
            if on_generated_token is not None:
                on_generated_token(next_token)

            if next_token.item() in eos_token_ids:
                similarities.append(comparison)
                if on_debug_comparison is not None:
                    on_debug_comparison(comparison)
                break

            generated_token_tensor = generated_ids[0, input_ids.shape[1] :]
            generated_token_ids = generated_token_tensor.tolist()
            generated_text = tokenizer.decode(
                generated_token_tensor, skip_special_tokens=True
            )
            repetition_counts = repeated_text_signature_counts(
                repetition_text_window(generated_text)
            )
            repetition_present = bool(repetition_counts)
            repetition_detected = any(
                count > active_repetition_counts.get(signature, 0)
                for signature, count in repetition_counts.items()
            )
            if repetition_detected:
                print("\nrepetition detected", flush=True)
            active_repetition_counts = repetition_counts
            consecutive_repetition_count += int(repetition_detected)
            if repetition_detected and run_args.repetition_recovery:
                repetition_recovery_tokens_remaining = REPETITION_RECOVERY_TOKEN_COUNT
                repetition_recovery_penalty_tokens_remaining = (
                    REPETITION_RECOVERY_TOKEN_COUNT
                )
            repetition_recovery_active = repetition_recovery_tokens_remaining > 0
            periodic_perturbation = periodic_perturbation_active(
                run_args.perturb_every_n_tokens,
                run_args.perturb_for_k_tokens,
                token_index + 1,
            )
            periodic_perturbation_direction = None
            repetition_recovery_direction = None
            if (
                periodic_perturbation
                and (token_index + 1) % run_args.perturb_every_n_tokens == 0
            ):
                print(f"(perturb at {token_index + 1})", flush=True)
                if run_args.perturb_mode == "repel-history":
                    centroid = recent_token_centroid(
                        model.get_input_embeddings(),
                        generated_ids,
                        run_args.perturb_recent_m_tokens,
                    )
                    periodic_perturbation_centroids.append(centroid)
            if periodic_perturbation:
                if run_args.perturb_mode == "repel-history":
                    periodic_perturbation_direction = (
                        periodic_perturbation_history_direction(
                            periodic_perturbation_centroids,
                            run_args.perturb_history_decay,
                        )
                    )
            elif repetition_recovery_active:
                repetition_recovery_direction = -recent_token_centroid(
                    model.get_input_embeddings(),
                    generated_ids,
                    run_args.perturb_recent_m_tokens,
                )
            recovery_settings = repetition_recovery_settings(
                run_config.noise_level_range,
                run_args.cosine_reject,
                run_args.pre_margin_thres,
                tuple(run_args.post_margin_thres)
                if run_args.post_margin_thres is not None
                else None,
                run_args.post_margin_ratio_thres,
                condition_thresholds,
                allow_generated_recirculation(token_index + 1),
                generated_token_ids,
                enabled=run_args.repetition_recovery,
                decoded_text=generated_text,
                consecutive_repetition_count=consecutive_repetition_count,
                repetition_detected=repetition_recovery_active,
            )
            comparison["repetition_present"] = repetition_present
            comparison["repetition_detected"] = repetition_detected
            comparison["repetition_recovery_attempt"] = (
                consecutive_repetition_count if repetition_detected else None
            )
            comparison["repetition_recovery_active"] = repetition_recovery_active
            comparison["repetition_recovery_noise_level_range"] = (
                recovery_settings.noise_level_range
                if repetition_recovery_active
                else None
            )
            comparison["repetition_recovery_cosine_reject"] = (
                recovery_settings.cosine_reject if repetition_recovery_active else None
            )
            comparison["repetition_recovery_pre_margin_threshold"] = (
                recovery_settings.pre_margin_threshold
                if repetition_recovery_active
                else None
            )
            comparison["repetition_recovery_post_margin_threshold"] = (
                recovery_settings.post_margin_threshold
                if repetition_recovery_active
                else None
            )
            comparison["repetition_recovery_forces_recirculation"] = (
                repetition_recovery_active
            )
            comparison["periodic_perturbation_active"] = periodic_perturbation
            similarities.append(comparison)
            if on_debug_comparison is not None:
                on_debug_comparison(comparison)
            student_run_config = dataclasses.replace(
                run_config,
                noise_level_range=(
                    periodic_perturbation_noise_level_range(
                        recovery_settings.noise_level_range,
                        (
                            tuple(run_args.perturb_noise_level_range)
                            if run_args.perturb_noise_level_range is not None
                            else None
                        ),
                    )
                    if periodic_perturbation
                    else recovery_settings.noise_level_range
                ),
                perturbation_direction=(
                    periodic_perturbation_direction
                    if periodic_perturbation
                    else repetition_recovery_direction
                ),
                perturbation_target_token_id=(
                    target_token_id
                    if periodic_perturbation
                    and run_args.perturb_mode == "towards-target"
                    else None
                ),
                perturbation_direction_scale=(
                    periodic_perturbation_direction_scale(
                        run_args.perturb_every_n_tokens,
                        token_index + 1,
                    )
                    if periodic_perturbation
                    else 1.0
                ),
            )
            # This recirculate() processes the generated token and collects first-pass logits, margins,
            # similarities, and any applicable noise diagnostics for comparison.
            student_logits, student_cache = recirculate(
                next_token,
                blocks=blocks,
                cache=student_cache,
                step=student_step,
                rewind_one=rewind_dynamic_cache,
                config=student_run_config,
                similarity_stats=similarity_stats,
                adjacent_layer_stats=adjacent_layer_stats,
                passes=run_args.passes if use_recirculation else 1,
                rewind_layer=rewind_dynamic_cache_layer,
                condition_thresholds=recovery_settings.condition_thresholds,
                pre_margin_threshold=recovery_settings.pre_margin_threshold,
                post_margin_threshold=recovery_settings.post_margin_threshold,
                post_margin_ratio_threshold=(
                    recovery_settings.post_margin_ratio_threshold
                ),
                adaptive_recirculation=effective_adaptive_recirculation(
                    run_args.ada_recirculate,
                    run_args.cond_recirculate,
                    repetition_recovery_active
                    or periodic_perturbation,
                ),
                recirculation_allowed=(
                    recovery_settings.recirculation_allowed
                    or periodic_perturbation
                ),
                force_recirculation=(
                    recovery_settings.force_recirculation
                    or periodic_perturbation
                ),
                cosine_reject=recovery_settings.cosine_reject,
                cosine_top_k=run_args.cosine_top_k,
                gating_pair_index=run_args.gating_pair_index,
                first_pass_logits=first_pass_logits,
                first_pass_similarities=first_pass_similarities,
                pass_probability_margins=pass_probability_margins,
                recirculated_flags=recirculated_flags,
                rejected_flags=rejected_flags,
                adaptive_recirculated_flags=adaptive_recirculated_flags,
                adaptive_rejected_flags=adaptive_rejected_flags,
                adaptive_recirculation_counts=adaptive_recirculation_counts,
                final_pass_same_top1_flags=final_pass_same_top1_flags,
                rejection_reasons=rejection_reasons,
                final_pass_cosine_similarities=final_pass_cosine_similarities,
                pass_cosine_similarities=pass_cosine_similarities,
                pass_top_k_token_ids=pass_top_k_token_ids,
                target_token_id=target_token_id,
                pass_target_token_probabilities=pass_target_token_probabilities,
                candidate_target_token_probabilities=candidate_target_token_probabilities,
                aggregate_target_token_probabilities=aggregate_target_token_probabilities,
                cosine_reject_thresholds=cosine_reject_thresholds,
                injected_noise_levels=injected_noise_levels,
                decode_injected_source_latent=decoded_latent_stats,
                perturbation_probe=probe_periodic_perturbation,
                initial_decoded_noise_source_latents=(
                    initial_decoded_noise_source_latents
                ),
                decoded_injected_source_latents=decoded_injected_source_latents,
                capture_cached_token=capture_dynamic_cache_token,
                restore_cached_token=restore_dynamic_cache_token,
                capture_rewind_state=capture_dynamic_cache_rewind_state,
                restore_rewind_state=restore_dynamic_cache_rewind_state,
                finalize_token_cache=finalize_dynamic_cache_token,
            )
            repetition_recovery_tokens_remaining = max(
                0, repetition_recovery_tokens_remaining - 1
            )
            current_token = next_token
            teacher_logits = first_pass_logits[-1]
            teacher_src_dst_sim = (
                sum(first_pass_similarities[-1]) / len(run_config.pairs)
            )
            teacher_next_logits = teacher_logits[:, -1, :]
            student_next_logits = student_logits[:, -1, :]

        # recirculated_flags also covers prompt positions and one trailing lookahead
        # call, neither of which produce a comparison entry, so count from
        # `similarities` instead to match the tokens actually reported.
        print()
        report_stats(
            [comparison["recirculated"] for comparison in similarities],
            [comparison["rejected"] for comparison in similarities],
            [comparison["adaptive_recirculated"] for comparison in similarities],
            [comparison["adaptive_rejected"] for comparison in similarities],
            [comparison["adaptive_recirculation_count"] for comparison in similarities],
            [comparison["final_pass_same_top1"] for comparison in similarities],
            [comparison["rejection_reasons"] for comparison in similarities],
        )

        return generated_ids, similarities, generated_stats()

    def synchronize_devices() -> None:
        if torch.cuda.is_available():
            for gpu in range(torch.cuda.device_count()):
                torch.cuda.synchronize(gpu)

    def run_recirculation_config(run_args: argparse.Namespace) -> RecirculationConfig:
        pairs = resolve_recirculation_pairs(
            run_args, len(blocks), global_attention_layers
        )
        run_config = RecirculationConfig(
            pairs=pairs,
            alpha=run_args.alpha,
            beta=run_args.beta,
            noise_level_range=tuple(run_args.noise_level_range),
            periodic_perturbation_candidate_count=run_args.periodic_perturbation_candidate_count,
            periodic_perturbation_steps=run_args.periodic_perturbation_steps,
            periodic_perturbation_step_decay=run_args.periodic_perturbation_step_decay,
            perturb_pre_margin_thres=run_args.perturb_pre_margin_thres,
            noise_decay_per_pass=run_args.noise_decay_per_pass,
            mode=run_args.mode,
        )
        if not run_config.pairs or any(
            not 0 <= destination < source < len(blocks)
            for source, destination in run_config.pairs
        ):
            raise ValueError(
                f"The model has {len(blocks)} decoder blocks, but the requested "
                f"source/destination pairs were {run_config.pairs}."
            )
        if run_config.mode == "layerwise" and len(run_config.pairs) != 1:
            raise ValueError("--mode layerwise requires exactly one --pair.")
        return run_config

    def score_knowedit_target(
        prompt_ids: Tensor,
        target: str,
        run_args: argparse.Namespace,
        use_recirculation: bool,
        score_name: str,
    ) -> float:
        target_ids = tokenizer.encode(
            target.strip(), add_special_tokens=False, return_tensors="pt"
        ).to(input_device)
        cache = DynamicCache(config=model.config)
        if use_recirculation:
            cache.activate_past_recording()
        run_config = run_recirculation_config(run_args)

        def step(tokens: Tensor) -> Tensor:
            nonlocal cache
            logits, cache = recirculate(
                tokens,
                blocks=blocks,
                cache=cache,
                step=student_step,
                rewind_one=rewind_dynamic_cache,
                config=run_config,
                passes=run_args.passes if use_recirculation else 1,
                rewind_layer=rewind_dynamic_cache_layer,
                condition_thresholds=(
                    run_args.act_sim_thres if run_args.cond_recirculate else None
                ),
                pre_margin_threshold=run_args.pre_margin_thres,
                post_margin_threshold=run_args.post_margin_thres,
                post_margin_ratio_threshold=run_args.post_margin_ratio_thres,
                adaptive_recirculation=effective_adaptive_recirculation(
                    run_args.ada_recirculate, run_args.cond_recirculate
                ),
                recirculation_allowed=(
                    cache.get_seq_length() == 0
                    or run_args.no_recirculate_after_tokens is None
                    or cache.get_seq_length() - prompt_ids.shape[1] + 1
                    < run_args.no_recirculate_after_tokens
                ),
                cosine_reject=run_args.cosine_reject,
                cosine_top_k=run_args.cosine_top_k,
                gating_pair_index=run_args.gating_pair_index,
                decode_injected_source_latent=(
                    decoded_latent_stats
                    if run_config.noise_level_range[1] > 0
                    else None
                ),
                capture_cached_token=capture_dynamic_cache_token,
                restore_cached_token=restore_dynamic_cache_token,
                capture_rewind_state=capture_dynamic_cache_rewind_state,
                restore_rewind_state=restore_dynamic_cache_rewind_state,
                finalize_token_cache=finalize_dynamic_cache_token,
            )
            return logits

        def report_top_two(
            token_index: int, top_two: tuple[tuple[int, float], ...]
        ) -> None:
            predictions = ", ".join(
                f"{tokenizer.decode([token_id])!r} ({probability:.4f})"
                for token_id, probability in top_two
            )
            emit(f"{score_name}_token_{token_index + 1}_top2 = {predictions}")

        return teacher_forced_token_accuracy(
            prompt_ids, target_ids, step, on_top_two=report_top_two
        )

    def timed_generate(
        use_recirculation: bool,
        run_args: argparse.Namespace,
        target_token_id: int | None = None,
        on_generated_token: Callable[[Tensor], None] | None = None,
        on_debug_comparison: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[Tensor, list[dict[str, Any]], dict[str, Any], float]:
        run_config = run_recirculation_config(run_args)
        synchronize_devices()
        start = time.perf_counter()
        generated_ids, similarities, stats = generate(
            use_recirculation=use_recirculation,
            run_args=run_args,
            run_config=run_config,
            target_token_id=target_token_id,
            on_generated_token=on_generated_token,
            on_debug_comparison=on_debug_comparison,
        )
        synchronize_devices()
        return generated_ids, similarities, stats, time.perf_counter() - start

    if args.ablations is False or args.ablations is None:
        baseline_options = (
            "model",
            "passes",
            "noise_level_range",
            "noise_decay_per_pass",
            "perturb_pre_margin_thres",
            "cond_recirculate",
            "act_sim_thres",
            "pre_margin_thres",
            "post_margin_thres",
            "post_margin_ratio_thres",
            "ada_recirculate",
            "cosine_reject",
            "cosine_top_k",
            "repetition_recovery",
            "perturb_every_n_tokens",
            "perturb_mode",
            "perturb_for_k_tokens",
            "perturb_recent_m_tokens",
            "perturb_history_decay",
            "periodic_perturbation_candidate_count",
            "periodic_perturbation_steps",
            "periodic_perturbation_step_decay",
            "repetition_penalty",
            "seed",
        )
        baseline_arguments = format_run_arguments(args, baseline_options)
        runs = [(args, True, f"Baseline: {baseline_arguments}")]
    else:
        ablated_options = tuple(
            dict.fromkeys(
                option
                for overrides in args.ablations
                for option, _ in overrides
            )
        )
        label_options = tuple(dict.fromkeys(("model", *ablated_options)))
        baseline_arguments = format_run_arguments(args, label_options)
        # True: use_recirculation is always enabled for both baseline and ablation runs.
        runs = [(args, True, f"Baseline: {baseline_arguments}")]
        for overrides in args.ablations:
            ablation_args = argparse.Namespace(**vars(args))
            for option, value in overrides:
                setattr(ablation_args, option, value)
            validate_run_arguments(ablation_args)
            arguments = format_run_arguments(ablation_args, label_options)
            runs.append((ablation_args, True, f"Ablation: {arguments}"))

    game24_puzzles = (
        (tuple(args.game24_puzzle),)
        if args.game24_puzzle is not None
        else load_game24_puzzles(args.game24_file)
        if args.game24_file is not None
        else None
    )
    game24_indices = (
        resolve_game24_indices(
            args.game24_index or tuple(range(1, len(game24_puzzles) + 1)),
            len(game24_puzzles),
        )
        if game24_puzzles is not None
        else ()
    )
    manual_countdown_puzzle = (
        (args.countdown_puzzle[0], tuple(args.countdown_puzzle[1:]))
        if args.countdown_puzzle is not None
        else None
    )
    countdown_puzzles = (
        (manual_countdown_puzzle,)
        if manual_countdown_puzzle is not None
        else load_countdown_puzzles(args.countdown_file)
        if args.countdown_file is not None
        else None
    )
    countdown_indices = (
        resolve_benchmark_indices(
            args.countdown_index or tuple(range(1, len(countdown_puzzles) + 1)),
            len(countdown_puzzles),
            "Countdown",
        )
        if countdown_puzzles is not None
        else ()
    )
    sudoku_puzzles = (
        load_sudoku_puzzles(args.sudoku_file)
        if args.do_sudoku
        else None
    )
    sudoku_indices = (
        resolve_benchmark_indices(
            args.sudoku_index or tuple(range(1, len(sudoku_puzzles) + 1)),
            len(sudoku_puzzles),
            "Sudoku",
        )
        if sudoku_puzzles is not None
        else ()
    )
    bbeh_examples = (
        load_bbeh_examples(args.bbeh_file)
        if args.bbeh_file is not None
        else None
    )
    bbeh_indices = (
        resolve_benchmark_indices(
            args.bbeh_index or tuple(range(1, len(bbeh_examples) + 1)),
            len(bbeh_examples),
            "BBEH",
        )
        if bbeh_examples is not None
        else ()
    )
    knowedit_examples = (
        load_knowedit_examples(args.knowedit_file)
        if args.knowedit_file is not None
        else None
    )
    knowedit_indices = (
        resolve_benchmark_indices(
            args.knowedit_index or tuple(range(1, len(knowedit_examples) + 1)),
            len(knowedit_examples),
            "KnowEdit",
        )
        if knowedit_examples is not None
        else ()
    )
    prompts = (
        ((1, args.prompt, None, None, None, None, None),)
        if args.prompt is not None
        else tuple(
            (
                index,
                format_game24_prompt(game24_puzzles[index - 1]),
                game24_puzzles[index - 1],
                None,
                None,
                None,
                None,
            )
            for index in game24_indices
        )
        if game24_puzzles is not None
        else tuple(
            (
                index,
                format_countdown_prompt(*countdown_puzzles[index - 1]),
                None,
                countdown_puzzles[index - 1],
                None,
                None,
                None,
            )
            for index in countdown_indices
        )
        if countdown_puzzles is not None
        else tuple(
            (
                index,
                format_sudoku_prompt(sudoku_puzzles[index - 1]),
                None,
                None,
                sudoku_puzzles[index - 1],
                None,
                None,
            )
            for index in sudoku_indices
        )
        if sudoku_puzzles is not None
        else tuple(
            (
                index,
                format_bbeh_prompt(bbeh_examples[index - 1]),
                None,
                None,
                None,
                bbeh_examples[index - 1],
                None,
            )
            for index in bbeh_indices
        )
        if bbeh_examples is not None
        else tuple(
            (
                index,
                format_knowedit_prompt(knowedit_examples[index - 1]),
                None,
                None,
                None,
                None,
                knowedit_examples[index - 1],
            )
            for index in knowedit_indices
        )
        if knowedit_examples is not None
        else tuple(
            (index, EXAMPLE_QUERIES[index - 1], None, None, None, None, None)
            for index in args.query_indices
        )
    )
    output_file = args.output.open("w+", encoding="utf-8") if args.output else None
    similarities_writer = (
        StreamingSimilarityWriter(args.similarities_output)
        if args.debug and args.similarities_output
        else None
    )
    output_records: list[dict[str, Any]] = []

    def emit(text: str) -> None:
        print(text)

    def save_output() -> None:
        if output_file is not None:
            output_file.seek(0)
            json.dump(output_records, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
            output_file.truncate()
            output_file.flush()

    try:
        for (
            prompt_index,
            prompt,
            game24_puzzle,
            countdown_puzzle,
            sudoku_puzzle,
            bbeh_example,
            knowedit_example,
        ) in prompts:
            query_record: dict[str, Any] = {
                "index": prompt_index,
                "prompt": prompt,
                "runs": [],
            }
            if game24_puzzle is not None:
                query_record["game24_puzzle"] = list(game24_puzzle)
            if countdown_puzzle is not None:
                target, numbers = countdown_puzzle
                query_record["countdown_target"] = target
                query_record["countdown_numbers"] = list(numbers)
            if sudoku_puzzle is not None:
                query_record["sudoku_puzzle"] = [list(row) for row in sudoku_puzzle]
            if bbeh_example is not None:
                query_record["bbeh_task"] = bbeh_example.task
                query_record["bbeh_target"] = bbeh_example.target
            if knowedit_example is not None:
                query_record["knowedit_source"] = knowedit_example.source
                query_record["knowedit_subject"] = knowedit_example.subject
                query_record["knowedit_target"] = knowedit_example.target_new
                if knowedit_example.reference is not None:
                    query_record["knowedit_ground_truth"] = knowedit_example.reference
            output_records.append(query_record)
            emit(f"\n=== Query {prompt_index} ===")
            emit(prompt)
            encoded_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
            input_ids = encoded_prompt.input_ids.to(input_device)
            if input_ids.shape[1] == 0:
                raise ValueError("The tokenizer produced an empty prompt.")
            prompt_length = input_ids.shape[1]

            for run_index, (run_args, use_recirculation, label) in enumerate(runs):
                emit(f"\n=== {label} ===")
                if similarities_writer is not None:
                    similarities_writer.start_run(prompt, label, run_args.seed)
                target_token_id = None
                if run_args.perturb_mode == "towards-target":
                    target_ids = tokenizer.encode(
                        knowedit_example.target_new.strip(),
                        add_special_tokens=False,
                    )
                    if not target_ids:
                        raise ValueError("KnowEdit target has no tokens after encoding.")
                    target_token_id = target_ids[0]
                run_ids, similarities, stats, run_seconds = timed_generate(
                    use_recirculation=use_recirculation,
                    run_args=run_args,
                    target_token_id=target_token_id,
                    on_generated_token=lambda token: print(
                        tokenizer.decode(token[0], skip_special_tokens=True),
                        end="",
                        flush=True,
                    ),
                    on_debug_comparison=(
                        similarities_writer.append
                        if similarities_writer is not None
                        else None
                    ),
                )
                output = tokenizer.decode(
                    run_ids[0, prompt_length:], skip_special_tokens=True
                )
                emit(f"({run_seconds:.2f} s)")
                run_record = {
                    "label": label,
                    "seed": run_args.seed,
                    "seconds": round(run_seconds, 2),
                    "output": output,
                    "stats": stats,
                }
                if game24_puzzle is not None:
                    run_record["game24_solved"] = is_game24_solution(
                        game24_puzzle, output
                    )
                if countdown_puzzle is not None:
                    target, numbers = countdown_puzzle
                    value = score_countdown_output(target, numbers, output)
                    run_record["countdown_valid"] = value is not None
                    if value is not None:
                        run_record["countdown_value"] = str(value)
                        run_record["countdown_distance"] = float(abs(value - target))
                        run_record["countdown_exact"] = value == target
                if sudoku_puzzle is not None:
                    run_record["sudoku_solved"] = is_sudoku_solution(
                        sudoku_puzzle, output
                    )
                if bbeh_example is not None:
                    run_record["bbeh_correct"] = is_bbeh_correct(
                        bbeh_example, output
                    )
                if knowedit_example is not None:
                    emit(f"knowedit_ground_truth = {knowedit_example.reference if knowedit_example.reference is not None else 'n/a'}")
                    emit(f"knowedit_target_new = {knowedit_example.target_new}")
                    for score_name, reference in (
                        ("knowedit_ground_truth_acc", knowedit_example.reference),
                        ("knowedit_target_new_acc", knowedit_example.target_new),
                    ):
                        if reference is None:
                            continue
                        run_record[score_name] = score_knowedit_target(
                            input_ids, reference, run_args, use_recirculation, score_name
                        )
                        emit(f"{score_name} = {run_record[score_name]:.4f}")
                if args.do_eval and knowedit_example is None:
                    evaluation = evaluate_single_answer(
                        prompt,
                        label,
                        output,
                        eval_provider=args.eval_provider,
                        model=model,
                        tokenizer=tokenizer,
                        input_device=input_device,
                        api_key=os.environ.get("OPENAI_API_KEY"),
                        base_url=args.openai_base_url,
                        evaluation_model=args.evaluation_model,
                    )
                    run_record["score"] = evaluation["score"]
                    run_record["rationale"] = evaluation["rationale"]
                query_record["runs"].append(run_record)
                save_output()
        emit(f"\n=== Summary: {len(output_records)} queries ===")
        for run_index, (_run_args, _use_recirculation, label) in enumerate(runs):
            completed_runs = [
                record["runs"][run_index]
                for record in output_records
                if len(record["runs"]) > run_index
            ]
            aggregate_stats = aggregate_recirculation_stats(
                [run["stats"] for run in completed_runs]
            )
            emit(f"\n{label} ({sum(run['seconds'] for run in completed_runs):.2f} s)")
            for line in format_run_stats(aggregate_stats):
                emit(line)
            game24_runs = [
                run for run in completed_runs if "game24_solved" in run
            ]
            if game24_runs:
                emit(
                    "game24_solved = "
                    f"{sum(run['game24_solved'] for run in game24_runs)}/"
                    f"{len(game24_runs)}"
                )
            countdown_runs = [
                run for run in completed_runs if "countdown_valid" in run
            ]
            if countdown_runs:
                valid_runs = [
                    run for run in countdown_runs if run["countdown_valid"]
                ]
                exact_runs = [
                    run for run in valid_runs if run["countdown_exact"]
                ]
                average_distance = (
                    sum(run["countdown_distance"] for run in valid_runs)
                    / len(valid_runs)
                    if valid_runs
                    else None
                )
                emit(
                    "countdown_valid = "
                    f"{len(valid_runs)}/{len(countdown_runs)}, exact = "
                    f"{len(exact_runs)}/{len(countdown_runs)}, "
                    "average_distance = "
                    f"{average_distance:.2f}" if average_distance is not None
                    else "countdown_valid = 0/"
                    f"{len(countdown_runs)}, exact = 0/{len(countdown_runs)}, "
                    "average_distance = n/a"
                )
            sudoku_runs = [run for run in completed_runs if "sudoku_solved" in run]
            if sudoku_runs:
                emit(
                    "sudoku_solved = "
                    f"{sum(run['sudoku_solved'] for run in sudoku_runs)}/"
                    f"{len(sudoku_runs)}"
                )
            bbeh_runs = [run for run in completed_runs if "bbeh_correct" in run]
            if bbeh_runs:
                emit(
                    "bbeh_correct = "
                    f"{sum(run['bbeh_correct'] for run in bbeh_runs)}/"
                    f"{len(bbeh_runs)}"
                )
            for score_name in ("knowedit_ground_truth_acc", "knowedit_target_new_acc"):
                scored_runs = [run for run in completed_runs if score_name in run]
                if scored_runs:
                    emit(
                        f"{score_name} = "
                        f"{sum(run[score_name] for run in scored_runs) / len(scored_runs):.4f} "
                        f"({len(scored_runs)}/{len(completed_runs)} with reference)"
                    )
            if args.do_eval and knowedit_examples is None:
                emit(format_average_eval_rating([run["score"] for run in completed_runs]))
    finally:
        if output_file is not None:
            output_file.close()
        if similarities_writer is not None:
            similarities_writer.close()


if __name__ == "__main__":
    main()