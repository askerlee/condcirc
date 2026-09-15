from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


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

from recirculation import (  # noqa: E402
    AdjacentLayerSimilarityStats,
    RecirculationConfig,
    SimilarityStats,
    recirculate,
)


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
)

REJECTION_GATES = (
    "post-margin-min",
    "post-margin-ratio",
    "post-margin-max",
    "cosine",
    "rank",
)


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


def format_run_stats(stats: dict[str, Any]) -> tuple[str, str]:
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
            gate: sum(stats["rejected_by_gate"][gate] for stats in stats_records)
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
        default=[(-5, 5)],
        help=(
            "Source/destination pair to recirculate. Repeat for multiple pairs "
            "(default: -5 5)."
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
        default=(0.0, 0.0),
        metavar=("MIN", "MAX"),
        help=(
            "Magnitude-matched Gaussian noise range. Each pass maps its "
            "normalized preceding margin from MIN to MAX "
            "(0 <= MIN <= MAX <= 0.5; default: 0 0)."
        ),
    )
    parser.add_argument(
        "--noise-decay-per-pass",
        type=float,
        default=0.8,
        metavar="COEFFICIENT",
        help=(
            "Multiply the noise weight by this coefficient after each "
            "recirculation pass (0 to 1; default: 0.8)."
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
        default=0,
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
        "--cosine-top-k",
        type=int,
        default=5,
        metavar="K",
        help=(
            "Compute P1/final-pass cosine over the union of their top-K "
            "tokens (default: 5)."
        ),
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
    parser.add_argument("--max-new-tokens", type=int, default=1000)
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
        help="Transformers device map. 'balanced' shards layers across all GPUs.",
    )
    parser.add_argument(
        "--gpu-memory",
        default="46GiB",
        help="Maximum model memory per GPU when sharding (default: 46GiB).",
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
        args.ablations = False
        args.query_index_signature = query_index_signature(argv)
        return args

    ablations_index = argv.index("--ablation")
    baseline_argv = argv[:ablations_index]
    args = parser.parse_args(baseline_argv)
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
        "list_queries",
        "max_new_tokens",
        "model",
        "no_recirculate_after_tokens",
        "openai_base_url",
        "output",
        "query_indices",
        "seed",
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
    return args


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def validate_run_arguments(args: argparse.Namespace) -> None:
    noise_min, noise_max = args.noise_level_range
    if not 0.0 <= noise_min <= noise_max <= 0.5:
        raise ValueError(
            "--noise-level-range requires 0 <= MIN <= MAX <= 0.5."
        )
    if not 0.0 <= args.noise_decay_per_pass <= 1.0:
        raise ValueError("--noise-decay-per-pass must be between 0 and 1.")
    if noise_max > 0 and args.mode != "source":
        raise ValueError("--noise-level-range requires --mode source.")
    if noise_max > 0 and (
        args.post_margin_thres is None
        or args.post_margin_thres[0] == args.post_margin_thres[1]
    ):
        raise ValueError(
            "--noise-level-range requires --post-margin-thres with MIN < MAX."
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


def main() -> None:
    args = parse_args()
    print(f"GPUs used: {torch.cuda.device_count()}", flush=True)
    if args.list_queries:
        for index, query in enumerate(EXAMPLE_QUERIES, start=1):
            print(f"{index:2}. {query}")
        return 

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

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    use_device_map = args.device_map != "none"
    if use_device_map and not torch.cuda.is_available():
        raise RuntimeError("--device-map requires CUDA; use --device-map none on CPU/MPS.")
    dtype = torch.bfloat16 if use_device_map or device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    load_kwargs: dict[str, Any] = {"dtype": dtype}
    if use_device_map:
        load_kwargs.update(
            device_map=args.device_map,
            max_memory={
                gpu: args.gpu_memory for gpu in range(torch.cuda.device_count())
            },
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if not use_device_map:
        model.to(device)
    model.eval()
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
    if args.output is None:
        pairs = resolve_recirculation_pairs(
            args, len(blocks), global_attention_layers
        )
        model_slug = args.model.rsplit("/", 1)[-1]
        pair_slug = "_".join(f"{source}-{destination}" for source, destination in pairs)
        query_signature = "" if args.query_index_signature == "all" else f"-{args.query_index_signature}"
        args.output = Path(
            f"{model_slug}-{pair_slug}-passes{args.passes}"
            f"-tokens{args.max_new_tokens}"
            f"{gate_signature}{cutoff_signature}{query_signature}.json"
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
        teacher_top = teacher_prob.argmax(dim=-1)
        student_top = student_prob.argmax(dim=-1)
        return {
            "cosine_similarity": round(float(cosine.item()), 3),
            "js_similarity": round(float((1.0 - js_divergence / torch.log(
                torch.tensor(2.0, device=js_divergence.device)
            )).clamp(0.0, 1.0).item()), 3),
            "teacher_top_token": tokenizer.decode(teacher_top),
            "student_top_token": tokenizer.decode(student_top),
            "top1_agreement": int((teacher_top == student_top).item()),
        }

    def generate(
        use_recirculation: bool,
        run_args: argparse.Namespace,
        run_config: RecirculationConfig,
    ) -> tuple[Tensor, list[dict[str, Any]], dict[str, Any]]:
        torch.manual_seed(run_args.seed)
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
                        for threshold, reason in (
                            (run_args.post_margin_thres, "post-margin-min"),
                            (
                                run_args.post_margin_ratio_thres,
                                "post-margin-ratio",
                            ),
                            (run_args.cosine_reject, "cosine"),
                        )
                        if threshold is not None
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
                prompt_logits, student_cache = recirculate(
                    input_ids,
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
                    adaptive_recirculation=run_args.ada_recirculate,
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
                    capture_cached_token=capture_dynamic_cache_token,
                    restore_cached_token=restore_dynamic_cache_token,
                    capture_rewind_state=capture_dynamic_cache_rewind_state,
                    restore_rewind_state=restore_dynamic_cache_rewind_state,
                    finalize_token_cache=finalize_dynamic_cache_token,
                )
                next_logits = prompt_logits[:, -1, :]

            generated_ids = input_ids.clone()
            for _ in range(run_args.max_new_tokens):
                next_token = sample_token(next_logits, run_args.temperature)
                generated_ids = torch.cat((generated_ids, next_token), dim=1)
                record_generated_token_stats()
                if next_token.item() in eos_token_ids:
                    break
                if use_recirculation:
                    token_logits, student_cache = recirculate(
                        next_token,
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
                        adaptive_recirculation=run_args.ada_recirculate,
                        recirculation_allowed=allow_generated_recirculation(
                            generated_ids.shape[1] - input_ids.shape[1]
                        ),
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
                next_logits = token_logits[:, -1, :]

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
        injected_noise_levels: list[list[float]] = []
        student_logits, student_cache = recirculate(
            input_ids,
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
            adaptive_recirculation=run_args.ada_recirculate,
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
            injected_noise_levels=injected_noise_levels,
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
            comparison["cosine_top_k"] = run_args.cosine_top_k
            comparison["injected_noise_levels"] = [
                round(level, 6) for level in injected_noise_levels[-1]
            ]
            next_token = sample_token(student_next_logits, run_args.temperature)
            comparison.update(
                token_index=token_index,
                selected_token=tokenizer.decode(next_token[0]),
            )
            similarities.append(comparison)
            generated_ids = torch.cat((generated_ids, next_token), dim=1)

            if next_token.item() in eos_token_ids:
                break

            student_logits, student_cache = recirculate(
                next_token,
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
                adaptive_recirculation=run_args.ada_recirculate,
                recirculation_allowed=allow_generated_recirculation(token_index + 1),
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
                injected_noise_levels=injected_noise_levels,
                capture_cached_token=capture_dynamic_cache_token,
                restore_cached_token=restore_dynamic_cache_token,
                capture_rewind_state=capture_dynamic_cache_rewind_state,
                restore_rewind_state=restore_dynamic_cache_rewind_state,
                finalize_token_cache=finalize_dynamic_cache_token,
            )
            teacher_logits = first_pass_logits[-1]
            teacher_src_dst_sim = (
                sum(first_pass_similarities[-1]) / len(run_config.pairs)
            )
            teacher_next_logits = teacher_logits[:, -1, :]
            student_next_logits = student_logits[:, -1, :]

        # recirculated_flags also covers prompt positions and one trailing lookahead
        # call, neither of which produce a comparison entry, so count from
        # `similarities` instead to match the tokens actually reported.
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

    def timed_generate(
        use_recirculation: bool, run_args: argparse.Namespace
    ) -> tuple[Tensor, list[dict[str, Any]], dict[str, Any], float]:
        pairs = resolve_recirculation_pairs(
            run_args, len(blocks), global_attention_layers
        )
        run_config = RecirculationConfig(
            pairs=pairs,
            alpha=run_args.alpha,
            beta=run_args.beta,
            noise_level_range=tuple(run_args.noise_level_range),
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
        synchronize_devices()
        start = time.perf_counter()
        generated_ids, similarities, stats = generate(
            use_recirculation=use_recirculation,
            run_args=run_args,
            run_config=run_config,
        )
        synchronize_devices()
        return generated_ids, similarities, stats, time.perf_counter() - start

    if args.ablations is False or args.ablations is None:
        baseline_options = (
            "model",
            "passes",
            "noise_level_range",
            "noise_decay_per_pass",
            "cond_recirculate",
            "act_sim_thres",
            "pre_margin_thres",
            "post_margin_thres",
            "post_margin_ratio_thres",
            "ada_recirculate",
            "cosine_reject",
            "cosine_top_k",
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
        runs = [(args, True, f"Baseline: {baseline_arguments}")]
        for overrides in args.ablations:
            ablation_args = argparse.Namespace(**vars(args))
            for option, value in overrides:
                setattr(ablation_args, option, value)
            validate_run_arguments(ablation_args)
            arguments = format_run_arguments(ablation_args, label_options)
            runs.append((ablation_args, True, f"Ablation: {arguments}"))

    prompts = (
        ((1, args.prompt),)
        if args.prompt is not None
        else tuple(
            (index, EXAMPLE_QUERIES[index - 1]) for index in args.query_indices
        )
    )
    output_file = args.output.open("w+", encoding="utf-8") if args.output else None
    similarities_file = (
        args.similarities_output.open("w", encoding="utf-8")
        if args.debug and args.similarities_output
        else None
    )
    output_records: list[dict[str, Any]] = []
    similarity_records: list[dict[str, Any]] = []

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
        for prompt_index, prompt in prompts:
            query_record: dict[str, Any] = {
                "index": prompt_index,
                "prompt": prompt,
                "runs": [],
            }
            output_records.append(query_record)
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
                run_ids, similarities, stats, run_seconds = timed_generate(
                    use_recirculation=use_recirculation, run_args=run_args
                )
                if run_index == 0:
                    emit(f"\n=== Query {prompt_index} ===")
                    emit(prompt)
                emit(f"\n=== {label} ({run_seconds:.2f} s) ===")
                output = tokenizer.decode(
                    run_ids[0, prompt_length:], skip_special_tokens=True
                )
                emit(output)
                run_record = {
                    "label": label,
                    "seconds": round(run_seconds, 2),
                    "output": output,
                    "stats": stats,
                }
                if args.do_eval:
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
                if similarities_file is not None:
                    similarity_records.append({
                        "prompt": prompt,
                        "run": label,
                        "similarities": similarities,
                    })
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
            if args.do_eval:
                emit(format_average_eval_rating([run["score"] for run in completed_runs]))
    finally:
        if output_file is not None:
            output_file.close()
        if similarities_file is not None:
            json.dump(
                similarity_records,
                similarities_file,
                ensure_ascii=False,
                indent=2,
            )
            similarities_file.write("\n")
            similarities_file.close()


if __name__ == "__main__":
    main()