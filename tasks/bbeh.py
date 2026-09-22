"""BIG-Bench Extra Hard dataset loading, prompting, and scoring helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BBEHExample:
    task: str
    input: str
    target: str


def _parse_examples(records: Any, task: str, path: Path) -> list[BBEHExample]:
    if not isinstance(records, list):
        raise ValueError(f"{path} has a non-list examples field for {task!r}.")
    examples = []
    for record in records:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("input"), str)
            or not isinstance(record.get("target"), str)
        ):
            raise ValueError(f"{path} contains an invalid BBEH example for {task!r}.")
        examples.append(BBEHExample(task, record["input"], record["target"]))
    return examples


def load_examples(path: Path) -> tuple[BBEHExample, ...]:
    """Load a BBEH benchmark directory, task.json, or combined data.json file."""
    if path.is_dir():
        task_files = sorted(path.glob("bbeh_*/task.json"))
        if not task_files:
            raise ValueError(f"{path} contains no BBEH task.json files.")
        return tuple(
            example
            for task_file in task_files
            for example in load_examples(task_file)
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read BBEH JSON from {path}.") from error

    examples: list[BBEHExample] = []
    if isinstance(payload, dict) and "examples" in payload:
        examples.extend(_parse_examples(payload["examples"], path.parent.name, path))
    elif isinstance(payload, dict):
        for task, task_payload in payload.items():
            if task == "canary":
                continue
            records = (
                task_payload.get("examples")
                if isinstance(task_payload, dict)
                else task_payload
            )
            examples.extend(_parse_examples(records, task, path))
    else:
        raise ValueError(f"{path} must contain a BBEH JSON object.")

    if not examples:
        raise ValueError(f"{path} contains no BBEH examples.")
    return tuple(examples)


def format_prompt(example: BBEHExample) -> str:
    return (
        f"{example.input.rstrip()}\n\n"
        "End with exactly one line in this form: The final answer is: <answer>"
    )


def _strip_latex(response: str) -> str:
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    for command in ("boxed{", "text{", "texttt{"):
        if command in response and response.endswith("}"):
            response = response[:-1].split(command)[-1]
    return response


def extract_answer(output: str) -> str:
    answer = output.strip()
    for prefix in (
        "The answer is:",
        "The final answer is ",
        "The final answer is: ",
        "The answer is ",
    ):
        if prefix in answer:
            answer = answer.split(prefix)[-1].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    return _strip_latex(answer)


def _fuzzy_match(prediction: str, reference: str) -> bool:
    if prediction == reference:
        return True
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True
    return prediction.endswith("?") and prediction[:-1] == reference


def is_correct(example: BBEHExample, output: str) -> bool:
    prediction = extract_answer(output).lower().replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    if prediction.endswith("."):
        prediction = prediction[:-1]
    reference = example.target.strip().lower().replace(", ", ",")
    return _fuzzy_match(prediction, reference)
