"""Target-blind answer evaluation over KnowEdit JSON records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KnowEditExample:
    prompt: str
    target_new: str
    subject: str
    source: str
    reference: str | None = None
    portability: tuple[tuple[str, str, str], ...] = ()


def load_examples(path: Path) -> tuple[KnowEditExample, ...]:
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read KnowEdit JSON from {path}.") from error
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} must contain a nonempty list of KnowEdit records.")

    examples = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid KnowEdit record {index} in {path}.")
        source = "wikibio" if "text" in record else "fact"
        prompt = record.get("text") if source == "wikibio" else record.get("prompt")
        target = record.get("labels") if source == "wikibio" else record.get("target_new")
        subject = record.get("concept", "") if source == "wikibio" else record.get("subject", "")
        if not all(
            isinstance(value, str) and value.strip() for value in (prompt, target)
        ):
            raise ValueError(
                f"Invalid KnowEdit record {index} in {path}: missing prompt or target."
            )
        if not isinstance(subject, str):
            raise ValueError(f"Invalid KnowEdit subject in record {index} of {path}.")
        original = record.get("ground_truth")
        if isinstance(original, list):
            original = next(
                (answer for answer in original if isinstance(answer, str) and answer.strip()),
                None,
            )
        if original is not None and not isinstance(original, str):
            raise ValueError(f"Invalid KnowEdit ground_truth in record {index} of {path}.")
        portability = []
        for category, questions in record.get("portability", {}).items():
            for question in questions:
                portability_prompt = question["prompt"]
                portability_answer = question["ground_truth"]
                if not all(isinstance(value, str) and value.strip() for value in (portability_prompt, portability_answer)):
                    raise ValueError(f"Invalid KnowEdit portability in record {index} of {path}.")
                portability.append((category, portability_prompt, portability_answer))
        examples.append(KnowEditExample(prompt, target, subject, source, original, tuple(portability)))
    return tuple(examples)


def format_prompt(example: KnowEditExample, prompt: str | None = None) -> str:
    question = example.prompt if prompt is None else prompt
    if example.source == "wikibio":
        return (
            "Continue the following passage with the next factual sentence only.\n\n"
            f"{question.rstrip()}"
        )
    return (
        "Answer the question or complete the statement with only the requested fact.\n\n"
        f"{question.strip()}"
    )

