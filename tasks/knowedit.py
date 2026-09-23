"""In-context knowledge-update evaluation over KnowEdit JSON records."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KnowEditExample:
    prompt: str
    target_new: str
    subject: str
    source: str


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
        examples.append(KnowEditExample(prompt, target, subject, source))
    return tuple(examples)


def format_prompt(example: KnowEditExample) -> str:
    if example.source == "wikibio":
        return (
            "Continue the following passage with the next factual sentence only.\n\n"
            f"{example.prompt.rstrip()}"
        )
    return (
        "For this question, use the following updated fact even if it differs "
        "from your prior knowledge. Answer with only the updated answer.\n\n"
        f"Question: {example.prompt.strip()}\n"
        f"Updated answer: {example.target_new.strip()}\n\n"
        f"Question: {example.prompt.strip()}\nAnswer:"
    )


def is_correct(example: KnowEditExample, output: str) -> bool:
    target = example.target_new.strip().casefold()
    answer = output.strip().casefold()
    if example.source == "wikibio":
        return answer.startswith(target)
    return re.match(rf"^{re.escape(target)}(?=$|[\s.,;:!?])", answer) is not None