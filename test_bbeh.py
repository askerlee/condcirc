import json
import tempfile
import unittest
from pathlib import Path

from tasks.bbeh import BBEHExample, format_prompt, is_correct, load_examples


class BBEHTest(unittest.TestCase):
    def test_loads_canonical_task_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bbeh_boolean_expressions" / "task.json"
            path.parent.mkdir()
            path.write_text(
                json.dumps(
                    {
                        "examples": [{"input": "Which option?", "target": "(E)"}],
                        "canary": "benchmark canary",
                    }
                ),
                encoding="utf-8",
            )

            examples = load_examples(path)

        self.assertEqual(
            examples,
            (BBEHExample("bbeh_boolean_expressions", "Which option?", "(E)"),),
        )

    def test_loads_combined_task_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text(
                json.dumps(
                    {
                        "task_a": {"examples": [{"input": "A?", "target": "yes"}]},
                        "task_b": [{"input": "B?", "target": "2"}],
                    }
                ),
                encoding="utf-8",
            )

            examples = load_examples(path)

        self.assertEqual([example.task for example in examples], ["task_a", "task_b"])

    def test_loads_complete_benchmark_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task in ("bbeh_task_b", "bbeh_task_a"):
                task_path = root / task / "task.json"
                task_path.parent.mkdir()
                task_path.write_text(
                    json.dumps({"examples": [{"input": task, "target": "ok"}]}),
                    encoding="utf-8",
                )

            examples = load_examples(root)

        self.assertEqual(
            [example.task for example in examples],
            ["bbeh_task_a", "bbeh_task_b"],
        )

    def test_prompt_requests_the_official_answer_prefix(self) -> None:
        prompt = format_prompt(BBEHExample("task", "Question?\n", "answer"))

        self.assertEqual(
            prompt,
            "Question?\n\nEnd with exactly one line in this form: "
            "The final answer is: <answer>",
        )

    def test_matches_official_normalization_cases(self) -> None:
        cases = (
            ("Ok The final answer is: \\boxed{4}.", "4", True),
            ("Alright! The final answer is: 2, 3, 4", "2,3,4", True),
            ("Ok The answer is: (A)", "a", True),
            ("Ok The answer is: **25**\nHere's why.", "25.0", True),
            ("The final answer is: [proved]", "proved", True),
            ("The final answer is: no?", "no", True),
            ("The final answer is: B", "a", False),
        )
        for output, target, expected in cases:
            with self.subTest(output=output, target=target):
                self.assertEqual(
                    is_correct(BBEHExample("task", "question", target), output),
                    expected,
                )

    def test_rejects_malformed_examples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            path.write_text('{"examples": [{"input": "missing target"}]}')

            with self.assertRaisesRegex(ValueError, "invalid BBEH example"):
                load_examples(path)


if __name__ == "__main__":
    unittest.main()