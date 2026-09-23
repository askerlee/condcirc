import json
import tempfile
import unittest
from pathlib import Path

from tasks.knowedit import format_prompt, load_examples


class KnowEditTest(unittest.TestCase):
    def test_loads_fact_records_without_leaking_target(self) -> None:
        records = [
            {"subject": "Epaspidoceras", "prompt": "Which family does Epaspidoceras belong to?", "target_new": "Noctuidae", "ground_truth": ["Aspidoceratidae"]},
            {"subject": "Frederic Piesch", "prompt": "The position held by Frederic Piesch is", "target_new": "Archbishop of Leon", "ground_truth": "mayor of Vienna"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ZsRE-test-all.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            examples = load_examples(path)

        self.assertEqual(len(examples), 2)
        self.assertEqual(
            format_prompt(examples[0]).count("Which family does Epaspidoceras belong to?"),
            1,
        )
        self.assertNotIn("Noctuidae", format_prompt(examples[0]))
        self.assertEqual(examples[0].target_new, "Noctuidae")
        self.assertEqual(examples[0].reference, "Aspidoceratidae")
        self.assertEqual(examples[1].reference, "mayor of Vienna")

    def test_zic3_question_does_not_leak_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ZsRE-test-all.json"
            path.write_text(
                json.dumps([{
                    "prompt": "What species is ZIC3 specific to?",
                    "target_new": "male",
                    "subject": "ZIC3",
                }]),
                encoding="utf-8",
            )
            example, = load_examples(path)

        prompt = format_prompt(example)
        self.assertIn("What species is ZIC3 specific to?", prompt)
        self.assertNotIn("male", prompt.casefold())
        self.assertNotIn("Updated answer", prompt)
        self.assertEqual(example.target_new, "male")
        self.assertIsNone(example.reference)

    def test_wikibio_continuation_and_invalid_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wikibio-test-all.json"
            path.write_text(json.dumps([{"text": "A biographical passage.", "labels": "He studied medicine.", "concept": "a scholar"}]), encoding="utf-8")
            example, = load_examples(path)
            self.assertIn("A biographical passage.", format_prompt(example))
            self.assertEqual(example.target_new, "He studied medicine.")
            self.assertIsNone(example.reference)
            path.write_text(json.dumps([{"prompt": "Missing target"}]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing prompt or target"):
                load_examples(path)


if __name__ == "__main__":
    unittest.main()