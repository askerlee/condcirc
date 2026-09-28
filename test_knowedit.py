import json
import tempfile
import unittest
from pathlib import Path

from tasks.knowedit import format_prompt, load_examples, select_variant


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

    def test_loads_portability_questions_and_answers(self) -> None:
        records = [{
            "subject": "Epaspidoceras",
            "prompt": "Which family does Epaspidoceras belong to?",
            "target_new": "Noctuidae",
            "portability": {"Reasoning": [{
                "prompt": "What is the common name for the family Epaspidoceras belongs to?",
                "ground_truth": "Owlet moths",
            }]},
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ZsRE-test-all.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            example, = load_examples(path)

        self.assertEqual(example.portability, ((
            "Reasoning", "What is the common name for the family Epaspidoceras belongs to?", "Owlet moths",
        ),))
        self.assertNotIn("Owlet moths", format_prompt(example))
        portability_prompt = format_prompt(example, example.portability[0][1])
        self.assertIn("What is the common name for the family Epaspidoceras belongs to?", portability_prompt)
        self.assertNotIn("Owlet moths", portability_prompt)

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

    def test_selects_indexed_variants_and_their_targets(self) -> None:
        records = [{
            "prompt": "Main question", "target_new": "New answer",
            "rephrase_prompt": "Rephrased question",
            "portability": {"Reasoning": [
                {"prompt": "First portable", "ground_truth": [["First answer", "Alias"]]},
                {"prompt": "Second portable", "ground_truth": "Second answer"},
            ]},
            "locality": {"Relation_Specificity": [
                {"prompt": "First local", "ground_truth": ["Local answer"]},
                {"prompt": "Second local", "ground_truth": [["Other answer", "Alias"]]},
            ]},
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "variants.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            example, = load_examples(path)

        self.assertEqual(select_variant(example, "rephrased_prompt", 1), ("Rephrased question", "New answer"))
        with self.assertRaisesRegex(ValueError, "index 2 is unavailable"):
            select_variant(example, "rephrased_prompt", 2)
        self.assertEqual(select_variant(example, "portability", 1), ("First portable", "First answer"))
        self.assertEqual(select_variant(example, "portability", 2), ("Second portable", "Second answer"))
        self.assertEqual(select_variant(example, "locality", 2), ("Second local", "Other answer"))
        with self.assertRaisesRegex(ValueError, "index 3 is unavailable"):
            select_variant(example, "locality", 3)
        with self.assertRaisesRegex(ValueError, "index 0 is unavailable"):
            select_variant(example, "portability", 0)

        records[0]["rephrase_prompt"] = ["Rephrased question"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "variants.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid KnowEdit rephrased_prompt"):
                load_examples(path)
            records[0].pop("rephrase_prompt")
            records[0]["rephrase"] = "Rephrased question"
            records[0]["rephrased_prompt"] = "Other rephrased question"
            path.write_text(json.dumps(records), encoding="utf-8")
            alias_example, = load_examples(path)
        self.assertIsNone(alias_example.rephrased_prompt)
        with self.assertRaisesRegex(ValueError, "index 1 is unavailable"):
            select_variant(alias_example, "rephrased_prompt", 1)


if __name__ == "__main__":
    unittest.main()