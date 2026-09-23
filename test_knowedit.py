import json
import tempfile
import unittest
from pathlib import Path

from tasks.knowedit import format_prompt, is_correct, load_examples


class KnowEditTest(unittest.TestCase):
    def test_loads_fact_records_and_scores_updated_answer(self) -> None:
        records = [
            {"subject": "Epaspidoceras", "prompt": "Which family does Epaspidoceras belong to?", "target_new": "Noctuidae", "ground_truth": ["Aspidoceratidae"]},
            {"subject": "Frederic Piesch", "prompt": "The position held by Frederic Piesch is", "target_new": "Archbishop of Leon", "ground_truth": "mayor of Vienna"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ZsRE-test-all.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            examples = load_examples(path)

        self.assertEqual(len(examples), 2)
        self.assertIn("Updated answer: Noctuidae", format_prompt(examples[0]))
        self.assertTrue(is_correct(examples[0], "Noctuidae."))
        self.assertFalse(is_correct(examples[0], "Aspidoceratidae"))
        self.assertFalse(is_correct(examples[0], "Noctuidaeidae"))

    def test_wikibio_continuation_and_invalid_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wikibio-test-all.json"
            path.write_text(json.dumps([{"text": "A biographical passage.", "labels": "He studied medicine.", "concept": "a scholar"}]), encoding="utf-8")
            example, = load_examples(path)
            self.assertIn("A biographical passage.", format_prompt(example))
            self.assertTrue(is_correct(example, "He studied medicine. He later taught."))
            self.assertFalse(is_correct(example, "He studied law."))
            path.write_text(json.dumps([{"prompt": "Missing target"}]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing prompt or target"):
                load_examples(path)


if __name__ == "__main__":
    unittest.main()