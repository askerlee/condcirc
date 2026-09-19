import tempfile
import unittest
from pathlib import Path

from game24 import format_prompt, is_solution, load_puzzles, parse_puzzle


class Game24Test(unittest.TestCase):
    def test_accepts_a_valid_final_answer(self) -> None:
        self.assertTrue(is_solution("1 3 4 6", "Reasoning\nAnswer: 6 / (1 - 3 / 4) = 24"))

    def test_rejects_wrong_or_reused_numbers(self) -> None:
        self.assertFalse(is_solution("2 3 4 6", "Answer: 6 * 4 = 24"))
        self.assertFalse(is_solution("2 3 4 6", "Answer: 6 * 6 - 4 - 2 = 24"))

    def test_rejects_unsupported_operations(self) -> None:
        self.assertFalse(is_solution("2 3 4 6", "Answer: 6 ** 2 - 4 - 2 - 3 = 24"))

    def test_formats_a_constrained_prompt(self) -> None:
        self.assertIn("2 3 4 6", format_prompt("2 3 4 6"))
        self.assertIn("Answer:", format_prompt("2 3 4 6"))

    def test_loads_upstream_csv_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "24.csv"
            path.write_text("Rank,Puzzles\n1,1 1 4 6\n", encoding="utf-8")
            self.assertEqual(load_puzzles(path), ((1, 1, 4, 6),))

    def test_rejects_a_github_html_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "24.csv"
            path.write_text("<!DOCTYPE html><html></html>", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "raw.githubusercontent.com"):
                load_puzzles(path)

    def test_requires_four_numbers(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly four"):
            parse_puzzle("1 2 3")