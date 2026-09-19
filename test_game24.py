import tempfile
import unittest
from pathlib import Path

from game24 import (
    _evaluate_expression,
    format_countdown_prompt,
    format_prompt,
    is_solution,
    load_countdown_puzzles,
    load_puzzles,
    parse_puzzle,
    score_countdown_output,
    solve_countdown,
)


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

    def test_loads_countdown_csv_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "countdown.csv"
            path.write_text(
                "numbers,target\n"
                '"25,50,75,100,3,6",952\n',
                encoding="utf-8",
            )

            self.assertEqual(
                load_countdown_puzzles(path),
                ((952, (25, 50, 75, 100, 3, 6)),),
            )

    def test_rejects_a_github_html_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "24.csv"
            path.write_text("<!DOCTYPE html><html></html>", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "raw.githubusercontent.com"):
                load_puzzles(path)

    def test_requires_four_numbers(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly four"):
            parse_puzzle("1 2 3")

    def test_countdown_finds_an_exact_solution_with_a_subset(self) -> None:
        expression = solve_countdown(999, (100, 50, 25, 10, 2, 1))

        value, _ = _evaluate_expression(expression)
        self.assertEqual(value, 999)

    def test_countdown_returns_the_closest_result(self) -> None:
        expression = solve_countdown(100, (1, 1, 1, 1, 1, 1))

        value, _ = _evaluate_expression(expression)
        self.assertEqual(value, 9)

    def test_countdown_requires_six_allowed_numbers(self) -> None:
        with self.assertRaisesRegex(ValueError, "six numbers"):
            solve_countdown(100, (1, 2, 3, 4, 5))
        with self.assertRaisesRegex(ValueError, "six numbers"):
            solve_countdown(100, (1, 2, 3, 4, 5, 11))

    def test_formats_and_scores_countdown_answers(self) -> None:
        numbers = (100, 50, 25, 10, 2, 1)

        self.assertIn("Target: 999", format_countdown_prompt(999, numbers))
        self.assertEqual(
            score_countdown_output(999, numbers, "Answer: 100 * 10 - 1 = 999"),
            999,
        )
        self.assertIsNone(
            score_countdown_output(999, numbers, "Answer: 100 * 10 - 7 = 993")
        )