import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tasks.sudoku import format_prompt, is_solution, load_puzzles, parse_puzzle


PUZZLE_4X4 = (
    (1, 0, 3, 4),
    (3, 4, 1, 0),
    (2, 1, 4, 3),
    (4, 3, 2, 1),
)
SOLUTION_4X4 = """Answer:
1 2 3 4
3 4 1 2
2 1 4 3
4 3 2 1"""


class SudokuTest(unittest.TestCase):
    def test_loads_sudoku_extreme_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            question = "12345678." + "." * 72
            content = f"source,question,answer,rating\nexample,{question},{'1' * 81},1\n"
            path.write_text(content, encoding="utf-8")

            puzzles = load_puzzles(path)
            self.assertEqual(len(puzzles), 1)
            self.assertEqual(puzzles[0][0], (1, 2, 3, 4, 5, 6, 7, 8, 0))
            self.assertEqual(puzzles[0][1], (0,) * 9)

            with patch("tasks.sudoku.urlopen", return_value=io.BytesIO(content.encode())):
                self.assertEqual(load_puzzles("https://example.org/test.csv"), puzzles)

    def test_loads_sudoku4llm_jsonl_with_text_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sudoku.jsonl"
            path.write_text(
                '{"puzzle": [[1, ".", 3, 4], [3, 4, 1, "."], [2, 1, 4, 3], [4, 3, 2, 1]], '
                '"config": {"grid_size": 4, "placeholder": "."}}\n',
                encoding="utf-8",
            )

            self.assertEqual(load_puzzles(path), (PUZZLE_4X4,))

    def test_formats_and_scores_a_valid_grid(self) -> None:
        self.assertIn("4x4 Sudoku", format_prompt(PUZZLE_4X4))
        self.assertTrue(is_solution(PUZZLE_4X4, SOLUTION_4X4))

    def test_rejects_invalid_or_clue_changing_grids(self) -> None:
        self.assertFalse(
            is_solution(
                PUZZLE_4X4,
                "1 2 3 4\n3 4 1 2\n2 1 4 3\n4 3 1 2",
            )
        )
        self.assertFalse(is_solution(((2, 0, 3, 4), *PUZZLE_4X4[1:]), SOLUTION_4X4))

    def test_rejects_unsupported_grid_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "4x4, 6x6, or 9x9"):
            parse_puzzle(((1, 2), (2, 1)))