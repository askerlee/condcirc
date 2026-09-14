import json
import tempfile
import unittest
from pathlib import Path

from average_eval_ratings import format_method_ratings, load_method_ratings


class AverageEvalRatingsTest(unittest.TestCase):
    def write_results(self, records: list[dict]) -> Path:
        temporary_file = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", delete=False
        )
        with temporary_file:
            json.dump(records, temporary_file)
        self.addCleanup(Path(temporary_file.name).unlink, missing_ok=True)
        return Path(temporary_file.name)

    def test_groups_scores_by_method_and_reports_partial_coverage(self) -> None:
        path = self.write_results(
            [
                {
                    "runs": [
                        {"label": "Baseline", "score": 8.0},
                        {"label": "Ablation", "score": 9.0},
                    ]
                },
                {
                    "runs": [
                        {"label": "Baseline", "score": 7.0},
                        {"label": "Ablation"},
                    ]
                },
            ]
        )

        output = format_method_ratings(load_method_ratings(path))

        self.assertEqual(
            output,
            "Baseline\n"
            "average_eval_model_rating = 7.50 (2/2 rated)\n\n"
            "Ablation\n"
            "average_eval_model_rating = 9.00 (1/2 rated)",
        )

    def test_rejects_invalid_scores(self) -> None:
        path = self.write_results([{"runs": [{"label": "Baseline", "score": 11}]}])

        with self.assertRaisesRegex(ValueError, "invalid score"):
            load_method_ratings(path)


if __name__ == "__main__":
    unittest.main()
