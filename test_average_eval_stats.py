import json
import tempfile
import unittest
from pathlib import Path

from average_eval_stats import format_method_ratings, load_method_ratings


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

    def test_aggregates_and_formats_recirculation_stats(self) -> None:
        path = self.write_results(
            [
                {
                    "runs": [
                        {
                            "label": "Baseline",
                            "score": 8.0,
                            "stats": {
                                "recirculated_tokens": {"count": 2, "total": 10},
                                "same_top1": 1,
                                "rejected": 3,
                                "rejected_by_gate": {"cosine": 2, "rank": 1},
                                "adaptive_recirculated_tokens": {
                                    "count": 2,
                                    "total": 10,
                                },
                                "adaptive_rejected": 1,
                                "average_adaptive_recirculations": 1.5,
                            },
                        }
                    ]
                },
                {
                    "runs": [
                        {
                            "label": "Baseline",
                            "score": 6.0,
                            "stats": {
                                "recirculated_tokens": {"count": 4, "total": 20},
                                "same_top1": 2,
                                "rejected": 5,
                                "rejected_by_gate": {"cosine": 3, "rank": 2},
                                "adaptive_recirculated_tokens": {
                                    "count": 1,
                                    "total": 20,
                                },
                                "adaptive_rejected": 2,
                                "average_adaptive_recirculations": 3.0,
                            },
                        }
                    ]
                },
            ]
        )

        output = format_method_ratings(load_method_ratings(path))

        self.assertEqual(
            output,
            "Baseline\n"
            "average_eval_model_rating = 7.00 (2/2 rated)\n"
            "recirculated_tokens = 6/30, same_top1 = 3, rejected = 8, "
            "by gate: cosine=5, rank=3\n"
            "adaptive_recirculated_tokens = 3/30, rejected = 3, "
            "average_adaptive_recirculations = 2.00",
        )


if __name__ == "__main__":
    unittest.main()
