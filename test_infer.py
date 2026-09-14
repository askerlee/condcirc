import unittest

from infer import summarize_recirculation_stats


class RecirculationStatsTest(unittest.TestCase):
    def test_summarizes_result_json_stats(self) -> None:
        rejection_reasons = [
            ("post-margin-min", "post-margin-max", "cosine", "rank"),
            ("post-margin-min", "post-margin-max", "cosine", "rank"),
            *(("post-margin-max", "cosine"),) * 4,
            *((),) * 294,
        ]

        stats = summarize_recirculation_stats(
            [True] * 7 + [False] * 293,
            [True] * 9 + [False] * 291,
            [True] * 3 + [False] * 297,
            [True] * 2 + [False] * 298,
            [1, 2, 2] + [0] * 297,
            [True] * 2 + [False] * 298,
            rejection_reasons,
        )

        self.assertEqual(
            stats,
            {
                "recirculated_tokens": {"count": 7, "total": 300},
                "same_top1": 2,
                "rejected": 9,
                "rejected_by_gate": {
                    "post-margin-min": 2,
                    "post-margin-max": 6,
                    "cosine": 6,
                    "rank": 2,
                },
                "adaptive_recirculated_tokens": {"count": 3, "total": 300},
                "adaptive_rejected": 2,
                "average_adaptive_recirculations": 1.67,
            },
        )


if __name__ == "__main__":
    unittest.main()