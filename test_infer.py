import unittest

from infer import (
    aggregate_recirculation_stats,
    format_average_eval_rating,
    format_run_stats,
    parse_args,
    summarize_recirculation_stats,
    validate_run_arguments,
)


class RecirculationStatsTest(unittest.TestCase):
    def test_parses_noise_weight(self) -> None:
        args = parse_args(["--noise", "0.25", "prompt"])

        self.assertEqual(args.noise, 0.25)

    def test_rejects_noise_weight_outside_range(self) -> None:
        args = parse_args(["--noise", "0.51", "prompt"])

        with self.assertRaisesRegex(ValueError, "between 0 and 0.5"):
            validate_run_arguments(args)

    def test_parses_no_recirculation_token_limit(self) -> None:
        args = parse_args(["--no-recirculate-after-N-tokens", "12", "prompt"])

        self.assertEqual(args.no_recirculate_after_tokens, 12)

    def test_parses_post_margin_ratio_threshold(self) -> None:
        args = parse_args(["--post-margin-ratio-thres", "1.5", "prompt"])

        self.assertEqual(args.post_margin_ratio_thres, 1.5)

    def test_parses_post_margin_min_and_max(self) -> None:
        args = parse_args(["--post-margin-thres", "0.1", "0.2", "prompt"])

        self.assertEqual(args.post_margin_thres, [0.1, 0.2])

    def test_formats_average_eval_model_rating(self) -> None:
        self.assertEqual(
            format_average_eval_rating([8.0, 7.5, 9.0]),
            "average_eval_model_rating = 8.17",
        )

    def test_summarizes_result_json_stats(self) -> None:
        rejection_reasons = [
            ("post-margin-min", "post-margin-ratio", "post-margin-max", "cosine", "rank"),
            ("post-margin-min", "post-margin-ratio", "post-margin-max", "cosine", "rank"),
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
                    "post-margin-ratio": 2,
                    "post-margin-max": 6,
                    "cosine": 6,
                    "rank": 2,
                },
                "adaptive_recirculated_tokens": {"count": 3, "total": 300},
                "adaptive_rejected": 2,
                "average_adaptive_recirculations": 1.67,
            },
        )

    def test_formats_aggregate_summary_after_all_queries(self) -> None:
        first = {
            "recirculated_tokens": {"count": 7, "total": 300},
            "same_top1": 2,
            "rejected": 9,
            "rejected_by_gate": {
                "post-margin-min": 2,
                "post-margin-ratio": 2,
                "post-margin-max": 6,
                "cosine": 6,
                "rank": 2,
            },
            "adaptive_recirculated_tokens": {"count": 3, "total": 300},
            "adaptive_rejected": 2,
            "average_adaptive_recirculations": 1.67,
        }
        second = {
            "recirculated_tokens": {"count": 5, "total": 200},
            "same_top1": 1,
            "rejected": 4,
            "rejected_by_gate": {
                "post-margin-min": 1,
                "post-margin-ratio": 1,
                "post-margin-max": 2,
                "cosine": 3,
                "rank": 0,
            },
            "adaptive_recirculated_tokens": {"count": 2, "total": 200},
            "adaptive_rejected": 1,
            "average_adaptive_recirculations": 2.5,
        }

        lines = format_run_stats(aggregate_recirculation_stats([first, second]))

        self.assertEqual(
            lines,
            (
                "recirculated_tokens = 12/500, same_top1 = 3, rejected = 13, "
                "by gate: post-margin-min=3, post-margin-ratio=3, "
                "post-margin-max=8, cosine=9, rank=2",
                "adaptive_recirculated_tokens = 5/500, rejected = 3, "
                "average_adaptive_recirculations = 2.00",
            ),
        )


if __name__ == "__main__":
    unittest.main()