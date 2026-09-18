import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from torch import nn

from infer import (
    aggregate_recirculation_stats,
    apply_repetition_penalty,
    configure_model_generation,
    enable_fp32_output_projection,
    format_average_eval_rating,
    format_run_stats,
    generated_pre_margin_threshold,
    output_recirculation_pairs,
    parse_args,
    summarize_recirculation_stats,
    set_random_seed,
    validate_run_arguments,
)


class RecirculationStatsTest(unittest.TestCase):
    def test_output_projection_returns_fp32_logits(self) -> None:
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output = nn.Linear(3, 2, bias=True).to(torch.bfloat16)

            def get_output_embeddings(self) -> nn.Module:
                return self.output

        model = Model()
        hidden_states = torch.tensor(
            [[[1.0, 2.0, 3.0]]], dtype=torch.bfloat16
        )
        expected = torch.nn.functional.linear(
            hidden_states.float(),
            model.output.weight.float(),
            model.output.bias.float(),
        )

        enable_fp32_output_projection(model)
        logits = model.output(hidden_states)

        self.assertEqual(logits.dtype, torch.float32)
        torch.testing.assert_close(logits, expected)

    def test_sets_python_and_torch_random_seeds(self) -> None:
        with (
            patch("infer.random.seed") as python_seed,
            patch("infer.torch.manual_seed") as torch_seed,
            patch("infer.torch.use_deterministic_algorithms") as deterministic,
            patch("infer.torch.cuda.is_available", return_value=False),
        ):
            set_random_seed(42)

        python_seed.assert_called_once_with(42)
        torch_seed.assert_called_once_with(42)
        deterministic.assert_called_once_with(True)

    def test_parses_noise_level_range(self) -> None:
        args = parse_args(
            [
                "--noise-level-range",
                "0.1",
                "0.25",
                "--noise-decay-per-pass",
                "0.75",
                "prompt",
            ]
        )

        self.assertEqual(args.noise_level_range, [0.1, 0.25])
        self.assertEqual(args.noise_decay_per_pass, 0.75)

    def test_parses_initial_pre_margin_relaxation(self) -> None:
        args = parse_args(
            [
                "--pre-margin-relax-tokens",
                "3",
                "--pre-margin-relax-factor",
                "1.5",
                "prompt",
            ]
        )

        self.assertEqual(args.pre_margin_relax_tokens, 3)
        self.assertEqual(args.pre_margin_relax_factor, 1.5)

    def test_relaxes_only_the_first_generated_tokens(self) -> None:
        self.assertEqual(generated_pre_margin_threshold(0.2, 0, 1.5, 1), 0.2)
        self.assertAlmostEqual(generated_pre_margin_threshold(0.2, 3, 1.5, 1), 0.3)
        self.assertAlmostEqual(generated_pre_margin_threshold(0.2, 3, 1.5, 3), 0.3)
        self.assertEqual(generated_pre_margin_threshold(0.2, 3, 1.5, 4), 0.2)

    def test_uses_gemma_specific_default_pair(self) -> None:
        args = parse_args(["--model", "google/gemma-4-26b-a4b-it", "prompt"])

        self.assertEqual(args.pairs, [(25, 19)])

    def test_explicit_pair_overrides_model_default(self) -> None:
        args = parse_args(
            [
                "--model",
                "google/gemma-4-26b-a4b-it",
                "--pair",
                "7",
                "3",
                "prompt",
            ]
        )

        self.assertEqual(args.pairs, [[7, 3]])

    def test_ablation_pair_is_used_for_conditional_output_name(self) -> None:
        args = parse_args(
            [
                "--model",
                "openai/gpt-oss-20b",
                "--ablation",
                "--cond-recirculate",
                "--pair",
                "-5",
                "12",
            ]
        )

        self.assertEqual(args.pairs, [(-5, 5)])
        self.assertEqual(output_recirculation_pairs(args), [[-5, 12]])

    def test_sets_gpt_oss_repetition_penalty(self) -> None:
        model = SimpleNamespace(generation_config=SimpleNamespace(repetition_penalty=1.0))

        configure_model_generation(model, "openai/gpt-oss-20b")

        self.assertEqual(model.generation_config.repetition_penalty, 1.1)

    def test_cli_repetition_penalty_overrides_gpt_oss_default(self) -> None:
        model = SimpleNamespace(generation_config=SimpleNamespace(repetition_penalty=1.0))

        configure_model_generation(model, "openai/gpt-oss-20b", 1.25)

        self.assertEqual(model.generation_config.repetition_penalty, 1.25)

    def test_leaves_other_model_repetition_penalty_unchanged(self) -> None:
        model = SimpleNamespace(generation_config=SimpleNamespace(repetition_penalty=1.0))

        configure_model_generation(model, "Qwen/Qwen3.6-35B-A3B-FP8")

        self.assertEqual(model.generation_config.repetition_penalty, 1.0)

    def test_defaults_unset_repetition_penalty_to_one(self) -> None:
        model = SimpleNamespace(generation_config=SimpleNamespace(repetition_penalty=None))

        configure_model_generation(model, "google/gemma-3-4b-it")

        self.assertEqual(model.generation_config.repetition_penalty, 1.0)

    def test_applies_repetition_penalty_to_previous_tokens(self) -> None:
        logits = torch.tensor([[-2.0, 2.0, 4.0]])
        previous_token_ids = torch.tensor([[0, 1]])

        penalized_logits = apply_repetition_penalty(
            logits, previous_token_ids, 1.1
        )

        torch.testing.assert_close(
            penalized_logits, torch.tensor([[-2.2, 2.0 / 1.1, 4.0]])
        )

    def test_skips_repetition_penalty_after_accepted_recirculation(self) -> None:
        logits = torch.tensor([[-2.0, 2.0, 4.0]])
        previous_token_ids = torch.tensor([[0, 1]])

        sampling_logits = apply_repetition_penalty(
            logits, previous_token_ids, 1.1, recirculated=True
        )

        torch.testing.assert_close(sampling_logits, logits)

    def test_parses_perturb_pre_margin_threshold(self) -> None:
        args = parse_args(["--perturb-pre-margin-thres", "0.12", "prompt"])

        self.assertEqual(args.perturb_pre_margin_thres, 0.12)

    def test_rejects_negative_perturb_pre_margin_threshold(self) -> None:
        args = parse_args(["--perturb-pre-margin-thres", "-0.01", "prompt"])

        with self.assertRaisesRegex(ValueError, "perturb-pre-margin-thres"):
            validate_run_arguments(args)

    def test_parses_repetition_penalty(self) -> None:
        args = parse_args(["--repetition-penalty", "1.25", "prompt"])

        self.assertEqual(args.repetition_penalty, 1.25)

    def test_rejects_nonpositive_repetition_penalty(self) -> None:
        args = parse_args(["--repetition-penalty", "0", "prompt"])

        with self.assertRaisesRegex(ValueError, "repetition-penalty"):
            validate_run_arguments(args)

    def test_rejects_noise_level_range_outside_range(self) -> None:
        args = parse_args(["--noise-level-range", "0.1", "0.51", "prompt"])

        with self.assertRaisesRegex(ValueError, "0 <= MIN <= MAX <= 0.5"):
            validate_run_arguments(args)

    def test_rejects_descending_noise_level_range(self) -> None:
        args = parse_args(["--noise-level-range", "0.3", "0.2", "prompt"])

        with self.assertRaisesRegex(ValueError, "0 <= MIN <= MAX <= 0.5"):
            validate_run_arguments(args)

    def test_rejects_noise_decay_outside_range(self) -> None:
        args = parse_args(["--noise-decay-per-pass", "1.1", "prompt"])

        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
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
                    "margin-narrowed": 0,
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
                "margin-narrowed": 0,
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
                "margin-narrowed": 0,
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
                "by gate: margin-narrowed=0, post-margin-min=3, post-margin-ratio=3, "
                "post-margin-max=8, cosine=9, rank=2",
                "adaptive_recirculated_tokens = 5/500, rejected = 3, "
                "average_adaptive_recirculations = 2.00",
            ),
        )


if __name__ == "__main__":
    unittest.main()