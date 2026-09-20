import unittest
import json
import tempfile
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import torch
from torch import nn

from infer import (
    aggregate_recirculation_stats,
    apply_repetition_penalty,
    configure_model_generation,
    enable_fp32_output_projection,
    format_average_eval_rating,
    format_index_ranges,
    format_run_arguments,
    format_run_stats,
    generated_cosine_reject,
    generated_noise_level_range,
    generated_pre_margin_threshold,
    has_third_repeated_suffix,
    has_third_repeated_text_suffix,
    output_recirculation_pairs,
    parse_args,
    repetition_recovery_settings,
    resolve_game24_indices,
    StreamingSimilarityWriter,
    summarize_recirculation_stats,
    set_random_seed,
    validate_run_arguments,
)


class RecirculationStatsTest(unittest.TestCase):
    def test_streams_valid_similarity_json_after_each_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "debug.json"
            writer = StreamingSimilarityWriter(path)
            writer.start_run("prompt", "run", 7)
            self.assertEqual(json.loads(path.read_text())[0]["similarities"], [])

            writer.append({"token_index": 0})
            first_snapshot = json.loads(path.read_text())
            writer.append({"token_index": 1})
            second_snapshot = json.loads(path.read_text())
            writer.start_run("another prompt", "another run", 8)
            writer.append({"token_index": 0})
            multi_run_snapshot = json.loads(path.read_text())
            writer.close()

        self.assertEqual(first_snapshot[0]["similarities"], [{"token_index": 0}])
        self.assertEqual(
            second_snapshot[0]["similarities"],
            [{"token_index": 0}, {"token_index": 1}],
        )
        self.assertEqual(len(multi_run_snapshot), 2)
        self.assertEqual(
            multi_run_snapshot[1]["similarities"], [{"token_index": 0}]
        )

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

    def test_parses_a_game24_puzzle(self) -> None:
        args = parse_args(["--game24-puzzle", "2", "3", "4", "6"])

        self.assertEqual(args.game24_puzzle, [2, 3, 4, 6])
        self.assertEqual(args.game24_index, ())

    def test_parses_a_countdown_puzzle(self) -> None:
        args = parse_args(
            ["--countdown-puzzle", "999", "100", "50", "25", "10", "2", "1"]
        )

        self.assertEqual(args.countdown_puzzle, [999, 100, 50, 25, 10, 2, 1])

    def test_parses_countdown_file_and_indices(self) -> None:
        default = parse_args(["--countdown-file", "countdown.csv"])
        args = parse_args(
            ["--countdown-file", "countdown.csv", "--countdown-index", "1-3"]
        )
        last = parse_args(
            ["--countdown-file", "countdown.csv", "--countdown-index", "-1"]
        )

        self.assertEqual(default.countdown_index, ())
        self.assertEqual(args.countdown_file, Path("countdown.csv"))
        self.assertEqual(args.countdown_index, (1, 2, 3))
        self.assertEqual(last.countdown_index, (-1,))

    def test_parses_game24_indices_beyond_builtin_queries(self) -> None:
        args = parse_args(
            ["--game24-file", "24.csv", "--game24-index", "21-23"]
        )

        self.assertEqual(args.game24_index, (21, 22, 23))

    def test_parses_and_resolves_negative_game24_indices(self) -> None:
        args = parse_args(["--game24-file", "24.csv", "--game24-index", "-3--1"])
        last = parse_args(["--game24-file", "24.csv", "--game24-index", "-1"])

        self.assertEqual(args.game24_index, (-3, -2, -1))
        self.assertEqual(last.game24_index, (-1,))
        self.assertEqual(resolve_game24_indices(args.game24_index, 10), (8, 9, 10))

    def test_rejects_zero_game24_index(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--game24-file", "24.csv", "--game24-index", "0"])

    def test_compacts_contiguous_indices_for_filenames(self) -> None:
        self.assertEqual(format_index_ranges((1, 2, 3, 7, 9, 10)), "1-3,7,9-10")

    def test_parses_startup_relaxation(self) -> None:
        args = parse_args(
            [
                "--startup-relax-tokens",
                "3",
                "--startup-noise-level-range",
                "0.2",
                "0.4",
                "--startup-cosine-reject",
                "0.9",
                "--pre-margin-relax-factor",
                "1.5",
                "prompt",
            ]
        )

        self.assertEqual(args.startup_relax_tokens, 3)
        self.assertEqual(args.startup_noise_level_range, [0.2, 0.4])
        self.assertEqual(args.startup_cosine_reject, 0.9)
        self.assertEqual(args.pre_margin_relax_factor, 1.5)

    def test_relaxes_only_the_first_generated_tokens(self) -> None:
        self.assertEqual(generated_pre_margin_threshold(0.2, 0, 1.5, 1), 0.2)
        self.assertAlmostEqual(generated_pre_margin_threshold(0.2, 3, 1.5, 1), 0.3)
        self.assertAlmostEqual(generated_pre_margin_threshold(0.2, 3, 1.5, 3), 0.3)
        self.assertEqual(generated_pre_margin_threshold(0.2, 3, 1.5, 4), 0.2)

    def test_uses_startup_noise_only_for_startup_tokens(self) -> None:
        base_range = (0.1, 0.2)
        startup_range = (0.3, 0.4)

        self.assertEqual(
            generated_noise_level_range(base_range, startup_range, 3, 1),
            startup_range,
        )
        self.assertEqual(
            generated_noise_level_range(base_range, startup_range, 3, 3),
            startup_range,
        )
        self.assertEqual(
            generated_noise_level_range(base_range, startup_range, 3, 4),
            base_range,
        )
        self.assertEqual(
            generated_noise_level_range(base_range, (0.0, 0.0), 3, 1),
            base_range,
        )

    def test_uses_startup_cosine_rejection_only_for_startup_tokens(self) -> None:
        self.assertEqual(generated_cosine_reject(0.8, 0.9, 3, 1), 0.9)
        self.assertEqual(generated_cosine_reject(0.8, 0.9, 3, 3), 0.9)
        self.assertEqual(generated_cosine_reject(0.8, 0.9, 3, 4), 0.8)
        self.assertEqual(generated_cosine_reject(0.8, None, 3, 1), 0.8)

    def test_parses_repetition_recovery_toggle(self) -> None:
        self.assertTrue(parse_args(["prompt"]).repetition_recovery)
        self.assertFalse(
            parse_args(["--no-repetition-recovery", "prompt"]).repetition_recovery
        )

    def test_detects_a_third_repeated_generated_token_sequence(self) -> None:
        repeated = list(range(8)) * 3

        self.assertTrue(has_third_repeated_suffix(repeated))
        self.assertFalse(has_third_repeated_suffix(repeated + [11]))

    def test_detects_an_interleaved_repeated_suffix(self) -> None:
        repeated_answer = list(range(8))
        token_ids = (
            [21, 22, 23]
            + repeated_answer
            + [24, 25, 26]
            + repeated_answer
            + [27, 28, 29]
            + repeated_answer
        )

        self.assertTrue(has_third_repeated_suffix(token_ids))

    def test_detects_a_repeated_decoded_text_suffix(self) -> None:
        text = "\n".join(
            (
                "10. $106 \\times 8 = 848$, leaving 52 to construct.",
                "11. $848 - (9 - 6 + 1)$... No.",
                "12. $106 \\times 8 = 848$, leaving 52 to construct.",
                "13. $848 - (9 - 3 - 2)$... No.",
                "14. $106 \\times 8 = 848$, leaving 52 to construct.",
            )
        )

        self.assertTrue(has_third_repeated_text_suffix(text))

    def test_detects_a_repeated_text_span_before_later_text(self) -> None:
        repeated_attempt = (
            "Let's try: $100 \\times (6 + 3) - (5 \\times 7 + 2 + 6)$... no."
        )
        text = "\n".join(
            (
                f"43. {repeated_attempt}",
                f"44. {repeated_attempt}",
                f"45. {repeated_attempt}",
                "(Wait, 100 times 9 is 900. Can we make 43 with 2, 5, 7?)",
            )
        )

        self.assertTrue(has_third_repeated_text_suffix(text))

    def test_ignores_shared_prefix_of_distinct_attempts(self) -> None:
        text = "\n".join(
            (
                "12. Let's try: $6 \\times (100 + 3 \\times 7 \\times 2) = 852$.",
                "13. Let's try: $100 \\times (2 + 7) - (6 \\times 7 + 1)$... no.",
                "14. Let's try: $(100 - 5) \\times (3 \\times 3)$... no.",
                "15. Let's try: $100 \\times",
            )
        )

        self.assertFalse(has_third_repeated_text_suffix(text))

    def test_ignores_long_shared_prefix_of_distinct_attempts(self) -> None:
        text = "\n".join(
            (
                "28. Let's try: $100 \\times (6 + 3) - (5 \\times 7 + 2 \\times 4)$... no.",
                "29. Let's try: $100 \\times (6 + 3) - (5 \\times 7 + 2 \\times 3)$... no.",
                "30. Let's try: $100 \\times (6 + 3) - (5 - 2) \\times (7 + 7)$... no.",
            )
        )

        self.assertFalse(has_third_repeated_text_suffix(text))

    def test_ignores_repeated_short_prose_fragments(self) -> None:
        text = "\n".join(
            (
                "1. First, multiply 9 and 25 to get 225.",
                "2. Subtract 6 from 225 to get 219.",
                "3. Multiply 219 by 4 to get 876.",
                "4. Multiply 8 and 4 to get 32.",
                "Alternative path:",
                "1. Multiply 25 by 3 to get 75.",
            )
        )

        self.assertFalse(has_third_repeated_text_suffix(text))

    def test_ignores_repeated_short_restart_headers(self) -> None:
        text = "\n".join(
            (
                "Let's try:",
                "1. $25 \\times 8 \\times 4 = 800$.",
                "2. $844 - 800 = 44$.",
                "Let's try:",
                "1. $25 + 6 = 31$.",
                "2. $31 \\times 3 \\times 9 = 837$.",
                "Let's try:",
            )
        )

        self.assertFalse(has_third_repeated_text_suffix(text))

    def test_decoded_text_takes_priority_over_raw_token_repetition(self) -> None:
        settings = repetition_recovery_settings(
            (0.1, 0.2),
            0.8,
            0.2,
            (0.1, 0.2),
            1.2,
            (0.7,),
            False,
            list(range(8)) * 3,
            decoded_text="Let's try:\nLet's try:\nLet's try:",
        )

        self.assertEqual(settings.noise_level_range, (0.1, 0.2))
        self.assertEqual(settings.cosine_reject, 0.8)
        self.assertEqual(settings.pre_margin_threshold, 0.2)
        self.assertEqual(settings.post_margin_threshold, (0.1, 0.2))
        self.assertEqual(settings.post_margin_ratio_threshold, 1.2)
        self.assertEqual(settings.condition_thresholds, (0.7,))
        self.assertFalse(settings.recirculation_allowed)
        self.assertFalse(settings.force_recirculation)

    def test_preserves_configured_noise_during_repetition_recovery(self) -> None:
        settings = repetition_recovery_settings(
            (0.1, 0.2),
            0.8,
            0.2,
            (0.1, 0.2),
            1.2,
            (0.7,),
            False,
            list(range(8)) * 3,
        )

        self.assertEqual(settings.noise_level_range, (0.1, 0.2))
        self.assertIsNone(settings.cosine_reject)
        self.assertIsNone(settings.pre_margin_threshold)
        self.assertIsNone(settings.post_margin_threshold)
        self.assertIsNone(settings.post_margin_ratio_threshold)
        self.assertIsNone(settings.condition_thresholds)
        self.assertTrue(settings.recirculation_allowed)
        self.assertTrue(settings.force_recirculation)

    def test_enables_noise_during_repetition_recovery_when_disabled(self) -> None:
        settings = repetition_recovery_settings(
            (0.0, 0.0),
            0.8,
            0.2,
            (0.1, 0.2),
            1.2,
            (0.7,),
            False,
            list(range(8)) * 3,
        )

        self.assertEqual(settings.noise_level_range, (0.1, 0.2))

    def test_doubles_noise_for_consecutive_repetition_recovery(self) -> None:
        settings = repetition_recovery_settings(
            (0.1, 0.2),
            0.8,
            0.2,
            (0.1, 0.2),
            1.2,
            (0.7,),
            False,
            list(range(8)) * 3,
            consecutive_repetition_count=2,
        )

        self.assertEqual(settings.noise_level_range, (0.2, 0.4))

    def test_escalates_noise_after_a_second_failed_recovery(self) -> None:
        first_recovery = repetition_recovery_settings(
            (0.1, 0.2),
            0.8,
            0.2,
            (0.1, 0.2),
            1.2,
            (0.7,),
            False,
            list(range(8)) * 3,
            consecutive_repetition_count=1,
        )
        second_recovery = repetition_recovery_settings(
            (0.1, 0.2),
            0.8,
            0.2,
            (0.1, 0.2),
            1.2,
            (0.7,),
            False,
            list(range(8)) * 3,
            consecutive_repetition_count=2,
        )

        self.assertEqual(first_recovery.noise_level_range, (0.1, 0.2))
        self.assertEqual(second_recovery.noise_level_range, (0.2, 0.4))

    def test_caps_noise_for_consecutive_repetition_recovery(self) -> None:
        settings = repetition_recovery_settings(
            (3.0, 6.0),
            0.8,
            0.2,
            (0.1, 0.2),
            1.2,
            (0.7,),
            False,
            list(range(8)) * 3,
            consecutive_repetition_count=2,
        )

        self.assertEqual(settings.noise_level_range, (1.0, 1.0))

    def test_can_disable_repetition_recovery(self) -> None:
        settings = repetition_recovery_settings(
            (0.1, 0.2),
            0.8,
            0.2,
            (0.1, 0.2),
            1.2,
            (0.7,),
            False,
            list(range(8)) * 3,
            enabled=False,
        )

        self.assertEqual(settings.noise_level_range, (0.1, 0.2))
        self.assertEqual(settings.cosine_reject, 0.8)
        self.assertEqual(settings.pre_margin_threshold, 0.2)
        self.assertEqual(settings.post_margin_threshold, (0.1, 0.2))
        self.assertEqual(settings.post_margin_ratio_threshold, 1.2)
        self.assertEqual(settings.condition_thresholds, (0.7,))
        self.assertFalse(settings.recirculation_allowed)
        self.assertFalse(settings.force_recirculation)

    def test_accepts_startup_noise_level_one(self) -> None:
        args = parse_args(
            [
                "--startup-relax-tokens",
                "1",
                "--startup-noise-level-range",
                "0.5",
                "1",
                "prompt",
            ]
        )

        validate_run_arguments(args)

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

    def test_ablation_seed_overrides_only_its_run(self) -> None:
        args = parse_args(
            [
                "--seed",
                "11",
                "--ablation",
                "--alpha",
                "0.25",
                "--ablation",
                "--seed",
                "17",
            ]
        )

        self.assertEqual(args.seed, 11)
        self.assertEqual(args.ablations, ((('alpha', 0.25),), (('seed', 17),)))

    def test_run_signature_includes_seed(self) -> None:
        args = parse_args(["--model", "openai/gpt-oss-20b", "--seed", "17"])

        self.assertEqual(
            format_run_arguments(args, ("model", "seed")),
            "model=gpt-oss-20b, seed=17",
        )

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