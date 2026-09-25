import unittest
import json
import io
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import torch
from torch import nn

from infer import (
    aggregate_recirculation_stats,
    apply_repetition_penalty,
    configure_model_generation,
    effective_adaptive_recirculation,
    enable_fp32_output_projection,
    format_average_eval_rating,
    format_index_ranges,
    format_run_arguments,
    format_run_stats,
    has_third_repeated_suffix,
    has_third_repeated_text_suffix,
    output_recirculation_pairs,
    parse_args,
    REPETITION_RECOVERY_TOKEN_COUNT,
    periodic_perturbation_active,
    periodic_perturbation_direction_scale,
    periodic_perturbation_history_direction,
    periodic_perturbation_noise_level_range,
    repeated_text_signature_counts,
    repetition_text_window,
    repetition_recovery_penalty,
    repetition_recovery_settings,
    resolve_gpu_memory_limits,
    resolve_game24_indices,
    StreamingSimilarityWriter,
    summarize_recirculation_stats,
    set_random_seed,
    teacher_forced_token_accuracy,
    top_k_decoded_tokens,
    validate_run_arguments,
)
import infer


class RecirculationStatsTest(unittest.TestCase):
    def test_top_k_decoded_tokens_follow_final_logits(self) -> None:
        logits = torch.tensor([[0.1, 0.8, 0.3, 0.6]])
        tokenizer = SimpleNamespace(decode=lambda token: str(int(token)))

        self.assertEqual(top_k_decoded_tokens(logits, tokenizer, 3), ["1", "3", "2"])
        self.assertEqual(top_k_decoded_tokens(logits, tokenizer, 6), ["1", "3", "2", "0"])

    def test_interval_one_perturbs_first_generated_token(self) -> None:
        model = nn.Module()
        model.config = SimpleNamespace()
        model.generation_config = SimpleNamespace(eos_token_id=2, repetition_penalty=1.0)
        model.embeddings = nn.Embedding(8, 3)
        model.output = nn.Linear(3, 8)
        model.get_input_embeddings = lambda: model.embeddings
        model.get_output_embeddings = lambda: model.output
        tokenizer = SimpleNamespace(
            eos_token_id=2,
            apply_chat_template=lambda *_args, **_kwargs: SimpleNamespace(
                input_ids=torch.tensor([[3, 4, 5]])
            ),
            decode=lambda *_args, **_kwargs: "E",
            encode=lambda *_args, **kwargs: torch.tensor([[1]]) if kwargs.get("return_tensors") else [1],
        )
        calls = []

        def fake_recirculate(tokens, **kwargs):
            calls.append((tokens.tolist(), kwargs["force_recirculation"] if "force_recirculation" in kwargs else False, kwargs["config"]))
            logits = torch.tensor([[[0.0, 5.0, 0.0]]]).expand(1, tokens.shape[1], 3)
            for _ in range(tokens.shape[1]):
                for name, value in (
                    ("recirculated_flags", kwargs.get("force_recirculation", False)),
                    ("rejected_flags", False),
                    ("adaptive_recirculated_flags", False),
                    ("adaptive_rejected_flags", False),
                    ("adaptive_recirculation_counts", 0),
                    ("final_pass_same_top1_flags", False),
                    ("rejection_reasons", ()),
                    ("first_pass_logits", logits[:, :1]),
                    ("first_pass_similarities", (0.5,)),
                    ("pass_probability_margins", [0.8]),
                    ("final_pass_cosine_similarities", None),
                    ("pass_cosine_similarities", [0.3] if kwargs.get("force_recirculation") else []),
                    ("pass_top_k_token_ids", [[1, 0, 2]] if kwargs.get("force_recirculation") else []),
                    ("pass_target_token_probabilities", [0.9] if kwargs.get("force_recirculation") else []),
                    ("cosine_reject_thresholds", 0.2 if kwargs.get("force_recirculation") else 0.8),
                    ("injected_noise_levels", [0.3] if kwargs.get("force_recirculation") else []),
                    ("initial_decoded_noise_source_latents", []),
                    ("decoded_injected_source_latents", [
                        {"margin": 0.1, "topk_tokens": ["Earth", "E", "Mars"]}
                    ] if kwargs.get("force_recirculation") else []),
                ):
                    if kwargs.get(name) is not None:
                        kwargs[name].append(value)
            return logits, kwargs["cache"]

        with tempfile.TemporaryDirectory() as directory:
            for debug, knowedit in ((False, False), (True, False), (True, True)):
                with self.subTest(debug=debug, knowedit=knowedit):
                    calls.clear()
                    with (
                        patch.object(sys, "argv", [
                            "infer.py", *([] if knowedit else ["question"]), "--model", "test-model",
                            "--device-map", "none", "--pair", "2", "0",
                            "--max-new-tokens", "1", "--perturb-every-n-tokens", "1",
                            "--noise-injected-source-top-k", "3",
                            "--output", str(Path(directory) / "output.json"),
                            *(["--knowedit-file", "example.json"] if knowedit else []),
                            *(["--debug"] if debug else []),
                        ]),
                        patch.object(infer, "load_knowedit_examples", return_value=[SimpleNamespace(source="s", subject="s", target_new="E", reference=None)]),
                        patch.object(infer, "format_knowedit_prompt", return_value="question"),
                        patch.object(infer, "teacher_forced_token_accuracy", return_value=1.0),
                        patch.object(infer.AutoTokenizer, "from_pretrained", return_value=tokenizer),
                        patch.object(infer.AutoModelForCausalLM, "from_pretrained", return_value=model),
                        patch.object(infer, "find_decoder_blocks", return_value=nn.ModuleList([nn.Identity() for _ in range(3)])),
                        patch.object(infer, "DynamicCache", return_value=SimpleNamespace(activate_past_recording=lambda: None)),
                        patch.object(infer, "recirculate", side_effect=fake_recirculate),
                        patch.object(infer, "enable_fp32_output_projection"),
                        patch.object(infer.torch.cuda, "is_available", return_value=False),
                        redirect_stdout(io.StringIO()),
                    ):
                        infer.main()
                    result = json.loads((Path(directory) / "output.json").read_text())

                    self.assertEqual([tokens for tokens, _, _ in calls], [[[3, 4]], [[5]], [[1]]])
                    self.assertEqual([forced for _, forced, _ in calls], [False, True, True])
                    self.assertIsNone(calls[0][2].perturbation_direction)
                    if knowedit:
                        self.assertEqual(calls[1][2].perturbation_target_token_id, 1)
                    else:
                        self.assertIsNotNone(calls[1][2].perturbation_direction)
                    self.assertEqual(result[0]["runs"][0]["stats"]["recirculated_tokens"]["count"], 1)
                    if debug:
                        comparisons = json.loads(
                            (Path(directory) / "output-debug.json").read_text()
                        )[0]["similarities"]
                        self.assertTrue(comparisons[0]["recirculated"])
                        self.assertEqual(comparisons[0]["injected_noise_levels"], [0.3])
                        self.assertEqual(comparisons[0]["pass_top_k_cosine_similarities"], [0.3])
                        self.assertEqual(comparisons[0]["pass_recirculation_topk_tokens"], [["E"] * 3])
                        if knowedit:
                            self.assertEqual(comparisons[0]["target_token"], "E")
                            self.assertAlmostEqual(
                                comparisons[0]["pre_recirculation_target_token_probability"],
                                torch.softmax(torch.tensor([0.0, 5.0, 0.0]), dim=-1)[1].item(),
                            )
                            self.assertEqual(comparisons[0]["pass_target_token_probabilities"], [0.9])
                        else:
                            self.assertNotIn("target_token", comparisons[0])
                            self.assertNotIn("pre_recirculation_target_token_probability", comparisons[0])
                        self.assertEqual(comparisons[0]["cosine_reject_threshold"], 0.2)
                        self.assertEqual(comparisons[0]["post_recirculation_topk_tokens"], ["E"] * 3)
                        self.assertEqual(
                            comparisons[0]["noise_injected_source_topk_tokens"],
                            [["Earth", "E", "Mars"]],
                        )
                        self.assertNotIn("noise_injected_source_top1_token", comparisons[0])
                        self.assertNotIn("noise_injected_source_top2_token", comparisons[0])

    def test_teacher_forced_accuracy_scores_before_feeding_each_target_token(self) -> None:
        seen_tokens = []
        top_two_reports = []
        predicted_tokens = iter((3, 9, 5))

        def step(tokens: torch.Tensor) -> torch.Tensor:
            seen_tokens.append(tokens.tolist())
            predicted_token = next(predicted_tokens)
            logits = torch.full((1, 1, 10), -10.0)
            logits[0, 0, 1] = 0.0
            logits[0, 0, predicted_token] = 2.0
            return logits

        accuracy = teacher_forced_token_accuracy(
            torch.tensor([[11, 12]]), torch.tensor([[3, 4, 5]]), step,
            on_top_two=lambda index, tokens: top_two_reports.append((index, tokens)),
        )

        self.assertAlmostEqual(accuracy, 2 / 3)
        self.assertEqual(seen_tokens, [[[11, 12]], [[3]], [[4]]])
        self.assertEqual([index for index, _ in top_two_reports], [0, 1, 2])
        for (_, predictions), expected_top in zip(top_two_reports, (3, 9, 5)):
            self.assertEqual([token_id for token_id, _ in predictions], [expected_top, 1])
            self.assertAlmostEqual(
                predictions[0][1],
                float(torch.softmax(torch.tensor([2.0, 0.0] + [-10.0] * 8), dim=0)[0]),
                delta=1e-6,
            )

    def test_adaptive_recirculation_requires_condition_or_forced_recovery(self) -> None:
        self.assertEqual(effective_adaptive_recirculation(2, False), 0)
        self.assertEqual(effective_adaptive_recirculation(2, True), 2)
        self.assertEqual(effective_adaptive_recirculation(2, False, True), 2)

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

    def test_resolves_automatic_gpu_memory_with_reserve(self) -> None:
        with (
            patch("infer.torch.cuda.device_count", return_value=2),
            patch(
                "infer.torch.cuda.mem_get_info",
                side_effect=[(140 * 1024**3, 141 * 1024**3), (80 * 1024**3, 81 * 1024**3)],
            ),
        ):
            limits = resolve_gpu_memory_limits("auto")

        self.assertEqual(limits, {0: "139264MiB", 1: "77824MiB"})

    def test_preserves_explicit_gpu_memory_limit(self) -> None:
        with patch("infer.torch.cuda.device_count", return_value=2):
            limits = resolve_gpu_memory_limits("46GiB")

        self.assertEqual(limits, {0: "46GiB", 1: "46GiB"})

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

    def test_parses_periodic_perturbation_interval_and_duration(self) -> None:
        args = parse_args(
            [
                "--perturb-every-n-tokens",
                "4",
                "--perturb-for-k-tokens",
                "2",
                "--perturb-recent-m-tokens",
                "12",
                "--perturb-history-decay",
                "0.6",
                "prompt",
            ]
        )

        self.assertEqual(args.perturb_every_n_tokens, 4)
        self.assertEqual(args.perturb_for_k_tokens, 2)
        self.assertEqual(args.perturb_recent_m_tokens, 12)
        self.assertEqual(args.perturb_history_decay, 0.6)

    def test_noise_injected_source_top_k_is_positive(self) -> None:
        self.assertEqual(parse_args(["prompt"]).noise_injected_source_top_k, 4)
        args = parse_args(["--noise-injected-source-top-k", "5", "prompt"])
        self.assertEqual(args.noise_injected_source_top_k, 5)
        validate_run_arguments(args)
        with self.assertRaisesRegex(ValueError, "--noise-injected-source-top-k"):
            validate_run_arguments(
                parse_args(["--noise-injected-source-top-k", "0", "prompt"])
            )

    def test_towards_target_perturbation_requires_knowedit(self) -> None:
        default = parse_args(["prompt"])
        args = parse_args(["--knowedit-file", "example.json"])
        explicit = parse_args(
            ["--knowedit-file", "example.json", "--perturb-mode", "repel-history"]
        )

        self.assertEqual(default.perturb_mode, "repel-history")
        self.assertEqual(args.perturb_mode, "towards-target")
        self.assertEqual(explicit.perturb_mode, "repel-history")
        self.assertEqual(
            parse_args(["--knowedit-file", "example.json", "--ablation"]).perturb_mode,
            "towards-target",
        )
        self.assertEqual(
            parse_args(["--ablation", "--knowedit-file", "example.json"]).perturb_mode,
            "towards-target",
        )
        validate_run_arguments(args)
        with self.assertRaisesRegex(ValueError, "requires --knowedit-file"):
            validate_run_arguments(parse_args(["--perturb-mode", "towards-target"]))

    def test_parses_periodic_perturbation_noise_override(self) -> None:
        default = parse_args(["prompt"])
        args = parse_args(
            ["--perturb-noise-level-range", "0.05", "0.15", "prompt"]
        )

        self.assertEqual(default.perturb_noise_level_range, (0.2, 0.4))
        self.assertEqual(args.perturb_noise_level_range, [0.05, 0.15])

    def test_periodic_perturbation_noise_override_takes_precedence(self) -> None:
        self.assertEqual(
            periodic_perturbation_noise_level_range((0.0, 0.0), None),
            (0.1, 0.2),
        )
        self.assertEqual(
            periodic_perturbation_noise_level_range((0.2, 0.3), None),
            (0.2, 0.3),
        )
        self.assertEqual(
            periodic_perturbation_noise_level_range((0.2, 0.3), (0.0, 0.0)),
            (0.0, 0.0),
        )

    def test_periodic_perturbation_activates_at_the_configured_interval(self) -> None:
        self.assertFalse(periodic_perturbation_active(0, 2, 4))
        self.assertFalse(periodic_perturbation_active(3, 2, 1))
        self.assertTrue(periodic_perturbation_active(3, 2, 3))
        self.assertTrue(periodic_perturbation_active(3, 2, 4))
        self.assertFalse(periodic_perturbation_active(3, 2, 5))
        self.assertTrue(periodic_perturbation_active(3, 2, 6))

    def test_periodic_perturbation_direction_decays_within_interval(self) -> None:
        self.assertEqual(periodic_perturbation_direction_scale(6, 6), 1.0)
        self.assertEqual(periodic_perturbation_direction_scale(6, 7), 0.9)
        self.assertEqual(periodic_perturbation_direction_scale(6, 8), 0.81)
        self.assertEqual(periodic_perturbation_direction_scale(6, 12), 1.0)

    def test_periodic_perturbation_direction_includes_decayed_history(self) -> None:
        direction = periodic_perturbation_history_direction(
            [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 5.0]])], 0.7
        )

        torch.testing.assert_close(direction, torch.tensor([[-3.7, -6.4]]))

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

    def test_parses_sudoku_file_and_indices(self) -> None:
        default = parse_args([])
        args = parse_args(
            [
                "--do-sudoku",
                "--sudoku-file",
                "sudoku.jsonl",
                "--sudoku-index",
                "1-3",
            ]
        )
        last = parse_args(
            ["--do-sudoku", "--sudoku-file", "sudoku.jsonl", "--sudoku-index", "-1"]
        )

        self.assertEqual(
            default.sudoku_file,
            "https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/test.csv",
        )
        self.assertEqual(default.sudoku_index, ())
        self.assertFalse(default.do_sudoku)
        self.assertEqual(args.sudoku_file, Path("sudoku.jsonl"))
        self.assertTrue(args.do_sudoku)
        self.assertEqual(args.sudoku_index, (1, 2, 3))
        self.assertEqual(last.sudoku_index, (-1,))

    def test_parses_bbeh_file_and_indices(self) -> None:
        default = parse_args([])
        args = parse_args(
            ["--bbeh-file", "bbeh/mini/data.json", "--bbeh-index", "1-3"]
        )
        last = parse_args(
            ["--bbeh-file", "bbeh/mini/data.json", "--bbeh-index", "-1"]
        )

        self.assertEqual(default.bbeh_index, ())
        self.assertIsNone(default.bbeh_file)
        self.assertEqual(args.bbeh_file, Path("bbeh/mini/data.json"))
        self.assertEqual(args.bbeh_index, (1, 2, 3))
        self.assertEqual(last.bbeh_index, (-1,))

    def test_parses_knowedit_file_and_indices(self) -> None:
        default = parse_args([])
        args = parse_args(
            ["--knowedit-file", "benchmark/ZsRE/ZsRE-test-all.json", "--knowedit-index", "1-3"]
        )
        last = parse_args(
            ["--knowedit-file", "benchmark/ZsRE/ZsRE-test-all.json", "--knowedit-index", "-1"]
        )

        self.assertIsNone(default.knowedit_file)
        self.assertEqual(default.knowedit_index, ())
        self.assertEqual(args.knowedit_file, Path("benchmark/ZsRE/ZsRE-test-all.json"))
        self.assertEqual(args.knowedit_index, (1, 2, 3))
        self.assertEqual(last.knowedit_index, (-1,))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--knowedit-file", "example.json", "--bbeh-file", "other.json"])

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

    def test_parses_repetition_recovery_toggle(self) -> None:
        self.assertTrue(parse_args(["prompt"]).repetition_recovery)
        self.assertFalse(
            parse_args(["--no-repetition-recovery", "prompt"]).repetition_recovery
        )

    def test_rejects_removed_forced_recirculation_option(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--forced-recirculation-budget", "7", "prompt"])

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

    def test_detects_three_consecutive_blank_lines(self) -> None:
        text = "$286 = (100 + 4) \\times 3$\n\n\n\n"

        self.assertTrue(has_third_repeated_text_suffix(text))

    def test_detects_three_consecutive_single_symbol_lines(self) -> None:
        text = "$286 = (100 + 4) \\times 3$\n$\n$\n$"

        self.assertTrue(has_third_repeated_text_suffix(text))

    def test_repetition_text_window_discards_distant_attempts(self) -> None:
        repeated_attempt = "100 * (6 + 3) - (5 * 7 + 2 + 6)... no."
        text = "\n".join(
            (repeated_attempt, repeated_attempt, repeated_attempt, "later " * 10)
        )

        self.assertTrue(
            has_third_repeated_text_suffix(repetition_text_window(text, 64))
        )
        self.assertFalse(
            has_third_repeated_text_suffix(repetition_text_window(text, 10))
        )

    def test_repeated_signature_count_increases_after_the_third_attempt(self) -> None:
        repeated_attempt = "858 = (100 + 25 + 3 * 6) * 6 ... no."
        third_attempt = "\n".join(repeated_attempt for _ in range(3))
        fourth_attempt = f"{third_attempt}\n{repeated_attempt}"

        third_counts = repeated_text_signature_counts(third_attempt)
        fourth_counts = repeated_text_signature_counts(fourth_attempt)

        self.assertEqual(list(third_counts.values()), [3])
        self.assertEqual(list(fourth_counts.values()), [4])

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
        self.assertEqual(settings.cosine_reject, 0.3)
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

    def test_recovery_attempt_is_absent_without_a_new_trigger(self) -> None:
        repetition_present = True
        repetition_detection_active = True
        repetition_detected = repetition_present and not repetition_detection_active
        attempt = 2 if repetition_detected else None

        self.assertFalse(repetition_detected)
        self.assertIsNone(attempt)

    def test_historical_repetition_does_not_force_recovery_without_new_trigger(self) -> None:
        settings = repetition_recovery_settings(
            (0.1, 0.2),
            0.8,
            0.2,
            (0.1, 0.2),
            1.2,
            (0.7,),
            False,
            list(range(8)) * 3,
            decoded_text="eight repeated words appear here exactly as before\n" * 3,
            repetition_detected=False,
        )

        self.assertFalse(settings.force_recirculation)
        self.assertEqual(settings.noise_level_range, (0.1, 0.2))
        self.assertEqual(settings.cosine_reject, 0.8)

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

        self.assertEqual(settings.noise_level_range, (0.4, 0.4))

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

    def test_uses_two_token_repetition_recovery_penalty(self) -> None:
        self.assertEqual(repetition_recovery_penalty(False), 1.0)
        self.assertEqual(REPETITION_RECOVERY_TOKEN_COUNT, 2)
        self.assertEqual(repetition_recovery_penalty(True), 1.1)
        self.assertEqual(repetition_recovery_penalty(True), 1.1)
        self.assertEqual(repetition_recovery_penalty(False), 1.0)

    def test_skips_repetition_penalty_after_accepted_recirculation(self) -> None:
        logits = torch.tensor([[-2.0, 2.0, 4.0]])
        previous_token_ids = torch.tensor([[0, 1]])

        sampling_logits = apply_repetition_penalty(
            logits, previous_token_ids, 1.1, recirculated=True
        )

        torch.testing.assert_close(sampling_logits, logits)

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