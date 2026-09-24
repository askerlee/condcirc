import unittest
from unittest.mock import patch

import torch
from torch import nn

from recirculation import (
    RecirculationConfig,
    _Hooks,
    _adaptive_noise_level,
    recirculate,
)


class RecirculationCacheTest(unittest.TestCase):
    def test_periodic_probe_selects_highest_target_probability(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        selected_inputs: list[torch.Tensor] = []

        def step(token: torch.Tensor, cache: list[int]):
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            selected_inputs.append(hidden.detach().clone())
            cache.append(len(cache))
            return torch.tensor([[[2.0, 1.0, 0.0]]]), cache

        with patch(
            "recirculation.torch.randn_like",
            side_effect=[
                torch.tensor([[[1.0, 0.0]]]),
                torch.tensor([[[-1.0, 0.0]]]),
                torch.tensor([[[0.0, 1.0]]]),
                torch.tensor([[[0.0, -1.0]]]),
            ],
        ):
            recirculate(
                torch.tensor([[1]]),
                blocks=blocks,
                cache=[],
                step=step,
                rewind_one=lambda cache: cache[:-1],
                config=RecirculationConfig(
                    pairs=((2, 0),),
                    alpha=0.5,
                    noise_level_range=(0.2, 0.2),
                    perturbation_target_token_id=1,
                ),
                passes=1,
                force_recirculation=True,
                decode_injected_source_latent=lambda *_args: {"margin": 0.0},
                perturbation_probe=lambda _token, _source, _destination, _target, candidate, *_args: torch.cat(
                    (
                        2 * candidate[:, -1, :1],
                        candidate[:, -1, :1],
                        torch.zeros_like(candidate[:, -1, :1]),
                    ),
                    dim=-1,
                ),
                capture_cached_token=lambda cache: cache[-1],
                restore_cached_token=lambda _cache, _cached_token: None,
            )

        expected = torch.tensor([[[0.8, 1.0]]])
        expected = expected * 2**0.5 / torch.linalg.vector_norm(
            expected, dim=-1, keepdim=True
        )
        torch.testing.assert_close(selected_inputs[-1], 0.5 * (torch.ones_like(expected) + expected))

    def test_periodic_probe_selects_least_aligned_noise_candidate(self) -> None:
        block_one_inputs: list[torch.Tensor] = []

        class CaptureBlock(nn.Module):
            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                block_one_inputs.append(hidden_states.detach().clone())
                return hidden_states

        blocks = nn.ModuleList([nn.Identity(), CaptureBlock(), nn.Identity()])

        def step(token: torch.Tensor, cache: list[int]):
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            cache.append(len(cache))
            return torch.tensor([[[2.0, 1.0, 0.0]]]), cache

        with patch(
            "recirculation.torch.randn_like",
            side_effect=[
                torch.tensor([[[1.0, 0.0]]]),
                torch.tensor([[[2.0, 0.0]]]),
                torch.tensor([[[-1.0, 0.0]]]),
                torch.tensor([[[0.0, 1.0]]]),
            ],
        ) as random_noise:
            recirculate(
                torch.tensor([[1]]),
                blocks=blocks,
                cache=[],
                step=step,
                rewind_one=lambda cache: cache[:-1],
                config=RecirculationConfig(
                    pairs=((2, 0),),
                    alpha=0.5,
                    noise_level_range=(0.2, 0.2),
                    perturbation_direction=torch.tensor([[1.0, 0.0]]),
                ),
                passes=1,
                force_recirculation=True,
                decode_injected_source_latent=lambda *_args: {"margin": 0.0},
                perturbation_probe=lambda _token, _source, _destination, _target, candidate, *_args: candidate[:, -1, :],
                capture_cached_token=lambda cache: cache[-1],
                restore_cached_token=lambda _cache, _cached_token: None,
            )

        self.assertEqual(random_noise.call_count, 4)
        expected = torch.tensor([[[0.8, 1.0]]])
        expected = expected * 2**0.5 / torch.linalg.vector_norm(
            expected, dim=-1, keepdim=True
        )
        expected = 0.5 * (torch.ones_like(expected) + expected)
        torch.testing.assert_close(block_one_inputs[-1], expected)

    def test_forced_recirculation_adds_configured_direction_to_noisy_pass(self) -> None:
        block_one_inputs: list[torch.Tensor] = []

        class CaptureBlock(nn.Module):
            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                block_one_inputs.append(hidden_states.detach().clone())
                return hidden_states

        blocks = nn.ModuleList([nn.Identity(), CaptureBlock(), nn.Identity()])

        def step(token: torch.Tensor, cache: list[int]):
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            cache.append(len(cache))
            return torch.tensor([[[2.0, 1.0, 0.0]]]), cache

        with patch("recirculation.torch.randn_like", return_value=torch.zeros((1, 1, 2))):
            recirculate(
                torch.tensor([[1]]),
                blocks=blocks,
                cache=[],
                step=step,
                rewind_one=lambda cache: cache[:-1],
                config=RecirculationConfig(
                    pairs=((2, 0),),
                    alpha=0.5,
                    noise_level_range=(0.2, 0.2),
                    perturbation_direction=torch.tensor([[-1.0, 0.0]]),
                    perturbation_direction_scale=0.5,
                ),
                passes=1,
                force_recirculation=True,
                decode_injected_source_latent=lambda *_args: {"margin": 0.0},
                capture_cached_token=lambda cache: cache[-1],
                restore_cached_token=lambda _cache, _cached_token: None,
            )

        expected = torch.tensor([[[1.0 - 0.05 * 2**0.5, 1.0]]])
        expected = expected * 2**0.5 / torch.linalg.vector_norm(
            expected, dim=-1, keepdim=True
        )
        torch.testing.assert_close(block_one_inputs[-1], expected)

    def test_force_recirculation_adds_noisy_pass_without_configured_passes(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        recirculated_flags: list[bool] = []
        adaptive_recirculation_counts: list[int] = []
        injected_noise_levels: list[list[float]] = []

        def step(token: torch.Tensor, cache: list[int]):
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            cache.append(len(cache))
            return torch.tensor([[[2.0, 1.0, 0.0]]]), cache

        recirculate(
            torch.tensor([[1]]),
            blocks=blocks,
            cache=[],
            step=step,
            rewind_one=lambda cache: cache[:-1],
            config=RecirculationConfig(
                pairs=((2, 0),), alpha=0.5, noise_level_range=(0.2, 0.4)
            ),
            passes=1,
            adaptive_recirculation=0,
            force_recirculation=True,
            decode_injected_source_latent=lambda *_args: {"margin": 0.0},
            capture_cached_token=lambda cache: cache[-1],
            restore_cached_token=lambda _cache, _cached_token: None,
            recirculated_flags=recirculated_flags,
            adaptive_recirculation_counts=adaptive_recirculation_counts,
            injected_noise_levels=injected_noise_levels,
        )

        self.assertEqual(recirculated_flags, [True])
        self.assertEqual(adaptive_recirculation_counts, [0])
        self.assertEqual(injected_noise_levels, [[0.4]])

    def test_forced_recirculation_retries_cosine_rejected_noise(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        pass_logits = (
            torch.tensor([[[5.0, 0.0, 0.0]]]),
            torch.tensor([[[0.0, 5.0, 0.0]]]),
            torch.tensor([[[4.0, 0.0, 0.0]]]),
        )
        call_count = 0
        rejected_flags: list[bool] = []
        rejection_reasons: list[tuple[str, ...]] = []

        def step(token: torch.Tensor, current_cache: list[int]):
            nonlocal call_count
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            current_cache.append(call_count + 1)
            logits = pass_logits[call_count]
            call_count += 1
            return logits, current_cache

        def rewind_one(current_cache: list[int]) -> list[int]:
            current_cache.pop()
            return current_cache

        def restore_cached_token(
            current_cache: list[int], cached_token: int
        ) -> None:
            current_cache[-1] = cached_token

        with patch(
            "recirculation._distribution_cosine_similarity",
            side_effect=(0.2, 0.4),
        ) as cosine_similarity:
            logits, final_cache = recirculate(
                torch.tensor([[1]]),
                blocks=blocks,
                cache=cache,
                step=step,
                rewind_one=rewind_one,
                config=RecirculationConfig(
                    pairs=((2, 0),), alpha=0.5, noise_level_range=(0.2, 0.4)
                ),
                passes=1,
                force_recirculation=True,
                condition_thresholds=(2.0,),
                pre_margin_threshold=-1.0,
                post_margin_threshold=(1.0, 1.0),
                post_margin_ratio_threshold=99.0,
                cosine_reject=0.8,
                decode_injected_source_latent=lambda *_args: {"margin": 0.0},
                capture_cached_token=lambda current_cache: current_cache[-1],
                restore_cached_token=restore_cached_token,
                rejected_flags=rejected_flags,
                rejection_reasons=rejection_reasons,
            )

        torch.testing.assert_close(logits, pass_logits[2])
        self.assertEqual(final_cache, [1])
        self.assertEqual(call_count, 3)
        self.assertEqual(cosine_similarity.call_count, 2)
        self.assertEqual(rejected_flags, [False])
        self.assertEqual(rejection_reasons, [()])

    def test_accepts_noise_level_one(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])

        _Hooks(
            blocks,
            RecirculationConfig(
                pairs=((2, 0),), alpha=0.5, noise_level_range=(0.5, 1.0)
            ),
        )

    def test_forced_recirculation_allows_noise_at_one(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])

        _Hooks(
            blocks,
            RecirculationConfig(
                pairs=((2, 0),), alpha=0.5, noise_level_range=(0.8, 1.0)
            ),
            force_recirculation=True,
        )

    def test_forced_recirculation_rejects_noise_above_one(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])

        with self.assertRaisesRegex(ValueError, "MAX <= 1"):
            _Hooks(
                blocks,
                RecirculationConfig(
                    pairs=((2, 0),), alpha=0.5, noise_level_range=(8.0, 12.0)
                ),
                force_recirculation=True,
            )

    def test_normal_recirculation_rejects_noise_above_one(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])

        with self.assertRaisesRegex(ValueError, "MAX <= 1"):
            _Hooks(
                blocks,
                RecirculationConfig(
                    pairs=((2, 0),), alpha=0.5, noise_level_range=(0.8, 1.6)
                ),
            )

    def test_noise_injection_skips_small_pre_margin(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        injected_source_latents: list[tuple[int, torch.Tensor]] = []

        def step(token: torch.Tensor, current_cache: list[int]):
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            current_cache.append(len(current_cache))
            return torch.tensor([[[0.1, 0.0, 0.0]]]), current_cache

        recirculate(
            torch.tensor([[1]]),
            blocks=blocks,
            cache=[],
            step=step,
            rewind_one=lambda current_cache: current_cache[:-1],
            config=RecirculationConfig(
                pairs=((2, 0),),
                alpha=0.5,
                noise_level_range=(0.1, 0.2),
                perturb_pre_margin_thres=0.05,
            ),
            passes=2,
            post_margin_threshold=(0.1, 0.2),
            injected_source_latents=injected_source_latents,
            decode_injected_source_latent=lambda _token, _source, latent, _cache, _state: {},
            capture_cached_token=lambda current_cache: current_cache[-1],
            restore_cached_token=lambda current_cache, cached_token: None,
        )

        self.assertEqual(injected_source_latents, [[]])

    def test_noise_injection_uses_decay_on_later_passes(self) -> None:
        def injected_noise_levels_for_decay(decay: float) -> list[float]:
            blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
            injected_noise_levels: list[list[float]] = []

            def step(token: torch.Tensor, current_cache: list[int]):
                hidden = torch.ones((1, 1, 2))
                for block in blocks:
                    hidden = block(hidden)
                current_cache.append(len(current_cache))
                return torch.tensor([[[2.0, 0.0, 0.0]]]), current_cache

            recirculate(
                torch.tensor([[1]]),
                blocks=blocks,
                cache=[],
                step=step,
                rewind_one=lambda current_cache: current_cache[:-1],
                config=RecirculationConfig(
                    pairs=((2, 0),),
                    alpha=0.5,
                    noise_level_range=(0.2, 0.2),
                    noise_decay_per_pass=decay,
                ),
                passes=3,
                post_margin_threshold=(0.1, 0.2),
                injected_noise_levels=injected_noise_levels,
                decode_injected_source_latent=(
                    lambda _token, _source, _latent, _cache, _state: {"margin": 0.0}
                ),
                capture_cached_token=lambda current_cache: current_cache[-1],
                restore_cached_token=lambda current_cache, cached_token: None,
            )
            return injected_noise_levels[0]

        self.assertEqual(injected_noise_levels_for_decay(0.0), [0.2, 0.0])
        self.assertEqual(injected_noise_levels_for_decay(0.5), [0.2, 0.1])

    def test_adaptive_noise_level_normalizes_and_caps_margin(self) -> None:
        levels = [
            _adaptive_noise_level(margin, (0.1, 0.3), (0.1, 0.4))
            for margin in (0.0, 0.2, 0.4)
        ]

        self.assertEqual(levels[0], 0.1)
        self.assertAlmostEqual(levels[1], 0.25)
        self.assertEqual(levels[2], 0.4)

    def test_noise_is_magnitude_matched_to_source(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        injected_source_latents: list[tuple[int, torch.Tensor]] = []
        hooks = _Hooks(
            blocks,
            RecirculationConfig(
                pairs=((2, 0),), alpha=1.0, noise_level_range=(0.0, 0.25)
            ),
            injected_source_latents=injected_source_latents,
        )
        destination = torch.tensor([[[3.0, 4.0]]])
        source = torch.tensor([[[1.0, 0.0]]])
        hooks.residuals[0] = destination
        hooks.injection_sources[2] = source
        hooks.active_pairs = (True,)
        hooks.mode = "inject"
        hooks.noise_level = 0.25

        torch.manual_seed(7)
        hooks.prepare_injections()
        mixed = blocks[1](destination)
        hooks.close()

        normalized_source = torch.tensor([[[5.0, 0.0]]])
        perturbation = mixed - normalized_source
        torch.testing.assert_close(
            torch.linalg.vector_norm(perturbation, dim=-1),
            0.25 * torch.linalg.vector_norm(normalized_source, dim=-1),
        )
        debug_latent = injected_source_latents[0][1]
        debug_perturbation = debug_latent - source
        torch.testing.assert_close(
            torch.linalg.vector_norm(debug_perturbation, dim=-1),
            0.25 * torch.linalg.vector_norm(source, dim=-1),
        )

    def test_noise_is_decayed_on_later_recirculation_passes(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        hooks = _Hooks(
            blocks,
            RecirculationConfig(
                pairs=((2, 0),),
                alpha=1.0,
                noise_level_range=(0.0, 0.4),
                noise_decay_per_pass=0.5,
            ),
        )
        destination = torch.tensor([[[3.0, 4.0]]])
        source = torch.tensor([[[1.0, 0.0]]])
        hooks.residuals[0] = destination
        hooks.injection_sources[2] = source
        hooks.active_pairs = (True,)
        hooks.mode = "inject"
        hooks.noise_level = 0.4
        normalized_source = torch.tensor([[[5.0, 0.0]]])

        perturbation_norms = []
        for pass_index in (1, 2, 3):
            hooks.pass_index = pass_index
            torch.manual_seed(7)
            hooks.prepare_injections()
            mixed = blocks[1](destination)
            perturbation_norms.append(
                torch.linalg.vector_norm(mixed - normalized_source, dim=-1)
            )
        hooks.close()

        torch.testing.assert_close(
            torch.stack(perturbation_norms),
            torch.tensor([[[2.0]], [[1.0]], [[0.5]]]),
        )

    def test_later_recirculation_passes_use_decayed_noise(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        injected_noise_levels: list[list[float]] = []
        injected_source_latents: list[list[tuple[int, torch.Tensor]]] = []
        decoded_injected_source_latents: list[list[dict[str, object]]] = []
        pass_logits = (
            torch.tensor([[[2.0, 1.0, 0.0]]]),
            torch.tensor([[[1.5, 1.0, 0.0]]]),
            torch.tensor([[[1.0, 0.9, 0.0]]]),
        )
        call_count = 0

        def step(token: torch.Tensor, current_cache: list[int]):
            nonlocal call_count
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            current_cache.append(call_count + 1)
            logits = pass_logits[call_count]
            call_count += 1
            return logits, current_cache

        with patch(
            "recirculation._adaptive_noise_level",
            wraps=_adaptive_noise_level,
        ) as adaptive_noise_level:
            recirculate(
                torch.tensor([[1]]),
                blocks=blocks,
                cache=cache,
                step=step,
                rewind_one=lambda current_cache: current_cache[:-1],
                config=RecirculationConfig(
                    pairs=((2, 0),),
                    alpha=0.5,
                    noise_level_range=(0.0, 0.4),
                    noise_decay_per_pass=0.5,
                ),
                passes=3,
                post_margin_threshold=(0.0, 1.0),
                injected_noise_levels=injected_noise_levels,
                injected_source_latents=injected_source_latents,
                decode_injected_source_latent=(
                    lambda _token, _source_index, _latent, current_cache, _state: (
                        {"margin": 0.0, "cache": tuple(current_cache)}
                    )
                ),
                decoded_injected_source_latents=decoded_injected_source_latents,
                capture_cached_token=lambda current_cache: current_cache[-1],
                restore_cached_token=lambda current_cache, cached_token: None,
            )

        used_margins = [call.args[0] for call in adaptive_noise_level.call_args_list]
        expected_margins = [
            float(
                (
                    torch.softmax(logits, dim=-1).topk(2, dim=-1).values[..., 0]
                    - torch.softmax(logits, dim=-1).topk(2, dim=-1).values[..., 1]
                ).item()
            )
            for logits in pass_logits[:2]
        ]
        self.assertEqual(used_margins, expected_margins)
        self.assertEqual(len(injected_noise_levels), 1)
        self.assertAlmostEqual(injected_noise_levels[0][0], 0.4 * expected_margins[0])
        self.assertAlmostEqual(
            injected_noise_levels[0][1], 0.2 * expected_margins[1]
        )
        self.assertEqual(len(injected_source_latents), 1)
        self.assertEqual(len(injected_source_latents[0]), 2)
        self.assertEqual(injected_source_latents[0][0][0], 2)
        self.assertEqual(injected_source_latents[0][1][0], 2)
        self.assertEqual(
            decoded_injected_source_latents,
            [[{"margin": 0.0, "cache": (1,)}, {"margin": 0.0, "cache": (2,)}]],
        )
        self.assertFalse(
            torch.equal(injected_source_latents[0][0][1], torch.ones((1, 1, 2)))
        )
        self.assertFalse(
            torch.equal(injected_source_latents[0][1][1], torch.ones((1, 1, 2)))
        )

    def test_noise_is_reversed_when_positive_direction_widens_margin(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        decoded_initial: list[list[dict[str, float]]] = []
        decoded_selected: list[list[dict[str, float]]] = []
        selected_latents: list[list[tuple[int, torch.Tensor]]] = []

        def step(token: torch.Tensor, current_cache: list[int]):
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            current_cache.append(len(current_cache))
            return torch.tensor([[[1.0, 0.0]]]), current_cache

        def decode(
            _token: torch.Tensor,
            _source_index: int,
            latent: torch.Tensor,
            _cache: list[int],
            _state: object,
        ) -> dict[str, float]:
            return {"margin": 0.8 if latent[..., 0].item() > 1.0 else 0.1}

        with patch(
            "recirculation.torch.randn_like",
            return_value=torch.tensor([[[1.0, 0.0]]]),
        ):
            recirculate(
                torch.tensor([[1]]),
                blocks=blocks,
                cache=[],
                step=step,
                rewind_one=lambda current_cache: current_cache[:-1],
                config=RecirculationConfig(
                    pairs=((2, 0),),
                    alpha=1.0,
                    noise_level_range=(0.4, 0.4),
                ),
                passes=2,
                post_margin_threshold=(0.0, 1.0),
                decode_injected_source_latent=decode,
                initial_decoded_noise_source_latents=decoded_initial,
                decoded_injected_source_latents=decoded_selected,
                injected_source_latents=selected_latents,
                capture_cached_token=lambda current_cache: current_cache[-1],
                restore_cached_token=lambda current_cache, cached_token: None,
            )

        self.assertEqual(decoded_initial, [[{"margin": 0.8}]])
        self.assertEqual(decoded_selected, [[{"margin": 0.1}]])
        self.assertLess(selected_latents[0][0][1][..., 0].item(), 1.0)

    def test_finalizes_cache_after_each_token(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        finalized: list[tuple[int, ...]] = []

        def step(token: torch.Tensor, current_cache: list[int]):
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            current_cache.append(int(token.item()))
            return torch.tensor([[[0.0, 1.0]]]), current_cache

        def finalize_token_cache(current_cache: list[int]) -> list[int]:
            finalized.append(tuple(current_cache))
            return current_cache

        recirculate(
            torch.tensor([[1, 2]]),
            blocks=blocks,
            cache=cache,
            step=step,
            rewind_one=lambda current_cache: current_cache,
            config=RecirculationConfig(pairs=((2, 0),), alpha=0.5),
            passes=1,
            finalize_token_cache=finalize_token_cache,
        )

        self.assertEqual(finalized, [(1,), (1, 2)])

    def test_single_pass_ignores_all_gates(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        logits = torch.tensor([[[0.0, 3.0, 1.0]]])
        recirculated_flags: list[bool] = []
        rejected_flags: list[bool] = []
        call_count = 0

        def step(token: torch.Tensor, current_cache: list[int]):
            nonlocal call_count
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            call_count += 1
            current_cache.append(call_count)
            return logits, current_cache

        result, final_cache = recirculate(
            torch.tensor([[1]]),
            blocks=blocks,
            cache=cache,
            step=step,
            rewind_one=lambda current_cache: current_cache,
            config=RecirculationConfig(pairs=((2, 0),), alpha=0.5),
            passes=1,
            condition_thresholds=[2.0],
            pre_margin_threshold=-1.0,
            post_margin_threshold=(0.0, 0.0),
            cosine_reject=1.0,
            recirculated_flags=recirculated_flags,
            rejected_flags=rejected_flags,
        )

        torch.testing.assert_close(result, logits)
        self.assertEqual(final_cache, [1])
        self.assertEqual(call_count, 1)
        self.assertEqual(recirculated_flags, [False])
        self.assertEqual(rejected_flags, [False])

    def test_disallowed_recirculation_runs_only_first_pass(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        call_count = 0

        def step(token: torch.Tensor, current_cache: list[int]):
            nonlocal call_count
            call_count += 1
            current_cache.append(call_count)
            return torch.tensor([[[0.0, 1.0]]]), current_cache

        recirculate(
            torch.tensor([[1]]),
            blocks=blocks,
            cache=cache,
            step=step,
            rewind_one=lambda current_cache: current_cache,
            config=RecirculationConfig(pairs=((2, 0),), alpha=0.5),
            passes=3,
            recirculation_allowed=False,
            capture_cached_token=lambda current_cache: current_cache[-1],
            restore_cached_token=lambda current_cache, cached_token: None,
        )

        self.assertEqual(call_count, 1)

    def test_same_top1_check_uses_final_pass(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        pass_logits = (
            torch.tensor([[[0.0, 3.0, 1.0]]]),
            torch.tensor([[[1.0, 4.0, 2.0]]]),
            torch.tensor([[[5.0, 0.0, 1.0]]]),
        )
        call_count = 0
        final_pass_same_top1_flags: list[bool] = []

        def step(token: torch.Tensor, current_cache: list[int]):
            nonlocal call_count
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            current_cache.append(call_count + 1)
            logits = pass_logits[call_count]
            call_count += 1
            return logits, current_cache

        def rewind_one(current_cache: list[int]) -> list[int]:
            current_cache.pop()
            return current_cache

        def restore_cached_token(
            current_cache: list[int], cached_token: int
        ) -> None:
            current_cache[-1] = cached_token

        logits, final_cache = recirculate(
            torch.tensor([[1]]),
            blocks=blocks,
            cache=cache,
            step=step,
            rewind_one=rewind_one,
            config=RecirculationConfig(pairs=((2, 0),), alpha=0.5),
            passes=3,
            final_pass_same_top1_flags=final_pass_same_top1_flags,
            capture_cached_token=lambda current_cache: current_cache[-1],
            restore_cached_token=restore_cached_token,
        )

        torch.testing.assert_close(logits, pass_logits[2])
        self.assertEqual(final_cache, [3])
        self.assertEqual(call_count, 3)
        self.assertEqual(final_pass_same_top1_flags, [False])

    def test_same_top1_on_final_pass_restores_p1_cache(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        pass_logits = (
            torch.tensor([[[0.0, 3.0, 1.0]]]),
            torch.tensor([[[5.0, 0.0, 1.0]]]),
            torch.tensor([[[1.0, 4.0, 2.0]]]),
        )
        call_count = 0
        final_pass_same_top1_flags: list[bool] = []

        def step(token: torch.Tensor, current_cache: list[int]):
            nonlocal call_count
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            current_cache.append(call_count + 1)
            logits = pass_logits[call_count]
            call_count += 1
            return logits, current_cache

        def rewind_one(current_cache: list[int]) -> list[int]:
            current_cache.pop()
            return current_cache

        def restore_cached_token(
            current_cache: list[int], cached_token: int
        ) -> None:
            current_cache[-1] = cached_token

        logits, final_cache = recirculate(
            torch.tensor([[1]]),
            blocks=blocks,
            cache=cache,
            step=step,
            rewind_one=rewind_one,
            config=RecirculationConfig(pairs=((2, 0),), alpha=0.5),
            passes=3,
            final_pass_same_top1_flags=final_pass_same_top1_flags,
            capture_cached_token=lambda current_cache: current_cache[-1],
            restore_cached_token=restore_cached_token,
        )

        torch.testing.assert_close(logits, pass_logits[2])
        self.assertEqual(final_cache, [1])
        self.assertEqual(call_count, 3)
        self.assertEqual(final_pass_same_top1_flags, [True])

    def test_rejection_gates_use_final_pass(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        pass_logits = (
            torch.tensor([[[3.0, 2.99, 0.0]]]),
            torch.tensor([[[0.0, 0.1, 0.2]]]),
            torch.tensor([[[2.8, 3.0, 0.0]]]),
        )
        call_count = 0
        rejected_flags: list[bool] = []
        rejection_reasons: list[tuple[str, ...]] = []
        final_pass_cosine_similarities: list[float | None] = []

        def step(token: torch.Tensor, current_cache: list[int]):
            nonlocal call_count
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            current_cache.append(call_count + 1)
            logits = pass_logits[call_count]
            call_count += 1
            return logits, current_cache

        def rewind_one(current_cache: list[int]) -> list[int]:
            current_cache.pop()
            return current_cache

        def restore_cached_token(
            current_cache: list[int], cached_token: int
        ) -> None:
            current_cache[-1] = cached_token

        logits, final_cache = recirculate(
            torch.tensor([[1]]),
            blocks=blocks,
            cache=cache,
            step=step,
            rewind_one=rewind_one,
            config=RecirculationConfig(pairs=((2, 0),), alpha=0.5),
            passes=3,
            post_margin_threshold=(0.05, 0.05),
            cosine_reject=0.8,
            rejected_flags=rejected_flags,
            rejection_reasons=rejection_reasons,
            final_pass_cosine_similarities=final_pass_cosine_similarities,
            capture_cached_token=lambda current_cache: current_cache[-1],
            restore_cached_token=restore_cached_token,
        )

        torch.testing.assert_close(logits, pass_logits[2])
        self.assertEqual(final_cache, [3])
        self.assertEqual(call_count, 3)
        self.assertEqual(rejected_flags, [False])
        self.assertEqual(rejection_reasons, [()])
        self.assertGreater(final_pass_cosine_similarities[0], 0.8)

    def test_post_margin_accepts_at_maximum_boundary(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        pass_logits = (
            torch.tensor([[[0.01, 0.0, 0.0]]]),
            torch.tensor([[[0.1, 0.0, 0.0]]]),
            torch.tensor([[[3.0, 1.0, 0.0]]]),
            torch.tensor([[[5.0, 0.0, 0.0]]]),
        )
        call_count = 0
        rejected_flags: list[bool] = []
        rejection_reasons: list[tuple[str, ...]] = []

        def step(token: torch.Tensor, current_cache: list[int]):
            nonlocal call_count
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            current_cache.append(call_count + 1)
            logits = pass_logits[call_count]
            call_count += 1
            return logits, current_cache

        def rewind_one(current_cache: list[int]) -> list[int]:
            current_cache.pop()
            return current_cache

        def restore_cached_token(
            current_cache: list[int], cached_token: int
        ) -> None:
            current_cache[-1] = cached_token

        logits, final_cache = recirculate(
            torch.tensor([[1, 2]]),
            blocks=blocks,
            cache=cache,
            step=step,
            rewind_one=rewind_one,
            config=RecirculationConfig(pairs=((2, 0),), alpha=0.5),
            passes=2,
            post_margin_threshold=(0.05, 0.05),
            rejected_flags=rejected_flags,
            rejection_reasons=rejection_reasons,
            capture_cached_token=lambda current_cache: current_cache[-1],
            restore_cached_token=restore_cached_token,
        )

        torch.testing.assert_close(
            logits, torch.cat((pass_logits[0], pass_logits[3]), dim=1)
        )
        self.assertEqual(final_cache, [1, 3])
        self.assertEqual(rejected_flags, [True, False])
        self.assertEqual(
            rejection_reasons,
            [("post-margin-min",), ()],
        )

    def test_post_margin_ratio_rejects_and_restores_p1(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        pass_logits = (
            torch.tensor([[[0.0, 1.0, 0.0]]]),
            torch.tensor([[[0.0, 0.1, 0.0]]]),
        )
        call_count = 0
        rejection_reasons: list[tuple[str, ...]] = []

        def step(token: torch.Tensor, current_cache: list[int]):
            nonlocal call_count
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            current_cache.append(call_count + 1)
            logits = pass_logits[call_count]
            call_count += 1
            return logits, current_cache

        def rewind_one(current_cache: list[int]) -> list[int]:
            current_cache.pop()
            return current_cache

        result, final_cache = recirculate(
            torch.tensor([[1]]),
            blocks=blocks,
            cache=cache,
            step=step,
            rewind_one=rewind_one,
            config=RecirculationConfig(pairs=((2, 0),), alpha=0.5),
            passes=2,
            post_margin_threshold=(0.01, 0.5),
            post_margin_ratio_threshold=0.5,
            rejection_reasons=rejection_reasons,
            capture_cached_token=lambda current_cache: current_cache[-1],
            restore_cached_token=lambda current_cache, cached: current_cache.__setitem__(
                -1, cached
            ),
        )

        torch.testing.assert_close(result, pass_logits[0])
        self.assertEqual(final_cache, [1])
        self.assertEqual(rejection_reasons, [("post-margin-ratio",)])

    def test_middle_post_margin_accepts_when_ratio_passes(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        pass_logits = (
            torch.tensor([[[0.0, 1.0, 0.0]]]),
            torch.tensor([[[0.8, 0.0, 0.0]]]),
        )
        call_count = 0
        rejection_reasons: list[tuple[str, ...]] = []

        def step(token: torch.Tensor, current_cache: list[int]):
            nonlocal call_count
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            current_cache.append(call_count + 1)
            logits = pass_logits[call_count]
            call_count += 1
            return logits, current_cache

        def rewind_one(current_cache: list[int]) -> list[int]:
            current_cache.pop()
            return current_cache

        result, final_cache = recirculate(
            torch.tensor([[1]]),
            blocks=blocks,
            cache=cache,
            step=step,
            rewind_one=rewind_one,
            config=RecirculationConfig(pairs=((2, 0),), alpha=0.5),
            passes=2,
            post_margin_threshold=(0.1, 0.7),
            post_margin_ratio_threshold=0.5,
            rejection_reasons=rejection_reasons,
            capture_cached_token=lambda current_cache: current_cache[-1],
            restore_cached_token=lambda current_cache, cached: current_cache.__setitem__(
                -1, cached
            ),
        )

        torch.testing.assert_close(result, pass_logits[1])
        self.assertEqual(final_cache, [2])
        self.assertEqual(rejection_reasons, [()])

    def test_fixed_recirculation_stops_when_margin_narrows(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        pass_logits = (
            torch.tensor([[[0.2, 0.0, 0.0]]]),
            torch.tensor([[[0.1, 0.0, 0.0]]]),
        )
        call_count = 0

        def step(token: torch.Tensor, current_cache: list[int]):
            nonlocal call_count
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            current_cache.append(call_count + 1)
            logits = pass_logits[call_count]
            call_count += 1
            return logits, current_cache

        def rewind_one(current_cache: list[int]) -> list[int]:
            current_cache.pop()
            return current_cache

        logits, final_cache = recirculate(
            torch.tensor([[1]]),
            blocks=blocks,
            cache=cache,
            step=step,
            rewind_one=rewind_one,
            config=RecirculationConfig(pairs=((2, 0),), alpha=0.5),
            passes=3,
            capture_cached_token=lambda current_cache: current_cache[-1],
            restore_cached_token=lambda current_cache, cached: current_cache.__setitem__(
                -1, cached
            ),
        )

        torch.testing.assert_close(logits, pass_logits[1])
        self.assertEqual(final_cache, [1])
        self.assertEqual(call_count, 2)

    def test_adaptive_recirculation_stops_when_post_margin_reaches_min(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        pass_logits = (
            torch.tensor([[[0.1, 0.0, 0.0]]]),
            torch.tensor([[[0.0, 2.0, 0.0]]]),
        )
        call_count = 0
        margins: list[list[float]] = []
        recirculated_flags: list[bool] = []
        rejected_flags: list[bool] = []
        adaptive_recirculated_flags: list[bool] = []
        adaptive_rejected_flags: list[bool] = []
        adaptive_counts: list[int] = []

        def step(token: torch.Tensor, current_cache: list[int]):
            nonlocal call_count
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            current_cache.append(call_count + 1)
            logits = pass_logits[call_count]
            call_count += 1
            return logits, current_cache

        def rewind_one(current_cache: list[int]) -> list[int]:
            current_cache.pop()
            return current_cache

        logits, final_cache = recirculate(
            torch.tensor([[1]]),
            blocks=blocks,
            cache=cache,
            step=step,
            rewind_one=rewind_one,
            config=RecirculationConfig(pairs=((2, 0),), alpha=0.5),
            passes=1,
            post_margin_threshold=(0.2, 0.2),
            adaptive_recirculation=3,
            pass_probability_margins=margins,
            recirculated_flags=recirculated_flags,
            rejected_flags=rejected_flags,
            adaptive_recirculated_flags=adaptive_recirculated_flags,
            adaptive_rejected_flags=adaptive_rejected_flags,
            adaptive_recirculation_counts=adaptive_counts,
            capture_cached_token=lambda current_cache: current_cache[-1],
            restore_cached_token=lambda current_cache, cached: current_cache.__setitem__(
                -1, cached
            ),
        )

        torch.testing.assert_close(logits, pass_logits[1])
        self.assertEqual(final_cache, [2])
        self.assertEqual(call_count, 2)
        self.assertEqual(len(margins[0]), 2)
        self.assertEqual(recirculated_flags, [True])
        self.assertEqual(rejected_flags, [False])
        self.assertEqual(adaptive_recirculated_flags, [True])
        self.assertEqual(adaptive_rejected_flags, [False])
        self.assertEqual(adaptive_counts, [1])

    def test_adaptive_recirculation_rejects_narrowing_margin_on_final_pass(
        self,
    ) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        pass_logits = (
            torch.tensor([[[0.1, 0.0, 0.0]]]),
            torch.tensor([[[0.05, 0.0, 0.0]]]),
            torch.tensor([[[0.02, 0.0, 0.0]]]),
            torch.tensor([[[0.01, 0.0, 0.0]]]),
        )
        call_count = 0
        margins: list[list[float]] = []
        adaptive_counts: list[int] = []
        rejection_reasons: list[tuple[str, ...]] = []

        def step(token: torch.Tensor, current_cache: list[int]):
            nonlocal call_count
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            current_cache.append(call_count + 1)
            logits = pass_logits[call_count]
            call_count += 1
            return logits, current_cache

        def rewind_one(current_cache: list[int]) -> list[int]:
            current_cache.pop()
            return current_cache

        logits, final_cache = recirculate(
            torch.tensor([[1]]),
            blocks=blocks,
            cache=cache,
            step=step,
            rewind_one=rewind_one,
            config=RecirculationConfig(pairs=((2, 0),), alpha=0.5),
            passes=1,
            post_margin_threshold=(0.2, 0.2),
            adaptive_recirculation=3,
            pass_probability_margins=margins,
            adaptive_recirculation_counts=adaptive_counts,
            rejection_reasons=rejection_reasons,
            capture_cached_token=lambda current_cache: current_cache[-1],
            restore_cached_token=lambda current_cache, cached: current_cache.__setitem__(
                -1, cached
            ),
        )

        torch.testing.assert_close(logits, pass_logits[0])
        self.assertEqual(final_cache, [1])
        self.assertEqual(call_count, 4)
        self.assertEqual(len(margins[0]), 4)
        self.assertEqual(adaptive_counts, [3])
        self.assertEqual(
            rejection_reasons,
            [("margin-narrowed",)],
        )

    def test_adaptive_recirculation_skips_replay_when_p1_margin_is_sufficient(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        logits = torch.tensor([[[3.0, 0.0, 0.0]]])
        call_count = 0
        adaptive_recirculated_flags: list[bool] = []
        adaptive_counts: list[int] = []

        def step(token: torch.Tensor, current_cache: list[int]):
            nonlocal call_count
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            call_count += 1
            current_cache.append(call_count)
            return logits, current_cache

        result, final_cache = recirculate(
            torch.tensor([[1]]),
            blocks=blocks,
            cache=cache,
            step=step,
            rewind_one=lambda current_cache: current_cache,
            config=RecirculationConfig(pairs=((2, 0),), alpha=0.5),
            passes=1,
            post_margin_threshold=(0.2, 0.2),
            adaptive_recirculation=3,
            adaptive_recirculated_flags=adaptive_recirculated_flags,
            adaptive_recirculation_counts=adaptive_counts,
            capture_cached_token=lambda current_cache: current_cache[-1],
            restore_cached_token=lambda current_cache, cached: current_cache.__setitem__(
                -1, cached
            ),
        )

        torch.testing.assert_close(result, logits)
        self.assertEqual(final_cache, [1])
        self.assertEqual(call_count, 1)
        self.assertEqual(adaptive_recirculated_flags, [False])
        self.assertEqual(adaptive_counts, [0])

    def test_adaptive_recirculation_rejects_after_retry_limit(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        logits = torch.tensor([[[0.1, 0.0, 0.0]]])
        call_count = 0
        rejection_reasons: list[tuple[str, ...]] = []
        adaptive_rejected_flags: list[bool] = []
        adaptive_counts: list[int] = []

        def step(token: torch.Tensor, current_cache: list[int]):
            nonlocal call_count
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            call_count += 1
            current_cache.append(call_count)
            return logits, current_cache

        def rewind_one(current_cache: list[int]) -> list[int]:
            current_cache.pop()
            return current_cache

        result, final_cache = recirculate(
            torch.tensor([[1]]),
            blocks=blocks,
            cache=cache,
            step=step,
            rewind_one=rewind_one,
            config=RecirculationConfig(pairs=((2, 0),), alpha=0.5),
            passes=1,
            post_margin_threshold=(0.2, 0.2),
            adaptive_recirculation=2,
            rejection_reasons=rejection_reasons,
            adaptive_rejected_flags=adaptive_rejected_flags,
            adaptive_recirculation_counts=adaptive_counts,
            capture_cached_token=lambda current_cache: current_cache[-1],
            restore_cached_token=lambda current_cache, cached: current_cache.__setitem__(
                -1, cached
            ),
        )

        torch.testing.assert_close(result, logits)
        self.assertEqual(final_cache, [1])
        self.assertEqual(call_count, 3)
        self.assertEqual(rejection_reasons, [("post-margin-min",)])
        self.assertEqual(adaptive_rejected_flags, [True])
        self.assertEqual(adaptive_counts, [2])

    def test_one_pass_adaptive_recirculation_honors_pre_gate(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        logits = torch.tensor([[[0.1, 0.0, 0.0]]])
        call_count = 0
        adaptive_counts: list[int] = []

        def step(token: torch.Tensor, current_cache: list[int]):
            nonlocal call_count
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            call_count += 1
            current_cache.append(call_count)
            return logits, current_cache

        result, final_cache = recirculate(
            torch.tensor([[1]]),
            blocks=blocks,
            cache=cache,
            step=step,
            rewind_one=lambda current_cache: current_cache,
            config=RecirculationConfig(pairs=((2, 0),), alpha=0.5),
            passes=1,
            pre_margin_threshold=-1.0,
            post_margin_threshold=(0.2, 0.2),
            adaptive_recirculation=2,
            adaptive_recirculation_counts=adaptive_counts,
            capture_cached_token=lambda current_cache: current_cache[-1],
            restore_cached_token=lambda current_cache, cached: current_cache.__setitem__(
                -1, cached
            ),
        )

        torch.testing.assert_close(result, logits)
        self.assertEqual(final_cache, [1])
        self.assertEqual(call_count, 1)
        self.assertEqual(adaptive_counts, [0])

    def test_adaptive_recirculation_does_not_require_post_margin(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        pass_logits = (
            torch.tensor([[[0.1, 0.0, 0.0]]]),
            torch.tensor([[[0.05, 0.0, 0.0]]]),
        )
        call_count = 0
        rejection_reasons: list[tuple[str, ...]] = []

        def step(token: torch.Tensor, current_cache: list[int]):
            nonlocal call_count
            hidden = torch.ones((1, 1, 2))
            for block in blocks:
                hidden = block(hidden)
            current_cache.append(call_count + 1)
            logits = pass_logits[call_count]
            call_count += 1
            return logits, current_cache

        def rewind_one(current_cache: list[int]) -> list[int]:
            current_cache.pop()
            return current_cache

        result, final_cache = recirculate(
            torch.tensor([[1]]),
            blocks=blocks,
            cache=cache,
            step=step,
            rewind_one=rewind_one,
            config=RecirculationConfig(pairs=((2, 0),), alpha=0.5),
            passes=1,
            adaptive_recirculation=2,
            rejection_reasons=rejection_reasons,
            capture_cached_token=lambda current_cache: current_cache[-1],
            restore_cached_token=lambda current_cache, cached: current_cache.__setitem__(
                -1, cached
            ),
        )

        torch.testing.assert_close(result, pass_logits[1])
        self.assertEqual(final_cache, [1])
        self.assertEqual(call_count, 2)
        self.assertEqual(rejection_reasons, [()])


class DynamicCacheTokenCaptureTest(unittest.TestCase):
    def test_rewind_restores_linear_attention_state(self) -> None:
        from infer import (
            capture_dynamic_cache_rewind_state,
            restore_dynamic_cache_rewind_state,
            rewind_dynamic_cache,
        )
        from transformers.cache_utils import DynamicCache, LinearAttentionLayer

        layer = LinearAttentionLayer(number_of_states=1)
        cache = DynamicCache()
        cache.layers = [layer]
        cache.activate_past_recording()
        layer.update_conv_state(
            torch.ones(1, 4, 4), state_idx=0, conv_kernel_size=4
        )
        layer.update_recurrent_state(torch.ones(1, 4, 16), state_idx=0)
        rewind_state = capture_dynamic_cache_rewind_state(cache)

        layer.update_conv_state(torch.full((1, 4, 1), 2.0), state_idx=0)
        layer.update_recurrent_state(torch.full((1, 4, 16), 2.0), state_idx=0)
        rewind_dynamic_cache(cache)
        restore_dynamic_cache_rewind_state(cache, rewind_state)

        torch.testing.assert_close(layer.conv_states[0], torch.ones(1, 4, 4))
        torch.testing.assert_close(
            layer.recurrent_states[0], torch.ones(1, 4, 16)
        )

    def test_capture_and_restore_with_linear_and_dynamic_layers(self) -> None:
        from infer import (
            capture_dynamic_cache_token,
            finalize_dynamic_cache_token,
            restore_dynamic_cache_token,
        )
        from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer

        l1 = DynamicLayer()
        l2 = LinearAttentionLayer(number_of_states=1)
        cache = DynamicCache()
        cache.layers = [l1, l2]
        cache.activate_past_recording()

        # P1 may retain the full prompt while recording history.
        l1.update(torch.ones(1, 2, 1, 8), torch.ones(1, 2, 1, 8))
        l2.update_conv_state(
            torch.ones(1, 4, 49), state_idx=0, conv_kernel_size=4
        )
        l2.update_recurrent_state(torch.ones(1, 4, 16), state_idx=0)

        captured = capture_dynamic_cache_token(cache)

        # A replay starts from the compact kernel window and adds one position.
        l1.keys[..., -1:, :] += 5.0
        l2.conv_states[0] = torch.full((1, 4, 5), 2.0)
        l2.recurrent_states[0] += 10.0

        restore_dynamic_cache_token(cache, captured)

        torch.testing.assert_close(l1.keys[..., -1:, :], torch.ones(1, 2, 1, 8))
        torch.testing.assert_close(l2.conv_states[0], torch.ones(1, 4, 49))
        torch.testing.assert_close(l2.recurrent_states[0], torch.ones(1, 4, 16))

        finalize_dynamic_cache_token(cache)

        torch.testing.assert_close(l2.conv_states[0], torch.ones(1, 4, 4))


if __name__ == "__main__":
    unittest.main()