import unittest

import torch
from torch import nn

from recirculation import RecirculationConfig, recirculate


class RecirculationCacheTest(unittest.TestCase):
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
            post_margin_threshold=0.0,
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
            post_margin_threshold=0.05,
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

    def test_post_margin_has_no_maximum(self) -> None:
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
            post_margin_threshold=0.05,
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
            post_margin_threshold=0.2,
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

    def test_adaptive_recirculation_rejects_narrowing_margin_after_retry_limit(
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
            post_margin_threshold=0.2,
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
        self.assertEqual(rejection_reasons, [("post-margin-min",)])

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
            post_margin_threshold=0.2,
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
            post_margin_threshold=0.2,
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
            post_margin_threshold=0.2,
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