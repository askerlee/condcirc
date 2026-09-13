import unittest

import torch
from torch import nn

from recirculation import RecirculationConfig, recirculate


class RecirculationCacheTest(unittest.TestCase):
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
            post_margin_thresholds=(0.0, 1.0),
            top1_boost_threshold=-1.0,
            top1_prob_threshold=-1.0,
            cosine_reject=1.0,
            rank_top_k=1,
            recirculated_flags=recirculated_flags,
            rejected_flags=rejected_flags,
        )

        torch.testing.assert_close(result, logits)
        self.assertEqual(final_cache, [1])
        self.assertEqual(call_count, 1)
        self.assertEqual(recirculated_flags, [False])
        self.assertEqual(rejected_flags, [False])

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

    def test_all_rejection_gates_use_final_pass(self) -> None:
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
        final_pass_top1_boosts: list[float | None] = []

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
            post_margin_thresholds=(0.05, 1.0),
            top1_boost_threshold=0.3,
            cosine_reject=0.8,
            rank_top_k=2,
            rejected_flags=rejected_flags,
            rejection_reasons=rejection_reasons,
            final_pass_cosine_similarities=final_pass_cosine_similarities,
            final_pass_top1_boosts=final_pass_top1_boosts,
            capture_cached_token=lambda current_cache: current_cache[-1],
            restore_cached_token=restore_cached_token,
        )

        torch.testing.assert_close(logits, pass_logits[2])
        self.assertEqual(final_cache, [3])
        self.assertEqual(call_count, 3)
        self.assertEqual(rejected_flags, [False])
        self.assertEqual(rejection_reasons, [()])
        self.assertGreater(final_pass_cosine_similarities[0], 0.8)
        self.assertLess(final_pass_top1_boosts[0], 0.3)

    def test_rejects_final_pass_outside_post_margin_range(self) -> None:
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
            post_margin_thresholds=(0.05, 0.5),
            rejected_flags=rejected_flags,
            rejection_reasons=rejection_reasons,
            capture_cached_token=lambda current_cache: current_cache[-1],
            restore_cached_token=restore_cached_token,
        )

        torch.testing.assert_close(
            logits, torch.cat((pass_logits[0], pass_logits[2]), dim=1)
        )
        self.assertEqual(final_cache, [1, 3])
        self.assertEqual(rejected_flags, [True, True])
        self.assertEqual(
            rejection_reasons,
            [("post-margin-min",), ("post-margin-max",)],
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
            post_margin_thresholds=(0.2, 1.0),
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

    def test_adaptive_recirculation_stops_when_margin_narrows(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        pass_logits = (
            torch.tensor([[[0.1, 0.0, 0.0]]]),
            torch.tensor([[[0.05, 0.0, 0.0]]]),
        )
        call_count = 0
        margins: list[list[float]] = []
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
            post_margin_thresholds=(0.2, 1.0),
            adaptive_recirculation=3,
            pass_probability_margins=margins,
            adaptive_recirculation_counts=adaptive_counts,
            capture_cached_token=lambda current_cache: current_cache[-1],
            restore_cached_token=lambda current_cache, cached: current_cache.__setitem__(
                -1, cached
            ),
        )

        torch.testing.assert_close(logits, pass_logits[1])
        self.assertEqual(final_cache, [1])
        self.assertEqual(call_count, 2)
        self.assertEqual(len(margins[0]), 2)
        self.assertEqual(adaptive_counts, [1])

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
            post_margin_thresholds=(0.2, 1.0),
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
            post_margin_thresholds=(0.2, 1.0),
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
            post_margin_thresholds=(0.2, 1.0),
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

    def test_one_pass_adaptive_recirculation_honors_final_gate(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        pass_logits = (
            torch.tensor([[[0.1, 0.0, 0.0]]]),
            torch.tensor([[[0.0, 2.0, 0.0]]]),
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
            post_margin_thresholds=(0.2, 1.0),
            adaptive_recirculation=2,
            top1_boost_threshold=-1.0,
            rejection_reasons=rejection_reasons,
            capture_cached_token=lambda current_cache: current_cache[-1],
            restore_cached_token=lambda current_cache, cached: current_cache.__setitem__(
                -1, cached
            ),
        )

        torch.testing.assert_close(result, pass_logits[0])
        self.assertEqual(final_cache, [1])
        self.assertEqual(call_count, 2)
        self.assertEqual(rejection_reasons, [("top1-boost",)])


if __name__ == "__main__":
    unittest.main()