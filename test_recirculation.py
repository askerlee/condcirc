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
            margin_threshold_p1=-1.0,
            margin_threshold_p2=1.0,
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
            torch.tensor([[[3.0, 2.0, 0.0]]]),
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
            margin_threshold_p2=0.05,
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


if __name__ == "__main__":
    unittest.main()