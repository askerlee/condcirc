import unittest

import torch
from torch import nn

from recirculation import RecirculationConfig, recirculate


class RecirculationCacheTest(unittest.TestCase):
    def test_same_top1_keeps_p2_logits_and_restores_p1_cache(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        cache: list[int] = []
        pass_logits = (
            torch.tensor([[[0.0, 3.0, 1.0]]]),
            torch.tensor([[[1.0, 4.0, 2.0]]]),
            torch.tensor([[[5.0, 0.0, 1.0]]]),
        )
        call_count = 0
        p2_same_top1_flags: list[bool] = []

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
            p2_same_top1_flags=p2_same_top1_flags,
            capture_cached_token=lambda current_cache: current_cache[-1],
            restore_cached_token=restore_cached_token,
        )

        torch.testing.assert_close(logits, pass_logits[1])
        self.assertEqual(final_cache, [1])
        self.assertEqual(call_count, 2)
        self.assertEqual(p2_same_top1_flags, [True])


if __name__ == "__main__":
    unittest.main()