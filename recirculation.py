"""Minimal reference for multi-pass, training-free recirculation.
   Originally implemented by Benhao Huang:
   https://gist.github.com/huskydoge/1ff29693e2172226ec26081f208b19d6

In source-to-destination mode, each configured ``(source, destination)`` pair
is recirculated for every token (with three passes):
    1. Run a normal cached pass; return its logits and save residuals h_d, h_s.
  2. Rewind the KV cache by one position.
  3. Run the token again, replacing the output of destination block d with
     beta * h_d + alpha * (||h_d|| / ||h_s||) * h_s.
    4. Capture the new residuals, rewind, and repeat the recirculation once more.
     5. Optionally reject P2 when its next-token distribution is insufficiently
         similar to P1, restoring P1 logits and KV cache; otherwise continue and
         commit the final pass.

When expected-embedding subtraction is enabled, each replay subtracts the
top-K expected next-token embedding from h_s before norm matching and mixing.

In layerwise mode, each block from destination through source is run ``passes``
times in place, feeding each pass output into the next pass after matching the
original input norm. Each extra pass replaces that layer's previous KV entry,
so the cache advances once per token.

``step(token, cache)``, ``rewind_one(cache)``, and
``rewind_layer(cache, layer_index)`` are model-specific adapters. Block indices
are zero-based outputs, so the mixture is injected as the input to block d + 1.
This is the serial reference, not the paper's serving pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RecirculationConfig:
    pairs: tuple[tuple[int, int], ...]
    alpha: float
    beta: float | None = None  # None selects the convex mix: beta = 1 - alpha.
    eps: float = 1e-8
    mode: Literal["source", "layerwise"] = "source"
    act_sim_as_alpha: bool = False
    act_sim_min_max: tuple[float, float] | None = None


@dataclass
class SimilarityStats:
    values: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float] | None:
        if not self.values:
            return None
        ordered = sorted(self.values)
        count = len(ordered)
        mean = sum(ordered) / count
        variance = sum((value - mean) ** 2 for value in ordered) / count

        def quantile(fraction: float) -> float:
            return ordered[min(count - 1, int(fraction * count))]

        return {
            "count": count,
            "mean": mean,
            "std": variance**0.5,
            "min": ordered[0],
            "p25": quantile(0.25),
            "median": quantile(0.5),
            "p75": quantile(0.75),
            "max": ordered[-1],
        }


@dataclass
class AdjacentLayerSimilarityStats:
    """Per-token cosine similarity between every pair of adjacent blocks."""

    per_layer: list[SimilarityStats] = field(default_factory=list)

    def ensure_layers(self, count: int) -> None:
        while len(self.per_layer) < count:
            self.per_layer.append(SimilarityStats())

    def summaries(self) -> list[dict[str, float] | None]:
        return [stats.summary() for stats in self.per_layer]


@dataclass
class MagnitudeDiffStats:
    fraction_sum: float = 0.0
    count: int = 0

    @property
    def mean(self) -> float | None:
        return self.fraction_sum / self.count if self.count else None

def _residual(output: Any) -> Tensor:
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, Tensor):
        raise TypeError("A transformer block must return its residual stream first.")
    return hidden


def _top1_top2_probability_margin(logits: Tensor) -> float:
    probabilities = torch.softmax(logits[:, -1, :].float(), dim=-1)
    top_two = torch.topk(probabilities, k=2, dim=-1).values
    return float((top_two[..., 0] - top_two[..., 1]).item())


def _top1_probability(logits: Tensor) -> float:
    probabilities = torch.softmax(logits[:, -1, :].float(), dim=-1)
    return float(probabilities.max(dim=-1).values.item())


def _distribution_kl_divergence(
    teacher_logits: Tensor, student_logits: Tensor
) -> float:
    teacher_log_probabilities = torch.log_softmax(
        teacher_logits[:, -1, :].float(), dim=-1
    )
    student_log_probabilities = torch.log_softmax(
        student_logits[:, -1, :].float(), dim=-1
    )
    teacher_probabilities = teacher_log_probabilities.exp()
    divergence = (
        teacher_probabilities
        * (teacher_log_probabilities - student_log_probabilities)
    ).sum(dim=-1)
    return float(divergence.mean().item())


class _Hooks:
    def __init__(
        self,
        blocks: Sequence[nn.Module],
        cfg: RecirculationConfig,
        magnitude_diff_stats: MagnitudeDiffStats,
        adjacent_layer_stats: AdjacentLayerSimilarityStats | None = None,
    ) -> None:
        if not cfg.pairs:
            raise ValueError("At least one source/destination pair is required.")
        if len({destination for _source, destination in cfg.pairs}) != len(cfg.pairs):
            raise ValueError("Each source/destination pair must have a unique destination.")
        if any(
            not 0 <= destination < source < len(blocks)
            for source, destination in cfg.pairs
        ):
            raise ValueError(
                "Expected 0 <= destination < source < number of blocks for every pair."
            )
        self.cfg = cfg
        self.mode = "off"
        self.destinations = {destination for _source, destination in cfg.pairs}
        self.sources = {source for source, _destination in cfg.pairs}
        self.residuals: dict[int, Tensor] = {}
        self.injection_sources: dict[int, Tensor] = {}
        self.injection_alphas: tuple[float, ...] | None = None
        self.expected_embedding: Tensor | None = None
        self.active_pairs = tuple(True for _pair in cfg.pairs)
        self.magnitude_diff_stats = magnitude_diff_stats
        self.adjacent_layer_stats = adjacent_layer_stats
        self.layer_residuals: dict[int, Tensor] = {}
        watched_layers = self.destinations | self.sources
        handles = [
            blocks[layer_index].register_forward_hook(self._save_residual(layer_index))
            for layer_index in watched_layers
        ]
        handles.extend(
            blocks[destination + 1].register_forward_pre_hook(
                self._inject(pair_index, source, destination)
            )
            for pair_index, (source, destination) in enumerate(cfg.pairs)
        )
        if adjacent_layer_stats is not None:
            adjacent_layer_stats.ensure_layers(len(blocks) - 1)
            handles.extend(
                block.register_forward_hook(self._save_layer(layer_index))
                for layer_index, block in enumerate(blocks)
            )
        self.handles = tuple(handles)

    def _save_layer(self, layer_index: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _inputs: tuple, output: Any) -> None:
            if self.mode == "capture":
                self.layer_residuals[layer_index] = (
                    _residual(output)[:, -1, :].detach().float()
                )

        return hook

    def record_adjacent_similarities(self) -> None:
        if self.adjacent_layer_stats is None:
            return
        for layer_index in range(len(self.adjacent_layer_stats.per_layer)):
            previous = self.layer_residuals.get(layer_index)
            current = self.layer_residuals.get(layer_index + 1)
            if previous is None or current is None:
                continue
            cosine = torch.nn.functional.cosine_similarity(
                previous, current.to(previous.device), dim=-1
            )
            self.adjacent_layer_stats.per_layer[layer_index].values.append(
                float(cosine.mean().item())
            )
        self.layer_residuals.clear()

    def _save_residual(self, layer_index: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _inputs: tuple, output: Any) -> None:
            if self.mode in ("capture", "inject"):
                residual = _residual(output).detach().clone()
                self.residuals[layer_index] = residual

        return hook

    def _inject(
        self, pair_index: int, source_index: int, destination_index: int
    ) -> Callable[..., tuple | None]:
        def hook(_module: nn.Module, inputs: tuple) -> tuple | None:
            if self.mode != "inject" or not self.active_pairs[pair_index]:
                return None
            try:
                destination = self.residuals[destination_index]
                source = self.injection_sources[source_index]
            except KeyError as error:
                raise RuntimeError(
                    "The first pass did not capture both residual streams for every pair."
                ) from error

            input_device = _residual(inputs).device
            destination = destination.to(device=input_device, dtype=torch.float32)
            source = source.to(device=input_device, dtype=torch.float32)
            source_norm_before = torch.linalg.vector_norm(
                source, dim=-1, keepdim=True
            )
            source *= torch.linalg.vector_norm(destination, dim=-1, keepdim=True) / (
                torch.linalg.vector_norm(source, dim=-1, keepdim=True).clamp_min(self.cfg.eps)
            )

            if self.expected_embedding is not None:
                expected = self.expected_embedding.to(
                    device=input_device, dtype=torch.float32
                )
                projection = (source * expected).sum(dim=-1, keepdim=True) / (
                    expected.square().sum(dim=-1, keepdim=True).clamp_min(self.cfg.eps)
                )
                source -= projection * expected

            alpha = (
                self.injection_alphas[pair_index]
                if self.injection_alphas is not None
                else self.cfg.alpha
            )
            beta = 1.0 - alpha if self.cfg.beta is None else self.cfg.beta
            if self.expected_embedding is not None:
                source_norm_after = torch.linalg.vector_norm(
                    source, dim=-1, keepdim=True
                )
                magnitude_diff_fraction = (
                    source_norm_before - source_norm_after
                ) / source_norm_before.clamp_min(self.cfg.eps)
                self.magnitude_diff_stats.fraction_sum += (
                    magnitude_diff_fraction.sum().item()
                )
                self.magnitude_diff_stats.count += magnitude_diff_fraction.numel()

            mixed = (beta * destination + alpha * source).to(inputs[0].dtype)
            return (mixed, *inputs[1:])

        return hook

    def activation_similarities(self) -> tuple[float, ...]:
        similarities = []
        for source_index, destination_index in self.cfg.pairs:
            try:
                destination = self.residuals[destination_index][:, -1, :].float()
                source = self.residuals[source_index][:, -1, :].to(
                    destination.device, dtype=torch.float32
                )
            except KeyError as error:
                raise RuntimeError(
                    "The first pass did not capture both residual streams for every pair."
                ) from error
            cosine = torch.nn.functional.cosine_similarity(destination, source, dim=-1)
            similarities.append(float(cosine.mean().item()))
        return tuple(similarities)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class _LayerwiseHooks:
    def __init__(
        self,
        blocks: Sequence[nn.Module],
        config: RecirculationConfig,
        cache: Any,
        rewind_layer: Callable[[Any, int], Any],
        passes: int,
        select_expert_subset: Callable[[int], None] | None,
    ) -> None:
        if len(config.pairs) != 1:
            raise ValueError("Layerwise mode requires exactly one source/destination pair.")
        source, destination = config.pairs[0]
        if not 0 <= destination < source < len(blocks):
            raise ValueError("Expected 0 <= destination < source < number of blocks.")
        self.cache = cache
        self.rewind_layer = rewind_layer
        self.passes = passes
        self.eps = config.eps
        self.select_expert_subset = select_expert_subset
        self.replaying = False
        self.handles = tuple(
            block.register_forward_hook(
                self._repeat_layer(layer_index), with_kwargs=True
            )
            for layer_index, block in enumerate(
                blocks[destination : source + 1],
                start=destination,
            )
        )

    def _repeat_layer(self, layer_index: int) -> Callable[..., Any]:
        def repeat(
            module: nn.Module,
            inputs: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> Any:
            if self.replaying:
                return output

            original_input = _residual(inputs if inputs else kwargs["hidden_states"])
            original_norm = torch.linalg.vector_norm(
                original_input.to(dtype=torch.float32), dim=-1, keepdim=True
            )
            repeated_output = output
            for pass_index in range(1, self.passes):
                hidden = _residual(repeated_output)
                hidden_float = hidden.to(dtype=torch.float32)
                hidden = (
                    hidden_float
                    * original_norm
                    / torch.linalg.vector_norm(
                        hidden_float, dim=-1, keepdim=True
                    ).clamp_min(self.eps)
                ).to(dtype=hidden.dtype)
                if inputs:
                    repeated_inputs = (hidden, *inputs[1:])
                    repeated_kwargs = kwargs
                elif "hidden_states" in kwargs:
                    repeated_inputs = inputs
                    repeated_kwargs = {**kwargs, "hidden_states": hidden}
                else:
                    raise TypeError(
                        "A transformer block must receive its residual stream as "
                        "the first argument or as hidden_states."
                    )

                self.cache = self.rewind_layer(self.cache, layer_index)
                if self.select_expert_subset is not None:
                    self.select_expert_subset(pass_index)
                self.replaying = True
                try:
                    repeated_output = module(*repeated_inputs, **repeated_kwargs)
                finally:
                    self.replaying = False

            if self.select_expert_subset is not None:
                self.select_expert_subset(0)
            return repeated_output

        return repeat

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def recirculate(
    input_ids: Tensor,
    *,
    blocks: Sequence[nn.Module],
    cache: Any,
    step: Callable[[Tensor, Any], tuple[Tensor, Any]],
    rewind_one: Callable[[Any], Any],
    config: RecirculationConfig,
    select_expert_subset: Callable[[int], None] | None = None,
    expected_embedding: Callable[[Tensor], Tensor] | None = None,
    magnitude_diff_stats: MagnitudeDiffStats | None = None,
    similarity_stats: SimilarityStats | None = None,
    adjacent_layer_stats: AdjacentLayerSimilarityStats | None = None,
    passes: int = 3,
    rewind_layer: Callable[[Any, int], Any] | None = None,
    condition_thresholds: Sequence[float] | None = None,
    margin_threshold: float | None = None,
    top1_prob_threshold: float | None = None,
    kl_reject: float | None = None,
    gating_pair_index: int = 0,
    first_pass_logits: list[Tensor] | None = None,
    first_pass_similarities: list[tuple[float, ...]] | None = None,
    pass_probability_margins: list[list[float]] | None = None,
    actual_alphas: list[tuple[float, ...] | None] | None = None,
    recirculated_flags: list[bool] | None = None,
    rejected_flags: list[bool] | None = None,
    kl_divergences: list[float | None] | None = None,
    capture_cached_token: Callable[[Any], Any] | None = None,
    average_cached_token: Callable[[Any, Sequence[Any]], None] | None = None,
    restore_cached_token: Callable[[Any, Any], None] | None = None,
) -> tuple[Tensor, Any]:
    """Run source-to-destination recirculation or layerwise repeated passes."""

    if passes < 1:
        raise ValueError("passes must be at least 1.")
    if average_cached_token is not None and capture_cached_token is None:
        raise ValueError(
            "average_cached_token requires capture_cached_token."
        )
    if kl_reject is not None and (
        capture_cached_token is None or restore_cached_token is None
    ):
        raise ValueError(
            "kl_reject requires capture_cached_token and restore_cached_token."
        )

    if config.mode == "layerwise":
        if condition_thresholds is not None:
            raise ValueError(
                "Conditional recirculation currently requires --mode source."
            )
        if margin_threshold is not None:
            raise ValueError(
                "Margin-gated recirculation currently requires --mode source."
            )
        if top1_prob_threshold is not None:
            raise ValueError(
                "Top1-probability-gated recirculation currently requires --mode source."
            )
        if kl_reject is not None:
            raise ValueError("P2 trust rejection currently requires --mode source.")
        if similarity_stats is not None:
            raise ValueError(
                "Source/destination similarity stats require --mode source."
            )
        if adjacent_layer_stats is not None:
            raise ValueError(
                "Adjacent-layer similarity stats require --mode source."
            )
        if rewind_layer is None:
            raise ValueError("Layerwise mode requires rewind_layer.")
        logits = []
        hooks = _LayerwiseHooks(
            blocks,
            config,
            cache,
            rewind_layer,
            passes,
            select_expert_subset,
        )
        try:
            for position in range(input_ids.shape[1]):
                token = input_ids[:, position : position + 1]
                if select_expert_subset is not None:
                    select_expert_subset(0)
                token_logits, cache = step(token, hooks.cache)
                hooks.cache = cache
                if pass_probability_margins is not None:
                    pass_probability_margins.append(
                        [_top1_top2_probability_margin(token_logits)]
                    )
                logits.append(token_logits)
        finally:
            if select_expert_subset is not None:
                select_expert_subset(0)
            hooks.close()
        return torch.cat(logits, dim=1), hooks.cache

    logits = []
    hooks = _Hooks(
        blocks,
        config,
        magnitude_diff_stats or MagnitudeDiffStats(),
        adjacent_layer_stats=adjacent_layer_stats,
    )
    if condition_thresholds is not None and len(condition_thresholds) not in (
        1,
        len(config.pairs),
    ):
        raise ValueError(
            "condition_thresholds must contain one value or one value per "
            f"source/destination pair ({len(config.pairs)})."
        )
    if condition_thresholds is not None and len(condition_thresholds) == 1:
        condition_thresholds = condition_thresholds * len(config.pairs)
    if config.act_sim_as_alpha and config.act_sim_min_max is not None:
        if len(config.act_sim_min_max) != 2:
            raise ValueError(
                "act_sim_min_max must contain exactly two values (MIN, MAX)."
            )
        if config.act_sim_min_max[0] >= config.act_sim_min_max[1]:
            raise ValueError("act_sim_min_max must have MIN < MAX.")
    if not 0 <= gating_pair_index < len(config.pairs):
        raise ValueError(
            f"gating_pair_index must be in [0, {len(config.pairs)})."
        )
    try:
        for position in range(input_ids.shape[1]):
            token = input_ids[:, position : position + 1]

            if select_expert_subset is not None:
                select_expert_subset(0)
            hooks.residuals.clear()
            hooks.mode = "capture"
            first_logits, cache = step(token, cache)
            hooks.mode = "off"
            if first_pass_logits is not None:
                first_pass_logits.append(first_logits)
            first_margin = (
                _top1_top2_probability_margin(first_logits)
                if margin_threshold is not None or pass_probability_margins is not None
                else None
            )
            first_top1_prob = (
                _top1_probability(first_logits)
                if top1_prob_threshold is not None
                else None
            )
            token_pass_probability_margins = (
                [first_margin]
                if pass_probability_margins is not None
                else None
            )
            final_logits = first_logits
            hooks.record_adjacent_similarities()

            similarities = (
                hooks.activation_similarities()
                if (
                    similarity_stats is not None
                    or condition_thresholds is not None
                    or first_pass_similarities is not None
                    or config.act_sim_as_alpha
                )
                else None
            )
            if first_pass_similarities is not None:
                assert similarities is not None
                first_pass_similarities.append(similarities)
            if similarity_stats is not None:
                assert similarities is not None
                similarity_stats.values.append(sum(similarities) / len(similarities))

            margin_gate = margin_threshold is None or first_margin <= margin_threshold
            top1_prob_gate = (
                top1_prob_threshold is None or first_top1_prob <= top1_prob_threshold
            )
            probability_gate = margin_gate and top1_prob_gate
            should_recirculate = probability_gate and (
                condition_thresholds is None
                or similarities[gating_pair_index]
                >= condition_thresholds[gating_pair_index]
            )
            hooks.active_pairs = (
                tuple(probability_gate for _pair in config.pairs)
                if condition_thresholds is None
                else tuple(
                    should_recirculate and similarity >= threshold
                    for similarity, threshold in zip(similarities, condition_thresholds)
                )
            )
            cached_token_passes = (
                [capture_cached_token(cache)]
                if should_recirculate and capture_cached_token is not None
                else None
            )
            p2_accepted = True
            p2_kl_divergence = None
            for pass_index in range(1, passes if should_recirculate else 1):
                cache = rewind_one(cache)
                if select_expert_subset is not None:
                    select_expert_subset(pass_index)
                hooks.injection_sources = {
                    source: hooks.residuals[source] for source in hooks.sources
                }
                if config.act_sim_as_alpha:
                    min_val, max_val = (
                        config.act_sim_min_max
                        if config.act_sim_min_max is not None
                        else (0.0, 1.0)
                    )
                    scale = max_val - min_val
                    hooks.injection_alphas = tuple(
                        min(
                            max((similarity - min_val) / scale, 0.0),
                            1.0,
                        )
                        * config.alpha
                        for similarity in similarities
                    )
                else:
                    hooks.injection_alphas = None
                hooks.expected_embedding = (
                    expected_embedding(final_logits[:, -1:, :])
                    if expected_embedding is not None
                    else None
                )
                hooks.mode = "inject"
                final_logits, cache = step(token, cache)
                hooks.mode = "off"
                if pass_index == 1:
                    p2_kl_divergence = _distribution_kl_divergence(
                        first_logits, final_logits
                    )
                    if kl_reject is not None:
                        p2_accepted = p2_kl_divergence <= kl_reject
                        if not p2_accepted:
                            assert cached_token_passes is not None
                            assert restore_cached_token is not None
                            restore_cached_token(cache, cached_token_passes[0])
                            final_logits = first_logits
                            break
                if cached_token_passes is not None:
                    cached_token_passes.append(capture_cached_token(cache))
                if token_pass_probability_margins is not None:
                    token_pass_probability_margins.append(
                        _top1_top2_probability_margin(final_logits)
                    )
            if (
                cached_token_passes is not None
                and average_cached_token is not None
                and p2_accepted
            ):
                average_cached_token(cache, cached_token_passes)
            if pass_probability_margins is not None:
                assert token_pass_probability_margins is not None
                pass_probability_margins.append(token_pass_probability_margins)
            if actual_alphas is not None:
                actual_alphas.append(
                    hooks.injection_alphas
                    if should_recirculate and p2_accepted and config.act_sim_as_alpha
                    else None
                )
            if recirculated_flags is not None:
                recirculated_flags.append(should_recirculate and p2_accepted)
            if rejected_flags is not None:
                rejected_flags.append(should_recirculate and not p2_accepted)
            if kl_divergences is not None:
                kl_divergences.append(p2_kl_divergence)
            logits.append(final_logits)
    finally:
        if select_expert_subset is not None:
            select_expert_subset(0)
        hooks.close()

    return torch.cat(logits, dim=1), cache

