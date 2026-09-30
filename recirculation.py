"""Minimal reference for multi-pass, training-free recirculation.
   Originally implemented by Benhao Huang:
   https://gist.github.com/huskydoge/1ff29693e2172226ec26081f208b19d6

In source-to-destination mode, the configured ``(source, destination)`` pair
is recirculated for every token (with three passes):
    1. Run a normal cached pass; return its logits and save residuals h_d, h_s.
  2. Rewind the KV cache by one position.
  3. Run the token again, replacing the output of destination block d with
      beta * h_d + alpha * (h_s' + noise * n), where h_s' is norm-matched to
      h_d and Gaussian n is norm-matched to h_s'. Noise is scaled by the
      preceding pass margin normalized to the configured post-margin band.
    4. Capture the new residuals, rewind, and repeat the recirculation once more.
      5. Evaluate the rejection gates on the final pass. Keep P1's KV cache when
          that pass has the same top-1 token, or restore both P1 logits and KV cache
          when a rejection gate fails; otherwise commit the final pass.

Without post-margin gates, fixed source recirculation stops early when its
top-1/top-2 probability margin narrows. With post-margin MIN/MAX gates, a pass
below MIN is rejected, a pass from MIN (inclusive) to MAX (exclusive) requires
the configured margin-ratio improvement, and a pass at or above MAX is accepted.
Adaptive source recirculation retries low-margin passes until its budget is
exhausted, then rejects the final adaptive trial if its margin narrows.

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
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FORCED_NOISE_MAX_ATTEMPTS = 1


@dataclass
class PerturbationSequence:
    saved: list[Tensor] = field(default_factory=list)
    replayed: Tensor | None = None
    replay_index: int = 0
    replay_application_count: int = 0
    source_shape: tuple[int, ...] | None = None
    replay_early_exit: bool = False


@dataclass(frozen=True)
class RecirculationConfig:
    pair: tuple[int, int]
    alpha: float
    beta: float | None = None  # None selects the convex mix: beta = 1 - alpha.
    noise_level_range: tuple[float, float] = (0.0, 0.0)
    perturbation_direction: Tensor | None = None
    perturbation_direction_scale: float = 1.0
    perturbation_target_token_id: int | None = None
    periodic_perturbation_candidate_count: int = 16
    periodic_perturbation_steps: int = 1
    periodic_perturbation_step_decay: float = 1.0
    save_perturbation: Path | None = None
    replay_perturbation: Path | None = None
    perturbation_sequence: PerturbationSequence = field(
        default_factory=PerturbationSequence, repr=False, compare=False
    )
    perturb_pre_margin_thres: float = 0.05
    noise_decay_per_pass: float = 0.5
    eps: float = 1e-8
    mode: Literal["source", "layerwise"] = "source"


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


def _residual(output: Any) -> Tensor:
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, Tensor):
        raise TypeError("A transformer block must return its residual stream first.")
    return hidden


def _top1_top2_probability_margin(logits: Tensor) -> float:
    probabilities = torch.softmax(logits[:, -1, :].float(), dim=-1)
    top_two = torch.topk(probabilities, k=2, dim=-1).values
    return float((top_two[..., 0] - top_two[..., 1]).item())


def _adaptive_noise_level(
    margin: float,
    post_margin_threshold: tuple[float, float],
    noise_level_range: tuple[float, float],
) -> float:
    post_margin_min, post_margin_max = post_margin_threshold
    noise_min, noise_max = noise_level_range
    normalized_margin = min(
        1.0,
        max(0.0, (margin - post_margin_min) / (post_margin_max - post_margin_min)),
    )
    return noise_min + normalized_margin * (noise_max - noise_min)


def _top1_tokens_match(p1_logits: Tensor, p2_logits: Tensor) -> bool:
    p1_top_token = p1_logits[:, -1, :].argmax(dim=-1)
    p2_top_token = p2_logits[:, -1, :].argmax(dim=-1)
    return bool((p1_top_token == p2_top_token).all().item())


def _distribution_cosine_similarity(
    teacher_logits: Tensor, student_logits: Tensor, top_k: int
) -> float:
    teacher_probabilities = torch.softmax(
        teacher_logits[:, -1, :].float(), dim=-1
    )
    student_probabilities = torch.softmax(
        student_logits[:, -1, :].float(), dim=-1
    )
    selected_count = min(top_k, teacher_probabilities.shape[-1])
    teacher_indices = torch.topk(
        teacher_probabilities, selected_count, dim=-1
    ).indices
    student_indices = torch.topk(
        student_probabilities, selected_count, dim=-1
    ).indices
    selected = torch.zeros_like(teacher_probabilities, dtype=torch.bool)
    selected.scatter_(dim=-1, index=teacher_indices, value=True)
    selected.scatter_(dim=-1, index=student_indices, value=True)
    similarity = torch.nn.functional.cosine_similarity(
        teacher_probabilities.masked_fill(~selected, 0.0),
        student_probabilities.masked_fill(~selected, 0.0),
        dim=-1,
    )
    return float(similarity.mean().item())


def perturbation_replay(
    normalized_source: Tensor, path: Path, sequence: PerturbationSequence
) -> Tensor | None:
    if sequence.replayed is None:
        sequence.replayed = torch.load(path, map_location="cpu", weights_only=True)
    perturbations = sequence.replayed
    if (
        not isinstance(perturbations, Tensor)
        or perturbations.ndim != normalized_source.ndim + 1
        or perturbations.shape[1:] != normalized_source.shape
    ):
        raise ValueError(
            f"Replay perturbations in {path} must be a tensor with shape "
            f"(N, {', '.join(map(str, normalized_source.shape))})."
        )
    if sequence.replay_index >= len(perturbations):
        return None
    perturbation = perturbations[sequence.replay_index].to(
        device=normalized_source.device, dtype=normalized_source.dtype
    )
    sequence.replay_index += 1
    return normalized_source + perturbation


def replay_skips_remaining_latents(config: RecirculationConfig, remaining: int) -> bool:
    if config.replay_perturbation is None or remaining == 0:
        return False
    sequence = config.perturbation_sequence
    if sequence.replayed is None:
        sequence.replayed = torch.load(config.replay_perturbation, map_location="cpu", weights_only=True)
    replayed = sequence.replayed
    if not isinstance(replayed, Tensor):
        return False
    if sequence.replay_index >= len(replayed):
        return sequence.replay_early_exit
    if not bool(torch.all(replayed[sequence.replay_index] == 0).item()):
        return sequence.replay_early_exit
    marker_limit = min(sequence.replay_index + remaining, len(replayed))
    while sequence.replay_index < marker_limit and bool(
        torch.all(replayed[sequence.replay_index] == 0).item()
    ):
        sequence.replay_index += 1
    return True


def save_skipped_latent_perturbations(
    config: RecirculationConfig, remaining: int, replayed_skips: bool = False
) -> None:
    if config.save_perturbation is None or remaining == 0:
        return
    sequence = config.perturbation_sequence
    if sequence.replayed is not None and len(sequence.saved) < sequence.replay_index:
        sequence.saved.extend(
            sequence.replayed[len(sequence.saved):sequence.replay_index].unbind(0)
        )
    if replayed_skips:
        torch.save(torch.stack(sequence.saved), config.save_perturbation)
        return
    if sequence.saved:
        zero = torch.zeros_like(sequence.saved[-1])
    elif sequence.source_shape is not None:
        zero = torch.zeros(sequence.source_shape)
    else:
        return
    sequence.saved.extend(zero.clone() for _ in range(remaining))
    torch.save(torch.stack(sequence.saved), config.save_perturbation)


class _Hooks:
    def __init__(
        self,
        blocks: Sequence[nn.Module],
        cfg: RecirculationConfig,
        force_recirculation: bool = False,
        adjacent_layer_stats: AdjacentLayerSimilarityStats | None = None,
        injected_source_latents: list[tuple[int, Tensor]] | None = None,
    ) -> None:
        source, destination = cfg.pair
        if not 0 <= destination < source < len(blocks):
            raise ValueError(
                "Expected 0 <= destination < source < number of blocks."
            )
        noise_min, noise_max = cfg.noise_level_range
        maximum_noise_level = 1.0
        if not 0.0 <= noise_min <= noise_max <= maximum_noise_level:
            raise ValueError(
                "noise_level_range requires 0 <= MIN <= MAX"
                f" <= {maximum_noise_level:g}."
            )
        if not 0.0 <= cfg.noise_decay_per_pass <= 1.0:
            raise ValueError("noise_decay_per_pass must be between 0 and 1.")
        if cfg.perturb_pre_margin_thres < 0:
            raise ValueError("perturb_pre_margin_thres must be nonnegative.")
        if cfg.periodic_perturbation_candidate_count < 1:
            raise ValueError("periodic_perturbation_candidate_count must be positive.")
        if cfg.periodic_perturbation_steps < 1:
            raise ValueError("periodic_perturbation_steps must be positive.")
        if not 0.0 <= cfg.periodic_perturbation_step_decay <= 1.0:
            raise ValueError("periodic_perturbation_step_decay must be between 0 and 1.")
        self.cfg = cfg
        self.mode = "off"
        self.pass_index = 1
        self.noise_level = 0.0
        self.destination_index = destination
        self.source_index = source
        self.residuals: dict[int, Tensor] = {}
        self.injection_source: Tensor | None = None
        self.active_pair = True
        self.adjacent_layer_stats = adjacent_layer_stats
        self.injected_source_latents = injected_source_latents
        self.prepared_source: Tensor | None = None
        self.noise_perturbation: tuple[Tensor, Tensor] | None = None
        self.noise_debug_position: int | None = None
        self.layer_residuals: dict[int, Tensor] = {}
        watched_layers = {source, destination}
        handles = [
            blocks[layer_index].register_forward_hook(self._save_residual(layer_index))
            for layer_index in watched_layers
        ]
        handles.append(
            blocks[destination + 1].register_forward_pre_hook(self._inject())
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

    def _inject(self) -> Callable[..., tuple | None]:
        def hook(_module: nn.Module, inputs: tuple) -> tuple | None:
            if self.mode != "inject" or not self.active_pair:
                return None
            try:
                destination = self.residuals[self.destination_index]
                source = self.prepared_source
                if source is None:
                    raise KeyError("prepared_source")
            except KeyError as error:
                raise RuntimeError("Recirculation injection was not prepared.") from error

            input_device = _residual(inputs).device
            destination = destination.to(device=input_device, dtype=torch.float32)
            source = source.to(device=input_device)
            alpha = self.cfg.alpha
            beta = 1.0 - alpha if self.cfg.beta is None else self.cfg.beta
            mixed = (beta * destination + alpha * source).to(inputs[0].dtype)
            return (mixed, *inputs[1:])

        return hook

    def iteratively_find_perturbation_candidate(
        self,
        source_index: int,
        destination_index: int,
        destination: Tensor,
        source: Tensor,
        destination_norm: Tensor,
        normalized_source: Tensor,
        direction_weight: float,
        perturbation_probe: Callable[[int, int, Tensor, Tensor], Tensor],
        candidate_target_token_probabilities: list[float] | None,
        aggregate_target_token_probabilities: list[float] | None,
    ) -> Tensor:
        target_token_id = self.cfg.perturbation_target_token_id
        if target_token_id is None:
            assert self.cfg.perturbation_direction is not None
            downstream_direction = self.cfg.perturbation_direction.to(
                device=source.device, dtype=torch.float32
            )
            baseline_output = source[:, -1, :]
        steps = self.cfg.periodic_perturbation_steps
        candidates_per_step = self.cfg.periodic_perturbation_candidate_count
        average_candidate_norm = None
        initial_normalized_source = normalized_source
        for _step in range(steps):
            candidates = []
            scores = []
            for _ in range(candidates_per_step):
                epsilon = (
                    direction_weight
                    * self.cfg.perturbation_direction_scale
                    * destination_norm
                    * torch.randn_like(source)
                    / source.shape[-1] ** 0.5
                )
                candidate = normalized_source + epsilon
                candidate_output = perturbation_probe(
                    source_index, destination_index, destination, candidate
                )
                if target_token_id is not None:
                    reversed_output = perturbation_probe(
                        source_index,
                        destination_index,
                        destination,
                        normalized_source - epsilon,
                    )
                    forward_score = torch.log_softmax(
                        candidate_output.float(), dim=-1
                    )[:, target_token_id]
                    reversed_score = torch.log_softmax(
                        reversed_output.float(), dim=-1
                    )[:, target_token_id]
                    keep_forward = (forward_score >= reversed_score).to(
                        epsilon.device
                    )
                    candidates.append(
                        torch.where(keep_forward[:, None, None], epsilon, -epsilon)
                    )
                    best_score = torch.maximum(forward_score, reversed_score)
                    scores.append(best_score)
                else:
                    candidates.append(candidate)
                    direction = downstream_direction.reshape_as(candidate_output)
                    scores.append(
                        torch.sum(
                            (candidate_output - baseline_output) * direction,
                            dim=-1,
                        )
                    )
            if target_token_id is not None:
                if candidate_target_token_probabilities is not None:
                    candidate_target_token_probabilities.extend(
                        float(score.exp().item()) for score in scores
                    )
                score_tensor = torch.stack(scores, dim=0)
                candidate_weights = torch.softmax(score_tensor, dim=0)
                candidate_weights = candidate_weights.masked_fill(
                    candidate_weights <= 0.5 / candidates_per_step,
                    0.0,
                )
                candidate_weights = candidate_weights / candidate_weights.sum(
                    dim=0, keepdim=True
                )
                candidate_tensor = torch.stack(candidates, dim=0)
                weighted_perturbation = (
                    candidate_tensor
                    * candidate_weights.to(source.device)[:, :, None, None]
                ).sum(dim=0)
                if average_candidate_norm is None:
                    average_candidate_norm = torch.linalg.vector_norm(
                        candidate_tensor, dim=-1, keepdim=True
                    ).mean(dim=0)
                weighted_perturbation = (
                    weighted_perturbation
                    * average_candidate_norm
                    / torch.linalg.vector_norm(
                        weighted_perturbation, dim=-1, keepdim=True
                    ).clamp_min(self.cfg.eps)
                )
                perturbed_source = normalized_source + (
                    self.cfg.periodic_perturbation_step_decay ** _step
                ) * weighted_perturbation
                if aggregate_target_token_probabilities is not None:
                    aggregate_output = perturbation_probe(
                        source_index, destination_index, destination, perturbed_source
                    )
                    aggregate_target_token_probabilities.append(
                        float(torch.softmax(aggregate_output.float(), dim=-1)[:, target_token_id].item())
                    )
                normalized_source = perturbed_source
            else:
                candidate_tensor = torch.stack(candidates, dim=0)
                score_tensor = torch.stack(scores, dim=0)
                best_index = score_tensor.argmin(dim=0)
                normalized_source = candidate_tensor.gather(
                    0,
                    best_index.to(candidate_tensor.device).view(1, -1, 1, 1).expand(
                        1, *candidate_tensor.shape[1:]
                    ),
                ).squeeze(0)
        if self.cfg.save_perturbation is not None:
            sequence = self.cfg.perturbation_sequence
            if not sequence.saved and sequence.replayed is not None:
                sequence.saved.extend(
                    sequence.replayed[:sequence.replay_index].unbind(0)
                )
            sequence.saved.append(
                (normalized_source - initial_normalized_source).detach().cpu().clone()
            )
            torch.save(
                torch.stack(sequence.saved),
                self.cfg.save_perturbation,
            )
        return normalized_source

    def prepare_injections(
        self,
        perturbation_probe: Callable[[int, int, Tensor, Tensor], Tensor] | None = None,
        candidate_target_token_probabilities: list[float] | None = None,
        aggregate_target_token_probabilities: list[float] | None = None,
        pre_recirculation_top_token_id: int | None = None,
        normalized_source_override: Tensor | None = None,
    ) -> list[tuple[int, Tensor]]:
        self.prepared_source = None
        self.noise_perturbation = None
        self.noise_debug_position = None
        debug_latents: list[tuple[int, Tensor]] = []
        if not self.active_pair:
            return debug_latents
        source_index, destination_index = self.cfg.pair
        try:
            destination = self.residuals[destination_index].float()
            if self.injection_source is None:
                raise KeyError("injection_source")
            source = self.injection_source.to(
                device=destination.device, dtype=torch.float32
            )
        except KeyError as error:
            raise RuntimeError(
                "The preceding pass did not capture both residual streams."
            ) from error
        source_norm = torch.linalg.vector_norm(source, dim=-1, keepdim=True)
        destination_norm = torch.linalg.vector_norm(
            destination, dim=-1, keepdim=True
        )
        normalized_source = source * destination_norm / source_norm.clamp_min(
            self.cfg.eps
        )
        self.cfg.perturbation_sequence.source_shape = tuple(normalized_source.shape)
        if normalized_source_override is not None:
            if normalized_source_override.shape != normalized_source.shape:
                raise ValueError("The saved normalized source has an incompatible shape.")
            normalized_source = normalized_source_override.to(
                device=destination.device, dtype=normalized_source.dtype
            )
        select_perturbation_candidate = (
            (perturbation_probe is not None or self.cfg.replay_perturbation is not None)
            and (
                self.cfg.perturbation_direction is not None
                or self.cfg.perturbation_target_token_id is not None
            )
        )
        if (
            select_perturbation_candidate
            and self.cfg.perturbation_target_token_id is not None
            and self.cfg.perturbation_target_token_id == pre_recirculation_top_token_id
        ):
            # pass_index is the recirculation pass index, not the latent token index.
            # This guard only checks at the first recirculation pass.
            if self.pass_index == 1:
                if self.cfg.replay_perturbation is not None:
                    perturbation_replay(
                        normalized_source, self.cfg.replay_perturbation,
                        self.cfg.perturbation_sequence,
                    )
                elif self.cfg.save_perturbation is not None:
                    save_skipped_latent_perturbations(self.cfg, 1)
            self.prepared_source = normalized_source
            return debug_latents
        # Don't search for perturbation candidates after the first recirculation pass
        if select_perturbation_candidate and self.pass_index > 1:
            self.prepared_source = normalized_source
            return debug_latents
        
        direction_weight = self.noise_level * (
            self.cfg.noise_decay_per_pass ** (self.pass_index - 1)
        )
        if self.noise_level > 0 and not select_perturbation_candidate:
            gaussian_noise = torch.randn_like(source)
            gaussian_direction = gaussian_noise / torch.linalg.vector_norm(
                gaussian_noise, dim=-1, keepdim=True
            ).clamp_min(self.cfg.eps)
            noise_weight = self.noise_level * (
                self.cfg.noise_decay_per_pass ** (self.pass_index - 1)
            )
            debug_latent = (
                source + noise_weight * source_norm * gaussian_direction
            ).detach().clone()
            debug_latents.append((source_index, debug_latent))
            normalized_perturbation = (
                noise_weight * destination_norm * gaussian_direction
            )
            raw_perturbation = noise_weight * source_norm * gaussian_direction
            self.noise_perturbation = (normalized_perturbation, raw_perturbation)
            if self.injected_source_latents is not None:
                self.noise_debug_position = len(self.injected_source_latents)
                self.injected_source_latents.append((source_index, debug_latent))
            normalized_source = normalized_source + normalized_perturbation
        if select_perturbation_candidate:
            if self.cfg.replay_perturbation is not None:
                replayed_source = perturbation_replay(
                    normalized_source,
                    self.cfg.replay_perturbation,
                    self.cfg.perturbation_sequence,
                )
            else:
                replayed_source = None
            if replayed_source is not None:
                if bool(torch.count_nonzero(
                    self.cfg.perturbation_sequence.replayed[
                        self.cfg.perturbation_sequence.replay_index - 1
                    ]
                ).item()):
                    self.cfg.perturbation_sequence.replay_application_count += 1
                normalized_source = replayed_source
            else:
                if perturbation_probe is None:
                    raise ValueError(
                        "Candidate search requires a perturbation probe after replay is exhausted."
                    )
                normalized_source = self.iteratively_find_perturbation_candidate(
                    source_index,
                    destination_index,
                    destination,
                    source,
                    destination_norm,
                    normalized_source,
                    direction_weight,
                    perturbation_probe,
                    candidate_target_token_probabilities,
                    aggregate_target_token_probabilities,
                )
        elif self.cfg.perturbation_direction is not None:
            direction = self.cfg.perturbation_direction.to(
                device=source.device, dtype=torch.float32
            )
            direction = direction / torch.linalg.vector_norm(
                direction, dim=-1, keepdim=True
            ).clamp_min(self.cfg.eps)
            normalized_source = normalized_source + (
                direction_weight
                * self.cfg.perturbation_direction_scale
                * destination_norm
                * direction
            )
        self.prepared_source = normalized_source
        if select_perturbation_candidate:
            debug_latents.append((source_index, normalized_source.detach().clone()))
        return debug_latents
    
    # reverse_noise() is called when a Gaussian-noise injection has been recorded
    # and decoding that noisy latent produces a margin greater than the 
    # preceding pass’s margin
    def reverse_noise(self) -> Tensor:
        assert self.noise_perturbation is not None
        assert self.prepared_source is not None
        assert self.injection_source is not None
        # prepared_source is the normalized source latent + normalized_perturbation.
        # (To be precise, it can also contain a direct-direction offset when perturbation_direction 
        # is set and no probe is supplied)
        # To reverse the noise, we subtract twice the normalized perturbation from the prepared source.
        normalized_perturbation, raw_perturbation = self.noise_perturbation
        self.prepared_source -= 2 * normalized_perturbation
        source_index = self.source_index
        reversed_latent = (
            self.injection_source.to(raw_perturbation.device).float()
            - raw_perturbation
        ).detach().clone()
        if self.injected_source_latents is not None:
            assert self.noise_debug_position is not None
            self.injected_source_latents[self.noise_debug_position] = (
                source_index, reversed_latent
            )
        return reversed_latent

    def activation_similarity(self) -> float:
        try:
            destination = self.residuals[self.destination_index][:, -1, :].float()
            source = self.residuals[self.source_index][:, -1, :].to(
                destination.device, dtype=torch.float32
            )
        except KeyError as error:
            raise RuntimeError(
                "The first pass did not capture both residual streams."
            ) from error
        cosine = torch.nn.functional.cosine_similarity(destination, source, dim=-1)
        return float(cosine.mean().item())

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
        source, destination = config.pair
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
            # NOTE: pass_index is 1-based.
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
    similarity_stats: SimilarityStats | None = None,
    adjacent_layer_stats: AdjacentLayerSimilarityStats | None = None,
    passes: int = 3,
    rewind_layer: Callable[[Any, int], Any] | None = None,
    condition_threshold: float | None = None,
    pre_margin_threshold: float | None = None,
    post_margin_threshold: tuple[float, float] | None = None,
    post_margin_ratio_threshold: float | None = None,
    adaptive_recirculation: int = 0,
    recirculation_allowed: bool = True,
    force_recirculation: bool = False,
    cosine_reject: float | None = None,
    cosine_top_k: int = 100,
    first_pass_logits: list[Tensor] | None = None,
    first_pass_similarities: list[float] | None = None,
    pass_probability_margins: list[list[float]] | None = None,
    recirculated_flags: list[bool] | None = None,
    rejected_flags: list[bool] | None = None,
    adaptive_recirculated_flags: list[bool] | None = None,
    adaptive_rejected_flags: list[bool] | None = None,
    adaptive_recirculation_counts: list[int] | None = None,
    final_pass_same_top1_flags: list[bool] | None = None,
    rejection_reasons: list[tuple[str, ...]] | None = None,
    final_pass_cosine_similarities: list[float | None] | None = None,
    pass_cosine_similarities: list[list[float]] | None = None,
    pass_top_k_token_ids: list[list[list[int]]] | None = None,
    target_token_id: int | None = None,
    pass_target_token_probabilities: list[list[float]] | None = None,
    candidate_target_token_probabilities: list[list[float]] | None = None,
    aggregate_target_token_probabilities: list[list[float]] | None = None,
    cosine_reject_thresholds: list[float | None] | None = None,
    injected_noise_levels: list[list[float]] | None = None,
    injected_source_latents: list[list[tuple[int, Tensor]]] | None = None,
    source_latents: list[dict[int, Tensor]] | None = None,
    normalized_sources: list[Tensor] | None = None,
    normalized_source_override: Tensor | None = None,
    decode_injected_source_latent: Callable[[Tensor, int, Tensor, Any, Any], Any]
    | None = None,
    initial_decoded_noise_source_latents: list[list[Any]] | None = None,
    decoded_injected_source_latents: list[list[Any]] | None = None,
    perturbation_probe: Callable[
        [Tensor, int, int, Tensor, Tensor, Any, Any, RecirculationConfig], Tensor
    ]
    | None = None,
    capture_cached_token: Callable[[Any], Any] | None = None,
    average_cached_token: Callable[[Any, Sequence[Any]], None] | None = None,
    restore_cached_token: Callable[[Any, Any], None] | None = None,
    capture_rewind_state: Callable[[Any], Any] | None = None,
    restore_rewind_state: Callable[[Any, Any], None] | None = None,
    finalize_token_cache: Callable[[Any], Any] | None = None,
) -> tuple[Tensor, Any]:
    """Run source-to-destination recirculation or layerwise repeated passes."""

    if passes < 1:
        raise ValueError("passes must be at least 1.")
    if adaptive_recirculation < 0:
        raise ValueError("adaptive_recirculation must be nonnegative.")
    if passes == 1 and not adaptive_recirculation and not force_recirculation:
        condition_threshold = None
        pre_margin_threshold = None
        post_margin_threshold = None
        post_margin_ratio_threshold = None
        cosine_reject = None
    if force_recirculation:
        condition_threshold = None
        pre_margin_threshold = None
        post_margin_threshold = None
        post_margin_ratio_threshold = None
        if cosine_reject is not None:
            cosine_reject = min(cosine_reject, 0.2)
    if post_margin_threshold is not None:
        post_margin_min, post_margin_max = post_margin_threshold
        if post_margin_min < 0 or post_margin_max < 0:
            raise ValueError("post_margin_threshold values must be nonnegative.")
        if post_margin_min > post_margin_max:
            raise ValueError("post_margin_threshold requires MIN <= MAX.")
    if not force_recirculation and config.noise_level_range[1] > 0 and (
        post_margin_threshold is None
        or post_margin_threshold[0] == post_margin_threshold[1]
    ):
        raise ValueError(
            "noise_level_range requires post_margin_threshold with MIN < MAX."
        )
    if config.noise_level_range[1] > 0 and decode_injected_source_latent is None:
        raise ValueError(
            "noise_level_range requires decode_injected_source_latent."
        )
    multi_pass = passes >= 2 or adaptive_recirculation > 0 or force_recirculation
    if (
        post_margin_ratio_threshold is not None
        and post_margin_threshold is None
    ):
        raise ValueError(
            "post_margin_ratio_threshold requires post_margin_threshold MIN/MAX."
        )
    if cosine_top_k < 1:
        raise ValueError("cosine_top_k must be at least 1.")
    if average_cached_token is not None and capture_cached_token is None:
        raise ValueError(
            "average_cached_token requires capture_cached_token."
        )
    if multi_pass and config.mode == "source" and (
        capture_cached_token is None or restore_cached_token is None
    ):
        raise ValueError(
            "Multi-pass source recirculation requires capture_cached_token and "
            "restore_cached_token."
        )
    if (capture_rewind_state is None) != (restore_rewind_state is None):
        raise ValueError(
            "capture_rewind_state and restore_rewind_state must be provided together."
        )

    if config.mode == "layerwise":
        if condition_threshold is not None:
            raise ValueError(
                "Conditional recirculation currently requires --mode source."
            )
        if pre_margin_threshold is not None:
            raise ValueError(
                "Margin-gated recirculation currently requires --mode source."
            )
        if post_margin_threshold is not None:
            raise ValueError(
                "Final-pass margin-gated recirculation requires --mode source."
            )
        if post_margin_ratio_threshold is not None:
            raise ValueError(
                "Final-pass margin-ratio-gated recirculation requires --mode source."
            )
        if cosine_reject is not None:
            raise ValueError("Final-pass trust rejection requires --mode source.")
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
        force_recirculation=force_recirculation,
        adjacent_layer_stats=adjacent_layer_stats,
        injected_source_latents=(
            []
            if injected_source_latents is not None
            or decode_injected_source_latent is not None
            else None
        ),
    )
    try:
        for position in range(input_ids.shape[1]):
            token = input_ids[:, position : position + 1]
            rewind_state = (
                capture_rewind_state(cache)
                if capture_rewind_state is not None
                else None
            )

            if select_expert_subset is not None:
                select_expert_subset(0)
            hooks.residuals.clear()
            hooks.mode = "capture"
            first_logits, cache = step(token, cache)
            hooks.mode = "off"
            first_pass_residuals = hooks.residuals.copy()
            if source_latents is not None:
                source_latents.append(
                    {
                        source: latent[:, -1, :].detach().clone()
                        for source, latent in first_pass_residuals.items()
                        if source == hooks.source_index
                    }
                )
            if first_pass_logits is not None:
                first_pass_logits.append(first_logits)
            first_margin = (
                _top1_top2_probability_margin(first_logits)
                if pre_margin_threshold is not None
                or passes >= 2
                or adaptive_recirculation
                or force_recirculation
                or pass_probability_margins is not None
                else None
            )
            token_pass_probability_margins = (
                [first_margin]
                if pass_probability_margins is not None
                else None
            )
            final_logits = first_logits
            hooks.record_adjacent_similarities()

            similarity = (
                hooks.activation_similarity()
                if (
                    similarity_stats is not None
                    or condition_threshold is not None
                    or first_pass_similarities is not None
                )
                else None
            )
            if first_pass_similarities is not None:
                assert similarity is not None
                first_pass_similarities.append(similarity)
            if similarity_stats is not None:
                assert similarity is not None
                similarity_stats.values.append(similarity)

            margin_gate = (
                pre_margin_threshold is None
                or first_margin <= pre_margin_threshold
            )
            probability_gate = margin_gate
            target_already_top_one = (
                config.perturbation_target_token_id is not None
                and int(first_logits[0, -1, :].argmax().item())
                == config.perturbation_target_token_id
            )
            should_recirculate = recirculation_allowed and (
                force_recirculation
                or (
                    probability_gate
                    and (
                        condition_threshold is None
                        or similarity >= condition_threshold
                    )
                )
            ) and not target_already_top_one
            if force_recirculation and config.perturbation_target_token_id is not None:
                sequence = config.perturbation_sequence
                sequence.replay_early_exit = False
                if config.replay_perturbation is not None:
                    sequence.replay_early_exit = replay_skips_remaining_latents(config, 1)
                    if sequence.replay_early_exit:
                        should_recirculate = False
                    elif target_already_top_one:
                        perturbation_replay(
                            first_pass_residuals[hooks.source_index],
                            config.replay_perturbation,
                            sequence,
                        )
                elif target_already_top_one:
                    sequence.source_shape = tuple(first_pass_residuals[hooks.source_index].shape)
                    save_skipped_latent_perturbations(config, 1)
            if passes == 1 and adaptive_recirculation and not force_recirculation:
                assert first_margin is not None
                should_recirculate = (
                    should_recirculate
                    and (
                        post_margin_threshold is None
                        or first_margin < post_margin_threshold[1]
                    )
                )
            hooks.active_pair = (
                probability_gate
                if condition_threshold is None
                else should_recirculate
            )
            cached_token_passes = (
                [capture_cached_token(cache)]
                if should_recirculate and capture_cached_token is not None
                else None
            )
            final_pass_accepted = True
            cache_restored = False
            final_pass_same_top1 = False
            final_pass_rejection_reasons: list[str] = []
            final_pass_cosine_similarity = None
            token_pass_cosine_similarities: list[float] = []
            token_pass_top_k_token_ids: list[list[int]] = []
            token_pass_target_token_probabilities: list[float] = []
            token_candidate_target_token_probabilities: list[float] = []
            token_aggregate_target_token_probabilities: list[float] = []
            max_passes = max(
                passes + adaptive_recirculation,
                2 if force_recirculation else 1,
            )
            adaptive_recirculation_count = 0
            forced_noise_attempt_count = 0
            retrying_forced_noise = False
            previous_pass_margin = first_margin
            token_injected_noise_levels: list[float] = []
            token_initial_decoded_noise_source_latents: list[Any] = []
            token_decoded_injected_source_latents: list[Any] = []
            if hooks.injected_source_latents is not None:
                hooks.injected_source_latents.clear()
            pass_index = 1
            while pass_index < (max_passes if should_recirculate else 1):
                if (
                    adaptive_recirculation > 0
                    and pass_index >= passes
                    and not retrying_forced_noise
                ):
                    adaptive_recirculation_count += 1
                retrying_forced_noise = False
                hooks.injection_source = hooks.residuals[hooks.source_index]
                hooks.pass_index = pass_index
                hooks.noise_level = (
                    config.noise_level_range[1]
                    if force_recirculation and config.noise_level_range[1] > 0
                    else _adaptive_noise_level(
                        previous_pass_margin,
                        post_margin_threshold,
                        config.noise_level_range,
                    )
                    if config.noise_level_range[1] > 0
                    and previous_pass_margin is not None
                    and previous_pass_margin >= config.perturb_pre_margin_thres
                    and post_margin_threshold is not None
                    else 0.0
                )
                token_injected_noise_levels.append(
                    hooks.noise_level
                    * (config.noise_decay_per_pass ** (pass_index - 1))
                )
                if force_recirculation and token_injected_noise_levels[-1] > 0:
                    forced_noise_attempt_count += 1
                prepared_debug_latents: list[tuple[int, Tensor]] = []
                noise_debug_latents = hooks.prepare_injections(
                    (
                        lambda source_index, destination_index, destination, candidate: (
                            perturbation_probe(
                                token,
                                source_index,
                                destination_index,
                                destination,
                                candidate,
                                cache,
                                rewind_state,
                                config,
                            )
                        )
                        if perturbation_probe is not None
                        else None
                    )
                    if perturbation_probe is not None
                    else None,
                    candidate_target_token_probabilities=(
                        token_candidate_target_token_probabilities
                        if candidate_target_token_probabilities is not None
                        else None
                    ),
                    aggregate_target_token_probabilities=(
                        token_aggregate_target_token_probabilities
                        if aggregate_target_token_probabilities is not None
                        else None
                    ),
                    pre_recirculation_top_token_id=(
                        int(first_logits[0, -1, :].argmax().item())
                        if config.perturbation_target_token_id is not None
                        else None
                    ),
                    normalized_source_override=(
                        normalized_source_override if pass_index == 1 else None
                    ),
                )
                if normalized_sources is not None and pass_index == 1:
                    assert hooks.prepared_source is not None
                    normalized_sources.append(hooks.prepared_source.detach().clone())
                if not prepared_debug_latents:
                    prepared_debug_latents = noise_debug_latents
                if decode_injected_source_latent is not None:
                    decoded_latents = [
                        decode_injected_source_latent(
                            token, source_index, latent, cache, rewind_state
                        )
                        for source_index, latent in prepared_debug_latents
                    ]
                    # noise_perturbation is only populated in prepare_injections()
                    # by the ordinary Gaussian branch.
                    if hooks.noise_perturbation is not None:
                        assert previous_pass_margin is not None
                        token_initial_decoded_noise_source_latents.extend(
                            decoded_latents
                        )
                        if float(decoded_latents[0]["margin"]) > previous_pass_margin:
                            reversed_latent = hooks.reverse_noise()
                            decoded_latents = [
                                decode_injected_source_latent(
                                    token,
                                    hooks.source_index,
                                    reversed_latent,
                                    cache,
                                    rewind_state,
                                )
                            ]
                    token_decoded_injected_source_latents.extend(decoded_latents)
                cache = rewind_one(cache)
                if restore_rewind_state is not None:
                    restore_rewind_state(cache, rewind_state)
                if select_expert_subset is not None:
                    select_expert_subset(pass_index)
                hooks.mode = "inject"
                final_logits, cache = step(token, cache)
                hooks.mode = "off"
                if pass_cosine_similarities is not None:
                    token_pass_cosine_similarities.append(
                        _distribution_cosine_similarity(
                            first_logits, final_logits, cosine_top_k
                        )
                    )
                if pass_top_k_token_ids is not None:
                    token_pass_top_k_token_ids.append(
                        torch.topk(
                            final_logits[:, -1, :].float(),
                            k=min(cosine_top_k, final_logits.shape[-1]),
                            dim=-1,
                        ).indices[0].tolist()
                    )
                if pass_target_token_probabilities is not None and target_token_id is not None:
                    token_pass_target_token_probabilities.append(
                        float(torch.softmax(final_logits[0, -1, :].float(), dim=-1)[target_token_id].item())
                    )
                pass_margin = (
                    _top1_top2_probability_margin(final_logits)
                    if passes >= 2
                    or adaptive_recirculation
                    or post_margin_threshold is not None
                    or token_pass_probability_margins is not None
                    else None
                )
                if token_pass_probability_margins is not None:
                    assert pass_margin is not None
                    token_pass_probability_margins.append(pass_margin)
                margin_narrowed = (
                    pass_margin is not None
                    and previous_pass_margin is not None
                    and pass_margin < previous_pass_margin
                )
                post_margin_min = (
                    post_margin_threshold[0]
                    if post_margin_threshold is not None
                    else None
                )
                post_margin_max = (
                    post_margin_threshold[1]
                    if post_margin_threshold is not None
                    else None
                )
                post_margin_ratio_met = (
                    post_margin_ratio_threshold is not None
                    and pass_margin is not None
                    and first_margin is not None
                    and pass_margin / max(first_margin, config.eps)
                    >= post_margin_ratio_threshold
                )
                post_margin_below_min = (
                    post_margin_min is not None
                    and pass_margin is not None
                    and pass_margin < post_margin_min
                )
                post_margin_in_middle = (
                    post_margin_min is not None
                    and post_margin_max is not None
                    and pass_margin is not None
                    and post_margin_min <= pass_margin < post_margin_max
                )
                post_margin_at_or_above_max = (
                    post_margin_max is not None
                    and pass_margin is not None
                    and pass_margin >= post_margin_max
                )
                margin_gate_met = post_margin_at_or_above_max or (
                    post_margin_in_middle and post_margin_ratio_met
                )
                has_margin_gate = (
                    post_margin_threshold is not None
                )
                adaptive_margin_narrowed = (
                    has_margin_gate
                    and adaptive_recirculation_count > 0
                    and margin_narrowed
                    and pass_index == max_passes - 1
                )
                should_retry_low_margin = (
                    pass_index >= passes - 1
                    and has_margin_gate
                    and not margin_gate_met
                    and pass_index < max_passes - 1
                )
                if pass_margin is not None:
                    previous_pass_margin = pass_margin
                if (
                    margin_narrowed
                    and not has_margin_gate
                    and not force_recirculation
                ) or adaptive_margin_narrowed or (
                    pass_index >= passes - 1 and not should_retry_low_margin
                ):
                    final_pass_cosine_similarity = (
                        token_pass_cosine_similarities[-1]
                        if pass_cosine_similarities is not None
                        else _distribution_cosine_similarity(
                            first_logits, final_logits, cosine_top_k
                        )
                    )
                    if adaptive_margin_narrowed:
                        final_pass_rejection_reasons.append("margin-narrowed")
                    if (
                        has_margin_gate
                        and not adaptive_margin_narrowed
                        and not margin_gate_met
                    ):
                        assert pass_margin is not None
                        if post_margin_below_min:
                            final_pass_rejection_reasons.append("post-margin-min")
                        elif post_margin_in_middle and not post_margin_ratio_met:
                            final_pass_rejection_reasons.append("post-margin-ratio")
                    if (
                        cosine_reject is not None
                        and final_pass_cosine_similarity < cosine_reject
                    ):
                        final_pass_rejection_reasons.append("cosine")
                    final_pass_accepted = not final_pass_rejection_reasons
                    if not final_pass_accepted:
                        assert cached_token_passes is not None
                        assert restore_cached_token is not None
                        restore_cached_token(cache, cached_token_passes[0])
                        cache_restored = True
                        final_logits = first_logits
                        if (
                            force_recirculation
                            and final_pass_rejection_reasons == ["cosine"]
                            and token_injected_noise_levels[-1] > 0
                            and forced_noise_attempt_count < FORCED_NOISE_MAX_ATTEMPTS
                        ):
                            final_pass_rejection_reasons.clear()
                            hooks.residuals = first_pass_residuals.copy()
                            previous_pass_margin = first_margin
                            cache_restored = False
                            retrying_forced_noise = True
                            continue
                        break
                    final_pass_same_top1 = _top1_tokens_match(
                        first_logits, final_logits
                    )
                    if final_pass_same_top1:
                        assert cached_token_passes is not None
                        assert restore_cached_token is not None
                        restore_cached_token(cache, cached_token_passes[0])
                        cache_restored = True
                    if cached_token_passes is not None:
                        cached_token_passes.append(capture_cached_token(cache))
                    break
                if cached_token_passes is not None:
                    cached_token_passes.append(capture_cached_token(cache))
                pass_index += 1
            if (
                cached_token_passes is not None
                and average_cached_token is not None
                and final_pass_accepted
                and not cache_restored
            ):
                average_cached_token(cache, cached_token_passes)
            if pass_probability_margins is not None:
                assert token_pass_probability_margins is not None
                pass_probability_margins.append(token_pass_probability_margins)
            if recirculated_flags is not None:
                recirculated_flags.append(
                    multi_pass
                    and should_recirculate
                    and final_pass_accepted
                )
            if rejected_flags is not None:
                rejected_flags.append(
                    should_recirculate and not final_pass_accepted
                )
            if adaptive_recirculated_flags is not None:
                adaptive_recirculated_flags.append(adaptive_recirculation_count > 0)
            if adaptive_rejected_flags is not None:
                adaptive_rejected_flags.append(
                    adaptive_recirculation_count > 0 and not final_pass_accepted
                )
            if adaptive_recirculation_counts is not None:
                adaptive_recirculation_counts.append(adaptive_recirculation_count)
            if final_pass_same_top1_flags is not None:
                final_pass_same_top1_flags.append(
                    should_recirculate and final_pass_same_top1
                )
            if rejection_reasons is not None:
                rejection_reasons.append(
                    tuple(final_pass_rejection_reasons)
                    if should_recirculate
                    else ()
                )
            if final_pass_cosine_similarities is not None:
                final_pass_cosine_similarities.append(
                    final_pass_cosine_similarity
                )
            if pass_cosine_similarities is not None:
                pass_cosine_similarities.append(token_pass_cosine_similarities)
            if pass_top_k_token_ids is not None:
                pass_top_k_token_ids.append(token_pass_top_k_token_ids)
            if pass_target_token_probabilities is not None:
                pass_target_token_probabilities.append(token_pass_target_token_probabilities)
            if candidate_target_token_probabilities is not None:
                candidate_target_token_probabilities.append(
                    token_candidate_target_token_probabilities
                )
            if aggregate_target_token_probabilities is not None:
                aggregate_target_token_probabilities.append(
                    token_aggregate_target_token_probabilities
                )
            if cosine_reject_thresholds is not None:
                cosine_reject_thresholds.append(cosine_reject)
            if injected_noise_levels is not None:
                injected_noise_levels.append(token_injected_noise_levels)
            if injected_source_latents is not None:
                assert hooks.injected_source_latents is not None
                injected_source_latents.append(hooks.injected_source_latents.copy())
            if initial_decoded_noise_source_latents is not None:
                initial_decoded_noise_source_latents.append(
                    token_initial_decoded_noise_source_latents
                )
            if decoded_injected_source_latents is not None:
                decoded_injected_source_latents.append(
                    token_decoded_injected_source_latents
                )
            if finalize_token_cache is not None:
                cache = finalize_token_cache(cache)
            logits.append(final_logits)
    finally:
        if select_expert_subset is not None:
            select_expert_subset(0)
        hooks.close()

    return torch.cat(logits, dim=1), cache

