from dataclasses import dataclass

import torch
from torch import Tensor

from ...losses.acoustic import ReconstructionLoss
from ...losses.composition import Stage1LossWeights
from ...models.model import Stage1Synthesis
from ..reporting import TrainingMetric


DIAGNOSTICS_EVERY_STEPS = 250
GradientTuple = tuple[Tensor | None, ...]


@dataclass(frozen=True)
class Stage1GradientLosses:
    encoder_kl: Tensor
    f0: Tensor
    n: Tensor
    reconstruction: Tensor
    adversarial: Tensor
    feature_matching: Tensor


def diagnostics_due(completed_step: int) -> bool:
    if completed_step < 0:
        raise ValueError("completed step must be nonnegative")
    return completed_step > 0 and completed_step % DIAGNOSTICS_EVERY_STEPS == 0


def weighted_gradients(
    loss: Tensor,
    weight: float,
    targets: tuple[Tensor, ...],
) -> GradientTuple:
    if not targets:
        raise ValueError("gradient diagnostics require targets")
    return torch.autograd.grad(
        loss * weight,
        targets,
        retain_graph=True,
        allow_unused=True,
    )


def gradient_norm(gradients: GradientTuple) -> float:
    values = tuple(gradient.float() for gradient in gradients if gradient is not None)
    if not values:
        raise ValueError("diagnostic objective produced no gradients")
    squared = torch.stack(tuple(value.square().sum() for value in values)).sum()
    return float(torch.sqrt(squared))


def gradient_cosine(left: GradientTuple, right: GradientTuple) -> float:
    return gradient_cosine_observation(left, right)[0]


def gradient_cosine_observation(
    left: GradientTuple,
    right: GradientTuple,
) -> tuple[float, float]:
    if len(left) != len(right):
        raise ValueError("gradient cosine inputs must align")
    pairs = tuple(
        (left_value.float(), right_value.float())
        for left_value, right_value in zip(left, right, strict=True)
        if left_value is not None and right_value is not None
    )
    if not pairs:
        raise ValueError("gradient cosine requires shared targets")
    dot = torch.stack(tuple((a * b).sum() for a, b in pairs)).sum()
    left_norm = torch.sqrt(torch.stack(tuple(a.square().sum() for a, _ in pairs)).sum())
    right_norm = torch.sqrt(torch.stack(tuple(b.square().sum() for _, b in pairs)).sum())
    if left_norm == 0 or right_norm == 0:
        return 0.0, 0.0
    return float(dot / (left_norm * right_norm)), 1.0


def reconstruction_metrics(loss: ReconstructionLoss) -> tuple[TrainingMetric, ...]:
    resolutions = tuple(
        TrainingMetric(
            "reconstruction/"
            f"fft_{item.resolution.n_fft}_"
            f"hop_{item.resolution.hop_length}_"
            f"win_{item.resolution.win_length}",
            float(item.value.detach().float()),
        )
        for item in loss.resolutions
    )
    bands = tuple(
        TrainingMetric(
            f"reconstruction/{item.band.minimum_hz // 1000}_{item.band.maximum_hz // 1000}khz",
            float(item.value.detach().float()),
        )
        for item in loss.bands
    )
    return (*resolutions, *bands)


def stage1_gradient_metrics(
    losses: Stage1GradientLosses,
    weights: Stage1LossWeights,
    synthesis: Stage1Synthesis,
) -> tuple[TrainingMetric, ...]:
    waveform_target = (synthesis.waveform,)
    shared_targets = (*waveform_target, synthesis.decoded.features)
    reconstruction = weighted_gradients(
        losses.reconstruction, weights.reconstruction, shared_targets
    )
    adversarial = weighted_gradients(
        losses.adversarial, weights.generator_adversarial, shared_targets
    )
    feature_matching = weighted_gradients(
        losses.feature_matching, weights.feature_matching, shared_targets
    )
    f0 = weighted_gradients(losses.f0, weights.f0, (synthesis.acoustic.f0,))
    n = weighted_gradients(losses.n, weights.n, (synthesis.acoustic.n,))
    kl = weighted_gradients(
        losses.encoder_kl,
        weights.encoder_kl,
        (synthesis.posterior.mean, synthesis.posterior.log_scale),
    )
    adversarial_cosine = gradient_cosine_observation(
        reconstruction[:1], adversarial[:1]
    )
    feature_matching_cosine = gradient_cosine_observation(
        reconstruction[:1], feature_matching[:1]
    )
    return (
        _norm_metric("gradient_by_loss/reconstruction/waveform", reconstruction[:1]),
        _norm_metric("gradient_by_loss/adversarial/waveform", adversarial[:1]),
        _norm_metric("gradient_by_loss/feature_matching/waveform", feature_matching[:1]),
        _norm_metric(
            "gradient_by_loss/reconstruction/generator_input", reconstruction[1:]
        ),
        _norm_metric("gradient_by_loss/adversarial/generator_input", adversarial[1:]),
        _norm_metric(
            "gradient_by_loss/feature_matching/generator_input", feature_matching[1:]
        ),
        _norm_metric("gradient_by_loss/f0/acoustic_f0", f0),
        _norm_metric("gradient_by_loss/n/acoustic_n", n),
        _norm_metric("gradient_by_loss/kl/posterior", kl),
        TrainingMetric(
            "gradient_cosine/reconstruction_adversarial",
            adversarial_cosine[0],
        ),
        TrainingMetric(
            "gradient_cosine/reconstruction_adversarial_defined",
            adversarial_cosine[1],
        ),
        TrainingMetric(
            "gradient_cosine/reconstruction_feature_matching",
            feature_matching_cosine[0],
        ),
        TrainingMetric(
            "gradient_cosine/reconstruction_feature_matching_defined",
            feature_matching_cosine[1],
        ),
    )


def stage1_training_metrics(
    completed_step: int,
    losses: Stage1GradientLosses,
    reconstruction: ReconstructionLoss,
    total: Tensor,
    weights: Stage1LossWeights,
    synthesis: Stage1Synthesis,
) -> tuple[TrainingMetric, ...]:
    metrics = (
        _tensor_metric("encoder_kl", losses.encoder_kl),
        _tensor_metric("f0", losses.f0),
        _tensor_metric("n", losses.n),
        _tensor_metric("reconstruction", losses.reconstruction),
        _tensor_metric("generator_adversarial", losses.adversarial),
        _tensor_metric("feature_matching", losses.feature_matching),
        _tensor_metric("generator_total", total),
    )
    if not diagnostics_due(completed_step):
        return metrics
    return (
        *metrics,
        *reconstruction_metrics(reconstruction),
        *stage1_gradient_metrics(losses, weights, synthesis),
    )


def _norm_metric(name: str, gradients: GradientTuple) -> TrainingMetric:
    return TrainingMetric(name, gradient_norm(gradients))


def _tensor_metric(name: str, value: Tensor) -> TrainingMetric:
    return TrainingMetric(name, float(value.detach().float()))
