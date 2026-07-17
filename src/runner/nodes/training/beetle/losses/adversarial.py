from dataclasses import dataclass
from collections.abc import Iterable
from typing import Protocol

import torch
from torch import Tensor, nn

from ..models.modules.discriminators import (
    DiscriminatorEvaluation,
    StyleTTSDiscriminatorOutput,
)


class PairedDiscriminator(Protocol):
    def __call__(self, real: Tensor, fake: Tensor) -> StyleTTSDiscriminatorOutput: ...

    def parameters(self, recurse: bool = True) -> Iterable[nn.Parameter]: ...


@dataclass(frozen=True)
class GeneratorAdversarialLoss:
    adversarial: Tensor
    feature_matching: Tensor


def discriminator_lsgan_loss(
    real_logits: list[Tensor],
    fake_logits: list[Tensor],
) -> Tensor:
    if not real_logits or len(real_logits) != len(fake_logits):
        raise ValueError("real and fake discriminator logits must align")
    losses = [
        (1 - real).square().mean() + fake.square().mean()
        for real, fake in zip(real_logits, fake_logits, strict=True)
    ]
    return torch.stack(losses).sum()


def generator_lsgan_loss(fake_logits: list[Tensor]) -> Tensor:
    if not fake_logits:
        raise ValueError("generator loss requires discriminator logits")
    return torch.stack([(1 - fake).square().mean() for fake in fake_logits]).sum()


def feature_matching_loss(
    real_features: list[list[Tensor]],
    fake_features: list[list[Tensor]],
) -> Tensor:
    if not real_features or len(real_features) != len(fake_features):
        raise ValueError("real and fake discriminator feature groups must align")
    losses = [
        (real.detach() - fake).abs().mean()
        for real_group, fake_group in zip(real_features, fake_features, strict=True)
        for real, fake in zip(real_group, fake_group, strict=True)
    ]
    return torch.stack(losses).sum() * 2


def discriminator_step_loss(
    discriminators: PairedDiscriminator,
    real: Tensor,
    generated: Tensor,
) -> Tensor:
    output = discriminators(real, generated.detach())
    return discriminator_lsgan_loss(output.real.logits, output.fake.logits)


def generator_step_loss(
    discriminators: PairedDiscriminator,
    real: Tensor,
    generated: Tensor,
) -> GeneratorAdversarialLoss:
    parameters = tuple(discriminators.parameters())
    trainability = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        output = discriminators(real, generated)
    finally:
        for parameter, requires_grad in zip(parameters, trainability, strict=True):
            parameter.requires_grad_(requires_grad)
    return GeneratorAdversarialLoss(
        adversarial=generator_lsgan_loss(output.fake.logits),
        feature_matching=feature_matching_loss(
            output.real.feature_maps,
            output.fake.feature_maps,
        ),
    )


__all__ = [
    "DiscriminatorEvaluation",
    "GeneratorAdversarialLoss",
    "StyleTTSDiscriminatorOutput",
    "discriminator_lsgan_loss",
    "discriminator_step_loss",
    "feature_matching_loss",
    "generator_lsgan_loss",
    "generator_step_loss",
]
