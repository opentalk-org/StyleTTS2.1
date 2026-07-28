from dataclasses import dataclass
import torch
from torch import Tensor
from torch.nn import functional as F

@dataclass(frozen=True)
class GeneratorAdversarialLoss:
    adversarial: Tensor
    feature_period: Tensor
    feature_resolution: Tensor
    feature_matching: Tensor


def discriminator_lsgan_loss(
    real_logits: list[Tensor],
    fake_logits: list[Tensor],
) -> Tensor:
    losses = [
        (1 - real).square().mean() + fake.square().mean()
        for real, fake in zip(real_logits, fake_logits, strict=True)
    ]
    return torch.stack(losses).sum()


def generator_lsgan_loss(fake_logits: list[Tensor]) -> Tensor:
    return torch.stack([(1 - fake).square().mean() for fake in fake_logits]).sum()


def discriminator_tprls_loss(
    real_logits: list[Tensor],
    fake_logits: list[Tensor],
) -> Tensor:
    losses = []
    for real, fake in zip(real_logits, fake_logits, strict=True):
        difference = real - fake
        median = torch.median(difference)
        relative = (difference - median).square()[real < fake + median].mean()
        losses.append(0.04 - F.relu(relative.new_tensor(0.04) - relative))
    return torch.stack(losses).sum()


def generator_tprls_loss(
    real_logits: list[Tensor],
    fake_logits: list[Tensor],
) -> Tensor:
    losses = []
    for real, fake in zip(real_logits, fake_logits, strict=True):
        difference = fake - real
        median = torch.median(difference)
        relative = (difference - median).square()[fake < real + median].mean()
        losses.append(0.04 - F.relu(relative.new_tensor(0.04) - relative))
    return torch.stack(losses).sum()


def feature_matching_loss(
    real_features: list[list[Tensor]],
    fake_features: list[list[Tensor]],
) -> Tensor:
    losses = [
        F.l1_loss(fake, real)
        for real_group, fake_group in zip(real_features, fake_features, strict=True)
        for real, fake in zip(real_group, fake_group, strict=True)
    ]
    return torch.stack(losses).sum() * 2


def discriminator_step_loss(
    real_logits: list[Tensor],
    fake_logits: list[Tensor],
) -> Tensor:
    return discriminator_lsgan_loss(
        real_logits,
        fake_logits,
    ) + discriminator_tprls_loss(
        real_logits,
        fake_logits,
    )


def generator_step_loss(
    real_logits: list[Tensor],
    fake_logits: list[Tensor],
    real_features: list[list[Tensor]],
    fake_features: list[list[Tensor]],
    period_count: int,
) -> GeneratorAdversarialLoss:
    adversarial = generator_lsgan_loss(
        fake_logits
    ) + generator_tprls_loss(
        real_logits,
        fake_logits,
    )
    feature_period = feature_matching_loss(
        real_features[:period_count],
        fake_features[:period_count],
    )
    feature_resolution = feature_matching_loss(
        real_features[period_count:],
        fake_features[period_count:],
    )
    return GeneratorAdversarialLoss(
        adversarial=adversarial,
        feature_period=feature_period,
        feature_resolution=feature_resolution,
        feature_matching=feature_period + feature_resolution,
    )


__all__ = [
    "GeneratorAdversarialLoss",
    "StyleTTSDiscriminatorOutput",
    "discriminator_lsgan_loss",
    "discriminator_step_loss",
    "discriminator_tprls_loss",
    "feature_matching_loss",
    "generator_lsgan_loss",
    "generator_step_loss",
    "generator_tprls_loss",
]
