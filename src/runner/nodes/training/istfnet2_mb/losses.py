from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def discriminator_loss(
    real_logits: list[Tensor],
    fake_logits: list[Tensor],
) -> Tensor:
    losses = [
        (1 - real).square().mean() + fake.square().mean()
        for real, fake in zip(real_logits, fake_logits, strict=True)
    ]
    return torch.stack(losses).sum()


def generator_loss(fake_logits: list[Tensor]) -> Tensor:
    losses = [(1 - fake).square().mean() for fake in fake_logits]
    return torch.stack(losses).sum()


def feature_loss(
    real_maps: list[list[Tensor]],
    fake_maps: list[list[Tensor]],
) -> Tensor:
    losses = [
        F.l1_loss(fake, real)
        for real_group, fake_group in zip(real_maps, fake_maps, strict=True)
        for real, fake in zip(real_group, fake_group, strict=True)
    ]
    return torch.stack(losses).sum() * 2


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
