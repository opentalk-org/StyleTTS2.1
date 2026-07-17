from __future__ import annotations

import torch
import torch.nn as nn


def build_adam_optimizers(
    generator: nn.Module,
    discriminator: nn.Module,
    generator_learning_rate: float,
    discriminator_learning_rate: float,
    betas: tuple[float, float],
) -> tuple[torch.optim.Adam, torch.optim.Adam]:
    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=generator_learning_rate,
        betas=betas,
        fused=True,
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=discriminator_learning_rate,
        betas=betas,
        fused=True,
    )
    return generator_optimizer, discriminator_optimizer
