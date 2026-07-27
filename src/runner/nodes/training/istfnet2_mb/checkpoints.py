from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_checkpoint(
    output_dir: Path,
    step: int,
    epoch: int,
    generator: nn.Module,
    mpd: nn.Module,
    mrsd: nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
) -> None:
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "generator": generator.state_dict(),
            "mpd": mpd.state_dict(),
            "mrsd": mrsd.state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
        },
        output_dir / f"checkpoint_{step:09d}.pt",
    )
