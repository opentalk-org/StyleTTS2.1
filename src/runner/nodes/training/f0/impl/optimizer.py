from __future__ import annotations

import torch
from torch import nn
from torch.optim import AdamW


def build_f0_optimizer(
    params: nn.ParameterList | list,
    *,
    lr: float,
    weight_decay: float,
    epochs: int,
    steps_per_epoch: int,
    pct_start: float = 0.0,
) -> tuple[AdamW, torch.optim.lr_scheduler.OneCycleLR]:
    optimizer = AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
        final_div_factor=5.0,
    )
    return optimizer, scheduler
