from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.optim import AdamW


def build_asr_optimizer(
    params: list[nn.Parameter],
    *,
    learning_rate: float,
    optimizer_params: dict[str, Any],
    epochs: int,
    steps_per_epoch: int,
    scheduler_params: dict[str, Any],
) -> tuple[AdamW, torch.optim.lr_scheduler.OneCycleLR]:
    wd = float(optimizer_params.get("weight_decay", 5e-4))
    betas_raw = optimizer_params.get("betas", (0.9, 0.98))
    if isinstance(betas_raw, list) and len(betas_raw) == 2:
        betas = (float(betas_raw[0]), float(betas_raw[1]))
    elif isinstance(betas_raw, tuple):
        betas = (float(betas_raw[0]), float(betas_raw[1]))
    else:
        betas = (0.9, 0.98)
    eps = float(optimizer_params.get("eps", 1e-9))
    optimizer = AdamW(
        params,
        lr=learning_rate,
        weight_decay=wd,
        betas=betas,
        eps=eps,
    )
    pct_start = float(scheduler_params.get("pct_start", 0.0))
    final_div = float(scheduler_params.get("final_div_factor", 5.0))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
        final_div_factor=final_div,
    )
    return optimizer, scheduler
