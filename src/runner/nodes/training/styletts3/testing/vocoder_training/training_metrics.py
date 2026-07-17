from __future__ import annotations

import statistics

import torch
import torch.nn as nn


def gradient_l2_norm(module: nn.Module) -> float:
    parameter_norms = [
        parameter.grad.detach().norm(2)
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    assert parameter_norms, "gradient norm requires at least one gradient"
    return float(torch.stack(parameter_norms).norm(2))


def mean_logit(logits: list[torch.Tensor]) -> float:
    assert logits, "mean logit requires at least one discriminator output"
    return float(torch.stack([logit.detach().mean() for logit in logits]).mean())


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    assert rows, "metric rows must not be empty"
    return {name: statistics.fmean(row[name] for row in rows) for name in rows[0]}
