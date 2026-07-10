from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MosLoss:
    total: torch.Tensor
    mos: torch.Tensor
    comparison: torch.Tensor


def mos_pair_loss(
    prediction_a: torch.Tensor,
    prediction_b: torch.Tensor,
    score_a: torch.Tensor,
    score_b: torch.Tensor,
    preferred_sign: torch.Tensor,
    comparison_weight: float,
) -> MosLoss:
    mos = F.mse_loss(prediction_a, score_a) + F.mse_loss(prediction_b, score_b)
    comparison = F.softplus(-preferred_sign * (prediction_a - prediction_b)).mean()
    return MosLoss(
        total=mos + comparison_weight * comparison,
        mos=mos,
        comparison=comparison,
    )
