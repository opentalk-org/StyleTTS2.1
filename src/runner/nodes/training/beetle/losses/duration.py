import torch
from torch import Tensor


def duration_flow_loss(negative_log_likelihood: Tensor, mask: Tensor) -> Tensor:
    if negative_log_likelihood.ndim != 1 or mask.ndim != 3 or mask.shape[1] != 1:
        raise ValueError("duration loss requires [B] likelihood and [B,1,T] mask")
    if negative_log_likelihood.shape[0] != mask.shape[0]:
        raise ValueError("duration likelihood and mask batch sizes must match")
    valid_tokens = mask.sum(dim=(1, 2))
    if torch.any(valid_tokens == 0):
        raise ValueError("duration loss requires a valid token in every item")
    normalized = negative_log_likelihood / valid_tokens.to(
        dtype=negative_log_likelihood.dtype
    )
    return normalized.mean()
