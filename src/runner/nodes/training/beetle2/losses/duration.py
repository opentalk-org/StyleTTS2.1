import torch
from torch import Tensor


def duration_flow_loss(negative_log_likelihood: Tensor, mask: Tensor) -> Tensor:
    valid_tokens = mask.sum(dim=(1, 2))
    torch._assert_async(
        torch.all(valid_tokens > 0),
        "duration loss requires a valid token in every item",
    )
    normalized = negative_log_likelihood / valid_tokens.to(
        dtype=negative_log_likelihood.dtype
    )
    return normalized.mean()
