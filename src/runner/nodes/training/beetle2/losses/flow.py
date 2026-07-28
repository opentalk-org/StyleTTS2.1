import torch
from torch import Tensor


def base_flow_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    valid_elements = mask.sum() * prediction.shape[1]
    torch._assert_async(valid_elements > 0, "flow loss requires a valid token")
    squared_error = (prediction - target).square() * mask
    return squared_error.sum() / valid_elements.to(dtype=prediction.dtype)
