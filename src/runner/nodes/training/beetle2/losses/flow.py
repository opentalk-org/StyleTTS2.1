import torch
from torch import Tensor


def alpha_flow_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    alpha: float,
    flow_matching_count: int,
) -> Tensor:
    per_item = flow_mse_per_item(prediction, target, mask)
    trajectory_weight = 1.0 if alpha == 0.0 else alpha
    objective_weight = torch.full_like(per_item, trajectory_weight)
    objective_weight[:flow_matching_count] = 1.0
    return (objective_weight * per_item).mean()


def flow_mse(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    return flow_mse_per_item(prediction, target, mask).mean()


def flow_inner_product(
    first: Tensor,
    second: Tensor,
    mask: Tensor,
) -> Tensor:
    valid_elements = mask.sum() * first.shape[1]
    torch._assert_async(valid_elements > 0, "flow metric requires a valid token")
    product = first * second * mask
    return product.sum() / valid_elements.to(dtype=first.dtype)


def flow_mse_per_item(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    valid_elements = mask.sum(dim=(1, 2)) * prediction.shape[1]
    torch._assert_async(
        torch.all(valid_elements > 0),
        "flow loss requires a valid token in every item",
    )
    squared_error = (prediction - target).square() * mask
    return squared_error.sum(dim=(1, 2)) / valid_elements.to(
        dtype=prediction.dtype
    )
