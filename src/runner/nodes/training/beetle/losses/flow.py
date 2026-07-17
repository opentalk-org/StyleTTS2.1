import torch
from torch import Tensor, nn

from ..models.modules.conditioning import ProjectedConditions
from ..models.modules.latent_flow.model import FlowTrainingSample


def base_flow_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("flow prediction and target must have equal [B,C,T] shapes")
    if mask.shape != (prediction.shape[0], 1, prediction.shape[2]):
        raise ValueError("flow loss mask must have shape [B,1,T]")
    valid_elements = mask.sum() * prediction.shape[1]
    if valid_elements == 0:
        raise ValueError("flow loss requires a valid token")
    squared_error = (prediction - target).square() * mask
    return squared_error.sum() / valid_elements.to(dtype=prediction.dtype)


def shortcut_loss(
    prediction: Tensor,
    ema_model: nn.Module,
    sample: FlowTrainingSample,
    conditions: ProjectedConditions,
    model_mask: Tensor,
    loss_mask: Tensor,
    minimum_steps: int,
) -> Tensor:
    if minimum_steps <= 1 or minimum_steps & (minimum_steps - 1):
        raise ValueError("minimum_steps must be a power of two above one")
    half_step = sample.step / 2
    smallest_half_step = 1.0 / minimum_steps
    query_step = torch.where(
        half_step <= smallest_half_step,
        torch.zeros_like(half_step),
        half_step,
    )
    with torch.no_grad():
        first_velocity = ema_model(
            sample.state,
            sample.time,
            query_step,
            conditions,
            model_mask,
        )
        midpoint = (sample.state + half_step * first_velocity) * model_mask
        second_velocity = ema_model(
            midpoint,
            sample.time + half_step,
            query_step,
            conditions,
            model_mask,
        )
        shortcut_target = (first_velocity + second_velocity) / 2
    return base_flow_loss(prediction, shortcut_target.detach(), loss_mask)
