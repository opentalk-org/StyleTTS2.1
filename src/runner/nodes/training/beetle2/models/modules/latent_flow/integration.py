import torch
from torch import Tensor, nn

from ..conditioning import ProjectedConditions


def integrate_latent_flow(
    model: nn.Module,
    noise: Tensor,
    conditions: ProjectedConditions,
    mask: Tensor,
    steps: int,
) -> Tensor:
    numeric_mask = mask.to(dtype=noise.dtype)
    state = noise * numeric_mask
    step_size = 1.0 / steps
    for index in range(steps):
        start_time = torch.full_like(
            mask,
            index * step_size,
            dtype=noise.dtype,
        ) * numeric_mask
        end_time = torch.full_like(
            mask,
            (index + 1) * step_size,
            dtype=noise.dtype,
        ) * numeric_mask
        mean_velocity = model(
            state,
            start_time,
            end_time,
            conditions,
            mask,
        )
        state = (state + step_size * mean_velocity) * numeric_mask
    return state
