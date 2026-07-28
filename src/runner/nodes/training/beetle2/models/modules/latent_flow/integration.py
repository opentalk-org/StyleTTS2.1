import math

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
    step_level = torch.full_like(
        mask,
        math.log2(steps),
        dtype=noise.dtype,
    ) * numeric_mask
    for index in range(steps):
        time = torch.full_like(mask, index * step_size, dtype=noise.dtype)
        time = time * numeric_mask
        velocity = model(state, time, step_level, conditions, mask)
        state = (state + step_size * velocity) * numeric_mask
    return state
