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
    if noise.ndim != 3 or mask.shape != (noise.shape[0], 1, noise.shape[2]):
        raise ValueError("flow integration requires [B,C,T] noise and [B,1,T] mask")
    if steps <= 0 or steps & (steps - 1):
        raise ValueError("flow integration steps must be a positive power of two")
    numeric_mask = mask.to(dtype=noise.dtype)
    state = noise * numeric_mask
    step_size = 1.0 / steps
    step = torch.full_like(mask, step_size, dtype=noise.dtype) * numeric_mask
    for index in range(steps):
        time = torch.full_like(mask, index * step_size, dtype=noise.dtype)
        time = time * numeric_mask
        velocity = model(state, time, step, conditions, mask)
        state = (state + step_size * velocity) * numeric_mask
    return state
