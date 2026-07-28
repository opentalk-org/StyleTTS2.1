import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class FlowTrainingSample:
    state: Tensor
    time: Tensor
    step_size: Tensor
    step_level: Tensor
    velocity: Tensor
    noise: Tensor


def sample_flow_training_case(
    latent: Tensor,
    noise: Tensor,
    mask: Tensor,
    minimum_steps: int,
    generator: torch.Generator,
) -> FlowTrainingSample:
    numeric_mask = mask.to(dtype=latent.dtype)
    noise = noise * numeric_mask
    item_shape = (latent.shape[0], 1, 1)
    step_level = torch.full(
        item_shape,
        int(math.log2(minimum_steps)),
        dtype=torch.long,
        device=latent.device,
    )
    step_size = torch.pow(2.0, -step_level.to(dtype=latent.dtype))
    start_count = torch.pow(2.0, step_level.to(dtype=latent.dtype))
    start_index = torch.floor(
        torch.rand(
            item_shape,
            dtype=latent.dtype,
            device=latent.device,
            generator=generator,
        )
        * start_count
    )
    time = (start_index * step_size).expand_as(mask) * numeric_mask
    step_size = step_size.expand_as(mask) * numeric_mask
    step_level = step_level.expand_as(mask) * mask
    noise_scale = 1 - 1e-5
    state = ((1 - noise_scale * time) * noise + time * latent) * numeric_mask
    velocity = (latent - noise_scale * noise) * numeric_mask
    return FlowTrainingSample(
        state,
        time,
        step_size,
        step_level,
        velocity,
        noise,
    )


def patch_mask(mask: Tensor, patch_size: int) -> Tensor:
    padding = (-mask.shape[-1]) % patch_size
    padded = F.pad(mask, (0, padding), value=False)
    return padded.view(*mask.shape[:-1], -1, patch_size).any(dim=-1)
