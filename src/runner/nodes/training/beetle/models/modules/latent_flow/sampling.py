import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class FlowTrainingSample:
    state: Tensor
    time: Tensor
    step: Tensor
    step_index: Tensor
    velocity: Tensor
    noise: Tensor


def sample_flow_training_case(
    latent: Tensor,
    mask: Tensor,
    minimum_steps: int,
    base_case_probability: float,
    patch_size: int,
    generator: torch.Generator,
) -> FlowTrainingSample:
    numeric_mask = mask.to(dtype=latent.dtype)
    noise = torch.randn(
        latent.shape,
        dtype=latent.dtype,
        device=latent.device,
        generator=generator,
    ) * numeric_mask
    patches = patch_mask(mask, patch_size)
    scalar_shape = patches.shape
    is_base = (
        torch.rand(scalar_shape, device=latent.device, generator=generator)
        < base_case_probability
    )
    if 0 < base_case_probability < 1 and patches.sum() >= 2:
        flat_mask = patches.flatten()
        flat_base = is_base.flatten()
        valid_indices = torch.nonzero(flat_mask, as_tuple=False).flatten()
        valid_base = flat_base.masked_select(flat_mask)
        if not torch.any(valid_base):
            flat_base[valid_indices[0]] = True
        if torch.all(valid_base):
            flat_base[valid_indices[-1]] = False
        is_base = flat_base.view_as(is_base)
    maximum_index = int(math.log2(minimum_steps))
    shortcut_index = torch.randint(
        0,
        maximum_index,
        scalar_shape,
        device=latent.device,
        generator=generator,
    )
    step_index = torch.where(
        is_base,
        torch.full_like(shortcut_index, maximum_index),
        shortcut_index,
    )
    step = torch.pow(2.0, -step_index.to(dtype=latent.dtype))
    start_count = torch.pow(2.0, step_index.to(dtype=latent.dtype))
    start_index = torch.floor(
        torch.rand(
            scalar_shape,
            dtype=latent.dtype,
            device=latent.device,
            generator=generator,
        )
        * start_count
    )
    time = expand_patches(start_index * step, patch_size, mask.shape[-1])
    step = expand_patches(step, patch_size, mask.shape[-1]) * numeric_mask
    time = time * numeric_mask
    step_index = expand_patches(step_index, patch_size, mask.shape[-1]) * mask
    state = ((1 - time) * noise + time * latent) * numeric_mask
    velocity = (latent - noise) * numeric_mask
    return FlowTrainingSample(state, time, step, step_index, velocity, noise)


def patch_mask(mask: Tensor, patch_size: int) -> Tensor:
    padding = (-mask.shape[-1]) % patch_size
    padded = F.pad(mask, (0, padding), value=False)
    return padded.view(*mask.shape[:-1], -1, patch_size).any(dim=-1)


def expand_patches(values: Tensor, patch_size: int, frame_count: int) -> Tensor:
    return values.repeat_interleave(patch_size, dim=-1)[..., :frame_count]
