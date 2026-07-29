import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from ....config import AlphaFlowConfig


@dataclass(frozen=True)
class FlowTrainingSample:
    state: Tensor
    intermediate_state: Tensor
    start_time: Tensor
    end_time: Tensor
    intermediate_time: Tensor
    velocity: Tensor
    noise: Tensor
    alpha: float
    flow_matching_count: int


def alpha_flow_ratio(step: int, config: AlphaFlowConfig) -> float:
    if step < config.schedule_start_step:
        return 1.0
    if step > config.schedule_end_step:
        return 0.0
    midpoint = (config.schedule_start_step + config.schedule_end_step) / 2
    width = config.schedule_end_step - config.schedule_start_step
    progress = (step - midpoint) / width
    alpha = 1 - 1 / (1 + math.exp(-progress * config.schedule_gamma))
    if alpha > 1 - config.clamp_value:
        return 1.0
    if alpha < config.clamp_value:
        return 0.0
    return alpha


def sample_flow_training_case(
    latent: Tensor,
    mask: Tensor,
    config: AlphaFlowConfig,
    step: int,
    generator: torch.Generator,
) -> FlowTrainingSample:
    numeric_mask = mask.to(dtype=latent.dtype)
    noise = torch.randn(
        latent.shape,
        dtype=latent.dtype,
        device=latent.device,
        generator=generator,
    ) * numeric_mask
    batch_size = latent.shape[0]
    flow_matching_count = int(batch_size * config.flow_matching_ratio)
    trajectory_count = batch_size - flow_matching_count
    flow_time = _logit_normal(
        flow_matching_count,
        config,
        latent,
        generator,
    )
    first = _logit_normal(trajectory_count, config, latent, generator)
    second = _logit_normal(trajectory_count, config, latent, generator)
    trajectory_start = torch.minimum(first, second)
    trajectory_end = torch.maximum(first, second)
    start = torch.cat((flow_time, trajectory_start))
    end = torch.cat((flow_time, trajectory_end))
    alpha = alpha_flow_ratio(step, config)
    intermediate = start + alpha * (end - start)
    start_time = _expand_time(start, mask, latent.dtype)
    end_time = _expand_time(end, mask, latent.dtype)
    intermediate_time = _expand_time(intermediate, mask, latent.dtype)
    velocity = (latent - noise) * numeric_mask
    state = ((1 - start_time) * noise + start_time * latent) * numeric_mask
    intermediate_state = (
        state + (intermediate_time - start_time) * velocity
    ) * numeric_mask
    return FlowTrainingSample(
        state=state,
        intermediate_state=intermediate_state,
        start_time=start_time,
        end_time=end_time,
        intermediate_time=intermediate_time,
        velocity=velocity,
        noise=noise,
        alpha=alpha,
        flow_matching_count=flow_matching_count,
    )


def patch_mask(mask: Tensor, patch_size: int) -> Tensor:
    padding = (-mask.shape[-1]) % patch_size
    padded = F.pad(mask, (0, padding), value=False)
    return padded.view(*mask.shape[:-1], -1, patch_size).any(dim=-1)


def _logit_normal(
    count: int,
    config: AlphaFlowConfig,
    reference: Tensor,
    generator: torch.Generator,
) -> Tensor:
    normal = torch.randn(
        count,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )
    return torch.sigmoid(normal * config.time_scale + config.time_location)


def _expand_time(values: Tensor, mask: Tensor, dtype: torch.dtype) -> Tensor:
    return values.view(-1, 1, 1).expand_as(mask).to(dtype=dtype) * mask
