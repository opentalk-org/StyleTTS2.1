"""Conditional velocity model following the equations in ``papers/latent-flow.md``."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ....config.architecture import LatentFlowConfig
from ..conditioning import AdaLNZero1d, ProjectedConditions
from ..conditioning_inputs import CONDITION_SOURCE_NAMES


@dataclass(frozen=True)
class FlowTrainingSample:
    state: Tensor
    time: Tensor
    step: Tensor
    velocity: Tensor
    noise: Tensor


def sample_flow_training_case(
    latent: Tensor,
    mask: Tensor,
    minimum_steps: int,
    base_case_probability: float,
    generator: torch.Generator,
) -> FlowTrainingSample:
    if latent.ndim != 3 or mask.shape != (latent.shape[0], 1, latent.shape[2]):
        raise ValueError("flow sampling requires [B,C,T] latent and [B,1,T] mask")
    if minimum_steps <= 1 or minimum_steps & (minimum_steps - 1):
        raise ValueError("minimum_steps must be a power of two above one")
    if not 0 <= base_case_probability <= 1:
        raise ValueError("base case probability must be between zero and one")
    numeric_mask = mask.to(dtype=latent.dtype)
    scalar_shape = mask.shape
    noise = (
        torch.randn(
            latent.shape,
            dtype=latent.dtype,
            device=latent.device,
            generator=generator,
        )
        * numeric_mask
    )
    is_base = (
        torch.rand(
            scalar_shape,
            device=latent.device,
            generator=generator,
        )
        < base_case_probability
    )
    if 0 < base_case_probability < 1 and mask.sum() >= 2:
        flat_mask = mask.flatten()
        flat_base = is_base.flatten()
        valid_indices = torch.nonzero(flat_mask, as_tuple=False).flatten()
        valid_base = flat_base.masked_select(flat_mask)
        if not torch.any(valid_base):
            flat_base[valid_indices[0]] = True
        if torch.all(valid_base):
            flat_base[valid_indices[-1]] = False
        is_base = flat_base.view_as(is_base)
    continuous_time = torch.rand(
        scalar_shape,
        dtype=latent.dtype,
        device=latent.device,
        generator=generator,
    )
    level = torch.randint(
        1,
        int(math.log2(minimum_steps)) + 1,
        scalar_shape,
        device=latent.device,
        generator=generator,
    )
    shortcut_step = torch.pow(2.0, level.to(dtype=latent.dtype)) / minimum_steps
    start_count = torch.round(shortcut_step.reciprocal())
    start_index = torch.floor(
        torch.rand(
            scalar_shape,
            dtype=latent.dtype,
            device=latent.device,
            generator=generator,
        )
        * start_count
    )
    shortcut_time = start_index * shortcut_step
    step = torch.where(is_base, torch.zeros_like(shortcut_step), shortcut_step)
    time = torch.where(is_base, continuous_time, shortcut_time) * numeric_mask
    step = step * numeric_mask
    state = ((1 - time) * noise + time * latent) * numeric_mask
    velocity = (latent - noise) * numeric_mask
    return FlowTrainingSample(state, time, step, velocity, noise)


class ScalarEmbedding(nn.Module):
    def __init__(self, embedding_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, embedding_channels, 1),
            nn.SiLU(),
            nn.Conv1d(embedding_channels, output_channels, 1),
        )

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        return self.layers(values) * mask.to(dtype=values.dtype)


class VelocityBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        condition_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        padding = (kernel_size * dilation - dilation) // 2
        self.adaln = AdaLNZero1d(channels, condition_channels)
        self.convolution = nn.Conv1d(
            channels,
            channels * 2,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.projection = nn.Conv1d(channels, channels, 1)

    def forward(
        self,
        features: Tensor,
        condition: Tensor,
        mask: Tensor,
    ) -> Tensor:
        numeric_mask = mask.to(dtype=features.dtype)
        hidden = self.convolution(self.adaln(features, condition, mask))
        activation, gate = hidden.chunk(2, dim=1)
        residual = self.projection(torch.tanh(activation) * torch.sigmoid(gate))
        return (features + residual) * (1 / math.sqrt(2)) * numeric_mask


class LatentFlowModel(nn.Module):
    def __init__(
        self,
        config: LatentFlowConfig,
        concat_layers: tuple[int, ...],
    ) -> None:
        super().__init__()
        if max(concat_layers) >= config.layer_count:
            raise ValueError("condition concatenation layer is outside latent flow")
        self.config = config
        self.input_projection = nn.Conv1d(
            config.latent_channels,
            config.hidden_channels,
            1,
        )
        self.time_embedding = ScalarEmbedding(
            config.time_embedding_channels,
            config.hidden_channels,
        )
        self.step_embedding = ScalarEmbedding(
            config.time_embedding_channels,
            config.hidden_channels,
        )
        self.blocks = nn.ModuleList(
            VelocityBlock(
                config.hidden_channels,
                config.condition_channels,
                config.kernel_size,
                config.dilation_cycle[index % len(config.dilation_cycle)],
            )
            for index in range(config.layer_count)
        )
        concat_channels = (
            config.hidden_channels
            + len(CONDITION_SOURCE_NAMES) * config.condition_channels
        )
        self.concat_projections = nn.ModuleDict(
            {
                str(index): nn.Conv1d(concat_channels, config.hidden_channels, 1)
                for index in concat_layers
            }
        )
        self.output_projection = nn.Conv1d(
            config.hidden_channels,
            config.latent_channels,
            1,
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        state: Tensor,
        time: Tensor,
        step: Tensor,
        conditions: ProjectedConditions,
        mask: Tensor,
    ) -> Tensor:
        scalar_shape = (state.shape[0], 1, state.shape[2])
        if state.ndim != 3 or time.shape != scalar_shape or step.shape != scalar_shape:
            raise ValueError("latent flow requires [B,C,T] state and [B,1,T] scalars")
        if mask.shape != scalar_shape:
            raise ValueError("latent flow mask must match scalar shape")
        numeric_mask = mask.to(dtype=state.dtype)
        combined = conditions.combined() * numeric_mask
        concatenated = conditions.concatenated() * numeric_mask
        features = self.input_projection(state * numeric_mask)
        features = features + self.time_embedding(time, mask)
        features = features + self.step_embedding(step, mask)
        for index, block in enumerate(self.blocks):
            key = str(index)
            if key in self.concat_projections:
                features = (
                    self.concat_projections[key](
                        torch.cat((features, concatenated), dim=1)
                    )
                    * numeric_mask
                )
            features = block(features, combined, mask)
        return self.output_projection(features) * numeric_mask
