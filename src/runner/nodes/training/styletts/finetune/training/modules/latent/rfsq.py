from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class ProsodyLatents:
    continuous: Tensor
    quantized: Tensor
    continuous_style: Tensor
    quantized_style: Tensor
    indices: Tensor
    quantization_error: Tensor


class ResidualFiniteScalarQuantizer(nn.Module):
    """Bounded continuous prosody space with a fixed residual scalar grid."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        stages: int,
        levels: int,
        stage_dropout: bool,
    ) -> None:
        super().__init__()
        if min(input_dim, latent_dim, stages) < 1:
            raise ValueError("residual FSQ dimensions and stages must be positive")
        if levels < 2:
            raise ValueError("residual FSQ requires at least two levels")
        self.latent_dim = latent_dim
        self.num_stages = stages
        self.levels = levels
        self.stage_dropout = stage_dropout
        self.to_latent = nn.Conv1d(input_dim, latent_dim, kernel_size=1)
        self.from_latent = nn.Conv1d(latent_dim, input_dim, kernel_size=1)
        stage_ids = torch.arange(stages, dtype=torch.float32)
        self.register_buffer(
            "stage_scales",
            (levels - 1.0) ** -stage_ids,
            persistent=True,
        )

    def forward(self, values: Tensor) -> ProsodyLatents:
        continuous = torch.tanh(self.to_latent(values))
        active_stages = self._active_stage_counts(
            continuous.size(0),
            continuous.device,
        )
        quantized, indices = self.quantize(continuous, active_stages)
        return ProsodyLatents(
            continuous=continuous,
            quantized=quantized,
            continuous_style=self.from_latent(continuous),
            quantized_style=self.from_latent(quantized),
            indices=indices,
            quantization_error=F.mse_loss(
                continuous.detach(),
                quantized.detach(),
            ),
        )

    def quantize(
        self,
        continuous: Tensor,
        active_stages: Tensor,
    ) -> tuple[Tensor, Tensor]:
        residual = continuous
        hard_quantized = torch.zeros_like(continuous)
        indices = []
        for stage, scale in enumerate(self.stage_scales):
            normalized = (residual / scale).clamp(-1.0, 1.0)
            stage_indices = torch.round(
                (normalized + 1.0) * (self.levels - 1) / 2
            ).long()
            stage_value = (
                stage_indices.to(continuous.dtype) * 2 / (self.levels - 1) - 1
            ) * scale
            enabled = stage < active_stages
            stage_value = stage_value * enabled[:, None, None]
            stage_indices = stage_indices.masked_fill(~enabled[:, None, None], -1)
            hard_quantized = hard_quantized + stage_value
            residual = residual - stage_value
            indices.append(stage_indices)
        hard_quantized = hard_quantized.clamp(-1.0, 1.0)
        straight_through = continuous + (hard_quantized - continuous).detach()
        return straight_through, torch.stack(indices, dim=1)

    def decode_continuous(self, latent: Tensor) -> Tensor:
        return self.from_latent(latent.clamp(-1.0, 1.0))

    def decode_quantized(self, latent: Tensor) -> Tensor:
        active_stages = torch.full(
            (latent.size(0),),
            self.num_stages,
            device=latent.device,
            dtype=torch.long,
        )
        quantized, _ = self.quantize(
            latent.clamp(-1.0, 1.0),
            active_stages,
        )
        return self.from_latent(quantized)

    def _active_stage_counts(self, batch: int, device: torch.device) -> Tensor:
        if not self.training or not self.stage_dropout:
            return torch.full((batch,), self.num_stages, device=device, dtype=torch.long)
        return torch.randint(1, self.num_stages + 1, (batch,), device=device)
