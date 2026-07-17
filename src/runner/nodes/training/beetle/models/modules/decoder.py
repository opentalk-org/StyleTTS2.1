from dataclasses import dataclass
import random

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from ...config.architecture import DecoderConfig
from .convolution import DecoderResidualBlock


@dataclass(frozen=True)
class DecoderOutput:
    features: Tensor
    f0: Tensor
    mask: Tensor


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        if config.decode_block_count != 4:
            raise ValueError("decoder decode_block_count must equal four")
        self.config = config
        self.f0_projection = weight_norm(
            nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1)
        )
        self.n_projection = weight_norm(
            nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1)
        )
        self.latent_residual = weight_norm(
            nn.Conv1d(config.latent_channels, config.residual_channels, 1)
        )
        self.encode = DecoderResidualBlock(
            config.latent_channels + 2,
            config.hidden_channels,
            config.dropout,
        )
        decode_input_channels = (
            config.hidden_channels + config.residual_channels + 2
        )
        self.decode = nn.ModuleList(
            [
                DecoderResidualBlock(
                    decode_input_channels,
                    config.hidden_channels,
                    config.dropout,
                )
                for _ in range(config.decode_block_count - 1)
            ]
            + [
                DecoderResidualBlock(
                    decode_input_channels,
                    config.generator_channels,
                    config.dropout,
                    upsample=True,
                )
            ]
        )

    def _smooth_signal(self, signal: Tensor, mask: Tensor, kernel_size: int) -> Tensor:
        numeric_mask = mask[:, 0].to(dtype=signal.dtype)
        masked = signal * numeric_mask
        if kernel_size == 0:
            return masked
        kernel = torch.ones(
            1,
            1,
            kernel_size,
            device=signal.device,
            dtype=signal.dtype,
        )
        smoothed = F.conv1d(
            masked.unsqueeze(1),
            kernel,
            padding=kernel_size // 2,
        ).squeeze(1)
        return smoothed * (numeric_mask / kernel_size)

    def _prepare_inputs(
        self,
        f0: Tensor,
        n: Tensor,
        frame_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        numeric_mask = frame_mask[:, 0].to(dtype=f0.dtype)
        if not self.training:
            return f0 * numeric_mask, n * numeric_mask
        f0_kernel = random.choice(self.config.f0_smoothing_kernel_sizes)
        n_kernel = random.choice(self.config.n_smoothing_kernel_sizes)
        return (
            self._smooth_signal(f0, frame_mask, f0_kernel),
            self._smooth_signal(n, frame_mask, n_kernel),
        )

    def forward(
        self,
        latent: Tensor,
        f0: Tensor,
        n: Tensor,
        latent_mask: Tensor,
        frame_mask: Tensor,
    ) -> DecoderOutput:
        if latent.ndim != 3 or latent.shape[1] != self.config.latent_channels:
            raise ValueError("decoder latent must have configured [B,C,L] geometry")
        batch_size, _, latent_frames = latent.shape
        frame_frames = latent_frames * 2
        if f0.shape != (batch_size, frame_frames):
            raise ValueError("decoder F0 must have shape [B,2L]")
        if n.shape != (batch_size, frame_frames):
            raise ValueError("decoder N must have shape [B,2L]")
        if latent_mask.shape != (batch_size, 1, latent_frames):
            raise ValueError("decoder latent mask must have shape [B,1,L]")
        if frame_mask.shape != (batch_size, 1, frame_frames):
            raise ValueError("decoder frame mask must have shape [B,1,2L]")

        boolean_latent_mask = latent_mask.to(dtype=torch.bool)
        boolean_frame_mask = frame_mask.to(dtype=torch.bool)
        numeric_latent_mask = boolean_latent_mask.to(dtype=latent.dtype)
        numeric_frame_mask = boolean_frame_mask.to(dtype=latent.dtype)
        prepared_f0, prepared_n = self._prepare_inputs(
            f0,
            n,
            boolean_frame_mask,
        )
        projected_f0 = self.f0_projection(prepared_f0.unsqueeze(1))
        projected_n = self.n_projection(prepared_n.unsqueeze(1))
        projected_f0 = projected_f0 * numeric_latent_mask
        projected_n = projected_n * numeric_latent_mask
        masked_latent = latent * numeric_latent_mask
        features = self.encode(
            torch.cat((masked_latent, projected_f0, projected_n), dim=1),
            boolean_latent_mask,
            boolean_latent_mask,
        )
        latent_residual = self.latent_residual(masked_latent) * numeric_latent_mask
        conditioning = (latent_residual, projected_f0, projected_n)
        for block in self.decode[:-1]:
            features = block(
                torch.cat((features, *conditioning), dim=1),
                boolean_latent_mask,
                boolean_latent_mask,
            )
        features = self.decode[-1](
            torch.cat((features, *conditioning), dim=1),
            boolean_latent_mask,
            boolean_frame_mask,
        )
        return DecoderOutput(
            features * numeric_frame_mask,
            prepared_f0 * numeric_frame_mask[:, 0],
            boolean_frame_mask,
        )
