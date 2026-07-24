from dataclasses import dataclass
import random

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from ...config.architecture import DecoderConfig
from .convolution import MaskedResidualBlock


@dataclass(frozen=True)
class DecoderOutput:
    features: Tensor
    f0: Tensor
    n: Tensor
    mask: Tensor


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        if config.decode_block_count != 4:
            raise ValueError("decoder decode_block_count must equal four")
        self.config = config
        self.latent_upsample = weight_norm(
            nn.ConvTranspose1d(
                config.latent_channels,
                config.generator_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )
        self.conditioning_projection = weight_norm(
            nn.Conv1d(2, config.generator_channels, kernel_size=3, padding=1)
        )
        self.refinement = nn.ModuleList(
            MaskedResidualBlock(
                config.generator_channels,
                kernel_size=3,
                dilation=3**index,
                dropout=config.dropout,
            )
            for index in range(config.decode_block_count)
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
        masked_latent = latent * numeric_latent_mask
        conditioning = torch.stack(
            (torch.log1p(prepared_f0), prepared_n),
            dim=1,
        )
        features = self.latent_upsample(masked_latent)
        features = features + self.conditioning_projection(
            conditioning * numeric_frame_mask
        )
        features = features * numeric_frame_mask
        for block in self.refinement:
            features = block(features, numeric_frame_mask)
        return DecoderOutput(
            features * numeric_frame_mask,
            prepared_f0 * numeric_frame_mask[:, 0],
            prepared_n * numeric_frame_mask[:, 0],
            boolean_frame_mask,
        )
