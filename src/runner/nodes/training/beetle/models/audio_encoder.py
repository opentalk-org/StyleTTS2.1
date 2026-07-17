from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..config.architecture import PosteriorEncoderConfig
from .modules.convolution import DilatedResidualStack


@dataclass(frozen=True)
class AudioPosterior:
    mean: Tensor
    log_scale: Tensor
    latent: Tensor
    mask: Tensor


class AudioEncoder(nn.Module):
    def __init__(self, config: PosteriorEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Conv1d(
            config.mel_channels,
            config.hidden_channels,
            config.downsample_kernel_size,
            stride=config.downsample_rate,
            padding=1,
        )
        self.stack = DilatedResidualStack(
            config.hidden_channels,
            config.kernel_size,
            config.dilation_cycle,
            config.cycles,
            config.dropout,
        )
        self.posterior_projection = nn.Conv1d(
            config.hidden_channels,
            config.latent_channels * 2,
            1,
        )

    def forward(
        self,
        mel: Tensor,
        mask: Tensor,
        generator: torch.Generator,
    ) -> AudioPosterior:
        if mel.ndim != 3 or mel.shape[1] != self.config.mel_channels:
            raise ValueError("mel must have configured [B,M,T] geometry")
        if mask.shape != (mel.shape[0], 1, mel.shape[2]):
            raise ValueError("mel mask must have shape [B,1,T]")
        if mel.shape[-1] % self.config.downsample_rate:
            raise ValueError("mel frames must be divisible by downsample_rate")
        full_mask = mask.to(dtype=torch.bool)
        latent_mask = full_mask.reshape(
            mel.shape[0],
            1,
            mel.shape[-1] // self.config.downsample_rate,
            self.config.downsample_rate,
        ).any(dim=-1)
        numeric_mask = latent_mask.to(dtype=mel.dtype)
        hidden = self.input_projection(mel * full_mask.to(dtype=mel.dtype)) * numeric_mask
        hidden = self.stack(hidden, numeric_mask)
        mean, log_scale = self.posterior_projection(hidden).chunk(2, dim=1)
        mean = mean * numeric_mask
        log_scale = log_scale.clamp(
            self.config.log_scale_min,
            self.config.log_scale_max,
        ) * numeric_mask
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        latent = (mean + noise * torch.exp(log_scale)) * numeric_mask
        return AudioPosterior(mean, log_scale, latent, latent_mask)
