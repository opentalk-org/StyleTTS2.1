from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ...config.architecture import FeatureConfig, PosteriorEncoderConfig
from .convolution import DilatedResidualStack
from .pitch import JDCNet


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


@dataclass(frozen=True)
class AcousticFeatures:
    f0: Tensor
    n: Tensor


class FeatureLinear(nn.Module):
    def __init__(self, config: FeatureConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Conv1d(config.latent_channels, 2, 1)

    def forward(
        self,
        latent: Tensor,
        latent_mask: Tensor,
        frame_mask: Tensor,
    ) -> AcousticFeatures:
        numeric_latent_mask = latent_mask.to(dtype=latent.dtype)
        projected = self.projection(latent * numeric_latent_mask) * numeric_latent_mask
        interpolated = F.interpolate(
            projected,
            scale_factor=self.config.upsample_rate,
            mode="linear",
            align_corners=False,
        )
        numeric_frame_mask = frame_mask[:, 0].to(dtype=latent.dtype)
        f0 = F.softplus(interpolated[:, 0]) * numeric_frame_mask
        n = interpolated[:, 1] * numeric_frame_mask
        return AcousticFeatures(f0, n)


class F0Extractor(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "F0Extractor":
        model = JDCNet(num_class=1, seq_len=192)
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        model.load_state_dict(payload["net"])
        return cls(model)

    def train(self, mode: bool = True) -> "F0Extractor":
        del mode
        super().train(False)
        return self

    def forward(self, mel: Tensor, mask: Tensor) -> Tensor:
        with torch.no_grad():
            pitch, _, _ = self.model(mel.unsqueeze(1))
        if pitch.ndim == 3 and pitch.shape[-1] == 1:
            pitch = pitch.squeeze(-1)
        if pitch.shape != (mel.shape[0], mel.shape[-1]):
            raise ValueError(
                f"F0 extractor returned {tuple(pitch.shape)} for mel {tuple(mel.shape)}"
            )
        return pitch * mask[:, 0].to(dtype=pitch.dtype)
