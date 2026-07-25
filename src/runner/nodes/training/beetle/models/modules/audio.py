from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from ...config.architecture import FeatureConfig, PosteriorEncoderConfig
from .pitch import JDCNet


@dataclass(frozen=True)
class AudioPosterior:
    mean: Tensor
    log_scale: Tensor
    latent: Tensor
    mask: Tensor


class GatedResidualStack(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        layer_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.channels = channels
        self.dropout = dropout
        self.input_layers = nn.ModuleList(
            weight_norm(
                nn.Conv1d(
                    channels,
                    channels * 2,
                    kernel_size,
                    padding=padding,
                )
            )
            for _ in range(layer_count)
        )
        self.residual_skip_layers = nn.ModuleList(
            weight_norm(
                nn.Conv1d(
                    channels,
                    channels * (2 if index < layer_count - 1 else 1),
                    1,
                )
            )
            for index in range(layer_count)
        )

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        features = features * mask
        skip = torch.zeros_like(features)
        last_index = len(self.input_layers) - 1
        for index, (input_layer, residual_skip_layer) in enumerate(
            zip(self.input_layers, self.residual_skip_layers, strict=True)
        ):
            hidden = input_layer(features)
            tanh_part, sigmoid_part = hidden.chunk(2, dim=1)
            hidden = torch.tanh(tanh_part) * torch.sigmoid(sigmoid_part)
            hidden = F.dropout(hidden, self.dropout, self.training)
            projected = residual_skip_layer(hidden)
            if index < last_index:
                residual, layer_skip = projected.split(self.channels, dim=1)
                features = (features + residual) * mask
            else:
                layer_skip = projected
            skip = skip + layer_skip
        return skip * mask


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
        self.stack = GatedResidualStack(
            config.hidden_channels,
            config.kernel_size,
            config.layer_count,
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

    def blend(
        self,
        predicted: "AcousticFeatures",
        predicted_ratio: float,
    ) -> "AcousticFeatures":
        target_ratio = 1.0 - predicted_ratio
        return AcousticFeatures(
            self.f0 * target_ratio + predicted.f0 * predicted_ratio,
            self.n * target_ratio + predicted.n * predicted_ratio,
        )


class FeatureLinear(nn.Module):
    def __init__(self, config: FeatureConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Conv1d(config.latent_channels, 3, 1)
        nn.init.zeros_(self.projection.weight[:2])
        nn.init.zeros_(self.projection.bias[:2])

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
        f0_magnitude = F.softplus(interpolated[:, 0]) * self.config.f0_scale_hz
        voicing = torch.sigmoid(interpolated[:, 1])
        f0 = f0_magnitude * voicing * numeric_frame_mask
        n = interpolated[:, 2] * numeric_frame_mask
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
        return pitch * mask[:, 0].to(dtype=pitch.dtype)
