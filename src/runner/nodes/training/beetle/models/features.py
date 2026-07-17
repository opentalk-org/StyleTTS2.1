from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config.architecture import FeatureConfig
from .modules.pitch import JDCNet


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
