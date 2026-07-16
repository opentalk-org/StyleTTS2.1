from __future__ import annotations

import torch
import torch.nn as nn

from runner.nodes.training.styletts.finetune.training.modules.decoder_blocks import AdaINResBlock1, DecoderBackbone
from runner.nodes.training.styletts.finetune.training.modules.utils import checkpoint_with_mixed_precision
from .core import INPUT_CHANNELS, ISTFTNet2MBCore, TEMPORAL_CHANNELS
from .source import HarmonicSourceFeatures, SOURCE_CHANNELS


class StyleTTSISTFTNet2MBGenerator(nn.Module):
    def __init__(
        self,
        input_channels: int = 512,
        style_dim: int = 128,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.input_projection = nn.Conv1d(input_channels, INPUT_CHANNELS, 7, padding=3)
        self.core = ISTFTNet2MBCore()
        self.harmonic_features = HarmonicSourceFeatures()
        self.source_projection = nn.Conv1d(SOURCE_CHANNELS, TEMPORAL_CHANNELS, 1)
        self.source_residual = AdaINResBlock1(
            TEMPORAL_CHANNELS,
            kernel_size=11,
            dilation=(1, 3, 5),
            style_dim=style_dim,
        )

    def forward(self, features: torch.Tensor, f0: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        assert features.ndim == 3, f"expected rank-3 decoder features, got rank {features.ndim}"
        assert f0.ndim == 2, f"expected rank-2 F0, got rank {f0.ndim}"
        assert features.shape[-1] == f0.shape[-1], (
            f"decoder features have {features.shape[-1]} frames but F0 has {f0.shape[-1]}"
        )
        temporal = self.core.upsample(self.input_projection(features))
        source = self.source_projection(self.harmonic_features(f0))
        source = self.source_residual(source, style)
        assert source.shape == temporal.shape, (
            f"source features have shape {source.shape}; expected {temporal.shape}"
        )
        combined = temporal + source
        if self.gradient_checkpointing and self.training:
            return checkpoint_with_mixed_precision(self.core.synthesize, combined)
        return self.core.synthesize(combined)


class StyleTTSISTFTNet2MBDecoder(DecoderBackbone):
    def __init__(
        self,
        dim_in: int = 512,
        style_dim: int = 128,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__(dim_in=dim_in, style_dim=style_dim)
        self.generator = StyleTTSISTFTNet2MBGenerator(
            input_channels=512,
            style_dim=style_dim,
            gradient_checkpointing=gradient_checkpointing,
        )

    def forward(
        self,
        asr: torch.Tensor,
        f0_curve: torch.Tensor,
        noise: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        features, prepared_f0 = super().forward(asr, f0_curve, noise, style)
        assert features.shape[-1] == prepared_f0.shape[-1], (
            f"decoder backbone produced {features.shape[-1]} frames but F0 has "
            f"{prepared_f0.shape[-1]}"
        )
        return self.generator(features, prepared_f0, style)

