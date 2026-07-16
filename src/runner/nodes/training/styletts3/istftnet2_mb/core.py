from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import FrequencyUpsample2D, LRELU_SLOPE, MRF1DConcat, ShuffleBlock2D
from .synthesis import FREQUENCY_BINS, ISTFT_HOP, MultiBandISTFT, SUBBANDS


INPUT_CHANNELS = 128
TEMPORAL_CHANNELS = 64
TEMPORAL_RATE = 5
OUTPUT_HOP = TEMPORAL_RATE * ISTFT_HOP * SUBBANDS


class ISTFTNet2MBCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temporal_up = nn.ConvTranspose1d(
            INPUT_CHANNELS,
            TEMPORAL_CHANNELS,
            10,
            TEMPORAL_RATE,
            padding=3,
            output_padding=1,
        )
        self.mrf = MRF1DConcat(TEMPORAL_CHANNELS)
        concatenated_channels = TEMPORAL_CHANNELS * 3
        self.few_frequency_bins = 4
        self.entry_2d = nn.Conv2d(
            concatenated_channels // self.few_frequency_bins,
            TEMPORAL_CHANNELS,
            3,
            padding=1,
        )
        self.shuffles = nn.ModuleList(ShuffleBlock2D(TEMPORAL_CHANNELS) for _ in range(3))
        self.frequency_upsamples = nn.ModuleList(
            [
                FrequencyUpsample2D(64, 32, 4),
                FrequencyUpsample2D(32, 16, 4),
                FrequencyUpsample2D(16, SUBBANDS * 2, 3),
            ]
        )
        self.istft = MultiBandISTFT()

    def upsample(self, features: torch.Tensor) -> torch.Tensor:
        assert features.ndim == 3, f"expected rank-3 frame features, got rank {features.ndim}"
        assert features.shape[1] == INPUT_CHANNELS, (
            f"expected {INPUT_CHANNELS} feature channels, got {features.shape[1]}"
        )
        temporal = self.temporal_up(F.leaky_relu(features, LRELU_SLOPE))
        expected_frames = features.shape[-1] * TEMPORAL_RATE
        assert temporal.shape[-1] == expected_frames, (
            f"temporal upsampling produced {temporal.shape[-1]} frames; expected {expected_frames}"
        )
        return temporal

    def subband_spectrogram(self, temporal: torch.Tensor) -> torch.Tensor:
        assert temporal.ndim == 3, f"expected rank-3 temporal features, got rank {temporal.ndim}"
        assert temporal.shape[1] == TEMPORAL_CHANNELS, (
            f"expected {TEMPORAL_CHANNELS} temporal channels, got {temporal.shape[1]}"
        )
        features = self.mrf(temporal)
        batch, channels, frames = features.shape
        features = features.view(
            batch,
            channels // self.few_frequency_bins,
            self.few_frequency_bins,
            frames,
        )
        features = self.entry_2d(features)
        for shuffle in self.shuffles:
            features = shuffle(features)
        for upsample in self.frequency_upsamples:
            features = upsample(features)
        assert features.shape[2] == FREQUENCY_BINS, (
            f"frequency path produced {features.shape[2]} bins; expected {FREQUENCY_BINS}"
        )
        return features

    def synthesize(self, temporal: torch.Tensor) -> torch.Tensor:
        return self.istft(self.subband_spectrogram(temporal))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        waveform = self.synthesize(self.upsample(features))
        expected_length = features.shape[-1] * OUTPUT_HOP
        assert waveform.shape[-1] == expected_length, (
            f"core produced {waveform.shape[-1]} samples; expected {expected_length}"
        )
        return waveform

