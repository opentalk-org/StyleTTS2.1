"""Paper-reference hop-256 iSTFTNet2-MB generator."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from runner.nodes.training.styletts3.istftnet2_mb.synthesis import PQMF


LRELU_SLOPE = 0.1
MEL_CHANNELS = 80
BASE_CHANNELS = 128
TEMPORAL_CHANNELS = 64
TEMPORAL_RATE = 4
SUBBANDS = 4
ISTFT_NFFT = 64
ISTFT_HOP = 16
FREQUENCY_BINS = ISTFT_NFFT // 2 + 1
OUTPUT_HOP = TEMPORAL_RATE * ISTFT_HOP * SUBBANDS


def _normalized(module: nn.Module) -> nn.Module:
    nn.init.normal_(module.weight, mean=0.0, std=0.01)
    return weight_norm(module)


def _same_padding(kernel: int, dilation: int = 1) -> int:
    return (kernel * dilation - dilation) // 2


class PaperResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        dilations = (1, 3, 5)
        self.convs1 = nn.ModuleList(
            _normalized(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel,
                    dilation=dilation,
                    padding=_same_padding(kernel, dilation),
                )
            )
            for dilation in dilations
        )
        self.convs2 = nn.ModuleList(
            _normalized(nn.Conv1d(channels, channels, kernel, padding=_same_padding(kernel)))
            for _ in dilations
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = conv2(F.leaky_relu(conv1(F.leaky_relu(features, LRELU_SLOPE)), LRELU_SLOPE))
            features = features + residual
        return features


class PaperMRF1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(PaperResBlock1D(channels, kernel) for kernel in (3, 7, 11))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([block(features) for block in self.blocks], dim=1)


class PaperShuffleBlock2D(nn.Module):
    """Multi-band ShuffleBlock whose active expansion is C rather than 2C."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        assert channels % 2 == 0, "ShuffleBlock channels must be even"
        half = channels // 2
        self.channels = channels
        self.conv1 = _normalized(nn.Conv2d(half, channels, 3, padding=1))
        self.conv2 = _normalized(nn.Conv2d(channels, half, 3, padding=1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, frequency, time = features.shape
        shuffled = features.view(batch, 2, channels // 2, frequency, time)
        shuffled = shuffled.transpose(1, 2).reshape(batch, channels, frequency, time)
        skip, active = shuffled.chunk(2, dim=1)
        active = self.conv1(F.leaky_relu(active, LRELU_SLOPE))
        active = self.conv2(F.leaky_relu(active, LRELU_SLOPE))
        return torch.cat([skip, active], dim=1)


class PaperMultiBandISTFT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pqmf = PQMF(subbands=SUBBANDS)
        self.register_buffer("istft_window", torch.hann_window(ISTFT_NFFT))

    def _inverse_band(self, spectrogram: torch.Tensor, length: int) -> torch.Tensor:
        magnitude = torch.exp(spectrogram[:, 0].float())
        phase = torch.sin(spectrogram[:, 1].float())
        spectrum = torch.polar(magnitude, phase)
        return torch.istft(
            spectrum,
            n_fft=ISTFT_NFFT,
            hop_length=ISTFT_HOP,
            win_length=ISTFT_NFFT,
            window=self.istft_window,
            center=True,
            length=length,
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        batch, channels, frequency, frames = spectrogram.shape
        assert channels == SUBBANDS * 2, f"expected 8 spectrogram channels, got {channels}"
        assert frequency == FREQUENCY_BINS, f"expected 33 frequency bins, got {frequency}"
        bands = spectrogram.view(batch, SUBBANDS, 2, frequency, frames)
        band_length = frames * ISTFT_HOP
        subband_waveforms = torch.stack(
            [self._inverse_band(bands[:, band], band_length) for band in range(SUBBANDS)],
            dim=1,
        )
        waveform = self.pqmf.synthesis(subband_waveforms)
        expected_length = frames * ISTFT_HOP * SUBBANDS
        assert waveform.shape == (batch, 1, expected_length), (
            f"PQMF produced {waveform.shape[-1]} samples; expected {expected_length}"
        )
        return waveform


class PaperISTFTNet2MB(nn.Module):
    def __init__(
        self,
        mel_channels: int = MEL_CHANNELS,
        base_channels: int = BASE_CHANNELS,
        bands: int = SUBBANDS,
        nfft: int = ISTFT_NFFT,
    ) -> None:
        super().__init__()
        assert mel_channels == MEL_CHANNELS, "paper model uses 80 mel channels"
        assert base_channels == BASE_CHANNELS, "paper model uses 128 base channels"
        assert bands == SUBBANDS, "paper model uses four subbands"
        assert nfft == ISTFT_NFFT, "paper model uses a 64-point subband iSTFT"
        self.conv_pre = _normalized(nn.Conv1d(MEL_CHANNELS, BASE_CHANNELS, 7, padding=3))
        self.temporal_up = _normalized(
            nn.ConvTranspose1d(BASE_CHANNELS, TEMPORAL_CHANNELS, 8, stride=4, padding=2)
        )
        self.mrf = PaperMRF1D(TEMPORAL_CHANNELS)
        self.entry_2d = _normalized(nn.Conv2d(48, TEMPORAL_CHANNELS, 3, padding=1))
        self.shuffle_blocks = nn.ModuleList(
            PaperShuffleBlock2D(TEMPORAL_CHANNELS) for _ in range(3)
        )
        self.frequency_upsamples = nn.ModuleList(
            [
                _normalized(nn.ConvTranspose2d(64, 32, (4, 3), (2, 1), (1, 1))),
                _normalized(nn.ConvTranspose2d(32, 16, (4, 3), (2, 1), (1, 1))),
                _normalized(nn.ConvTranspose2d(16, 8, (3, 3), (2, 1), (0, 1))),
            ]
        )
        self.istft = PaperMultiBandISTFT()

    def temporal_features(self, mel: torch.Tensor) -> torch.Tensor:
        assert mel.ndim == 3, f"expected rank-3 mel conditioning, got rank {mel.ndim}"
        assert mel.shape[1] == MEL_CHANNELS, f"expected 80 mel channels, got {mel.shape[1]}"
        features = self.conv_pre(mel)
        temporal = self.temporal_up(F.leaky_relu(features, LRELU_SLOPE))
        assert temporal.shape[-1] == mel.shape[-1] * TEMPORAL_RATE
        return temporal

    def subband_spectrogram(self, temporal: torch.Tensor) -> torch.Tensor:
        features = self.mrf(temporal)
        batch, channels, frames = features.shape
        features = features.view(batch, channels // 4, 4, frames)
        features = self.entry_2d(features)
        for block in self.shuffle_blocks:
            features = block(features)
        for upsample in self.frequency_upsamples:
            features = upsample(F.leaky_relu(features, LRELU_SLOPE))
        assert features.shape[2] == FREQUENCY_BINS, (
            f"frequency path produced {features.shape[2]} bins; expected {FREQUENCY_BINS}"
        )
        return features

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        waveform = self.istft(self.subband_spectrogram(self.temporal_features(mel)))
        assert waveform.shape[-1] == mel.shape[-1] * OUTPUT_HOP
        return waveform
