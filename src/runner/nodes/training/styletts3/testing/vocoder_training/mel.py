from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import melscale_fbanks

from runner.nodes.training.styletts3.testing.vocoder_training.geometry import (
    MEL_CHANNELS,
    MEL_FMAX,
    MEL_FMIN,
    SAMPLE_RATE,
    SYNTHESIS_HOP,
)


@dataclass(frozen=True)
class MelResolution:
    n_fft: int
    win_length: int
    hop_length: int
    n_mels: int = MEL_CHANNELS


CONDITIONING_RESOLUTION = MelResolution(2048, 1200, 300)
LOSS_RESOLUTIONS = (
    MelResolution(1024, 600, 120),
    MelResolution(2048, 1200, 240),
    MelResolution(512, 240, 50),
)


class LogMelSpectrogram(nn.Module):
    def __init__(self, resolution: MelResolution) -> None:
        super().__init__()
        self.resolution = resolution
        window = torch.hann_window(resolution.win_length)
        mel_basis = melscale_fbanks(
            n_freqs=resolution.n_fft // 2 + 1,
            f_min=MEL_FMIN,
            f_max=MEL_FMAX,
            n_mels=resolution.n_mels,
            sample_rate=SAMPLE_RATE,
            norm="slaney",
            mel_scale="slaney",
        ).transpose(0, 1)
        self.register_buffer("window", window)
        self.register_buffer("mel_basis", mel_basis)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        resolution = self.resolution
        padding = (resolution.n_fft - resolution.hop_length) // 2
        padded = F.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
        spectrum = torch.stft(
            padded,
            n_fft=resolution.n_fft,
            hop_length=resolution.hop_length,
            win_length=resolution.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        magnitude = spectrum.abs()
        mel = torch.matmul(self.mel_basis, magnitude)
        return torch.log(torch.clamp(mel, min=1e-5))


class MultiResolutionMelLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(LogMelSpectrogram(item) for item in LOSS_RESOLUTIONS)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = [F.l1_loss(transform(prediction), transform(target)) for transform in self.transforms]
        return torch.stack(losses).mean()


def conditioning_mel() -> LogMelSpectrogram:
    return LogMelSpectrogram(CONDITIONING_RESOLUTION)


def pad_to_hop(waveform: torch.Tensor, hop: int = SYNTHESIS_HOP) -> tuple[torch.Tensor, int]:
    original_length = waveform.shape[-1]
    right_padding = (-original_length) % hop
    return F.pad(waveform, (0, right_padding)), original_length

