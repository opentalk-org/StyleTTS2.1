from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import melscale_fbanks

from runner.nodes.training.styletts3.testing.vocoder_training.profiles import (
    MelGeometry,
    SignalGeometry,
)


class LogMelSpectrogram(nn.Module):
    def __init__(self, resolution: MelGeometry, sample_rate: int) -> None:
        super().__init__()
        self.resolution = resolution
        window = torch.hann_window(resolution.win_length)
        mel_basis = melscale_fbanks(
            n_freqs=resolution.n_fft // 2 + 1,
            f_min=resolution.fmin,
            f_max=resolution.fmax,
            n_mels=resolution.n_mels,
            sample_rate=sample_rate,
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
    def __init__(self, signal: SignalGeometry) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(
            LogMelSpectrogram(item, signal.sample_rate) for item in signal.reconstruction
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = [F.l1_loss(transform(prediction), transform(target)) for transform in self.transforms]
        return torch.stack(losses).mean()


def conditioning_mel(signal: SignalGeometry) -> LogMelSpectrogram:
    return LogMelSpectrogram(signal.conditioning, signal.sample_rate)


def pad_to_hop(waveform: torch.Tensor, hop: int) -> tuple[torch.Tensor, int]:
    original_length = waveform.shape[-1]
    right_padding = (-original_length) % hop
    return F.pad(waveform, (0, right_padding)), original_length
