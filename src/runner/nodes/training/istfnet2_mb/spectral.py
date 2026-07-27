from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchaudio.functional import melscale_fbanks

from .config import SignalConfig


class LogMelSpectrogram(nn.Module):
    def __init__(self, config: SignalConfig) -> None:
        super().__init__()
        mel_basis = melscale_fbanks(
            n_freqs=config.n_fft // 2 + 1,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.mel_channels,
            sample_rate=config.sample_rate,
            norm="slaney",
            mel_scale="slaney",
        ).transpose(0, 1)
        self.config = config
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", torch.hann_window(config.win_length))

    def forward(self, waveform: Tensor) -> Tensor:
        config = self.config
        padding = (config.n_fft - config.hop_length) // 2
        padded = F.pad(
            waveform.unsqueeze(1),
            (padding, padding),
            mode="reflect",
        ).squeeze(1)
        spectrum = torch.stft(
            padded,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        magnitude = torch.sqrt(
            torch.view_as_real(spectrum).square().sum(dim=-1) + 1e-9
        )
        mel = torch.matmul(self.mel_basis, magnitude)
        return torch.log(torch.clamp(mel, min=1e-5))
