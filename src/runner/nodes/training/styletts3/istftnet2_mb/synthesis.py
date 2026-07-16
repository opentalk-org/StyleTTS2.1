from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


SUBBANDS = 4
ISTFT_NFFT = 60
ISTFT_HOP = 15
FREQUENCY_BINS = ISTFT_NFFT // 2 + 1


class PQMF(nn.Module):
    def __init__(
        self,
        subbands: int = SUBBANDS,
        taps: int = 62,
        cutoff_ratio: float = 0.142,
        beta: float = 9.0,
    ) -> None:
        super().__init__()
        assert taps % 2 == 0, "PQMF prototype taps must be even"
        positions = torch.arange(taps + 1, dtype=torch.float64) - taps / 2
        prototype = cutoff_ratio * torch.sinc(cutoff_ratio * positions)
        prototype *= torch.kaiser_window(taps + 1, periodic=False, beta=beta, dtype=torch.float64)
        filters = []
        for band in range(subbands):
            phase = (
                (2 * band + 1) * torch.pi / (2 * subbands) * positions
                - ((-1) ** band) * torch.pi / 4
            )
            filters.append(2 * prototype * torch.cos(phase))
        synthesis_filter = torch.stack(filters).float().unsqueeze(0)
        upsample_filter = torch.zeros(subbands, subbands, subbands)
        band_indices = torch.arange(subbands)
        upsample_filter[band_indices, band_indices, 0] = 1.0
        self.register_buffer("synthesis_filter", synthesis_filter)
        self.register_buffer("upsample_filter", upsample_filter)
        self.pad = nn.ConstantPad1d(taps // 2, 0.0)
        self.subbands = subbands

    def synthesis(self, subband_waveforms: torch.Tensor) -> torch.Tensor:
        upsampled = F.conv_transpose1d(
            subband_waveforms,
            self.upsample_filter * self.subbands,
            stride=self.subbands,
        )
        return F.conv1d(self.pad(upsampled), self.synthesis_filter)


class MultiBandISTFT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pqmf = PQMF()
        self.register_buffer("istft_window", torch.hann_window(ISTFT_NFFT))

    def _inverse_band(self, spectrogram: torch.Tensor, length: int) -> torch.Tensor:
        magnitude = torch.exp(spectrogram[:, 0].float())
        phase = torch.sin(spectrogram[:, 1].float())
        spectrum = torch.polar(magnitude, phase)
        return torch.istft(
            spectrum,
            ISTFT_NFFT,
            hop_length=ISTFT_HOP,
            win_length=ISTFT_NFFT,
            window=self.istft_window,
            center=True,
            length=length,
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        batch, channels, frequency, frames = spectrogram.shape
        assert channels == SUBBANDS * 2, f"expected 8 spectrogram channels, got {channels}"
        assert frequency == FREQUENCY_BINS, f"expected 31 frequency bins, got {frequency}"
        bands = spectrogram.view(batch, SUBBANDS, 2, frequency, frames)
        band_length = frames * ISTFT_HOP
        waveforms = torch.stack(
            [self._inverse_band(bands[:, band], band_length) for band in range(SUBBANDS)],
            dim=1,
        )
        waveform = self.pqmf.synthesis(waveforms)
        expected_length = frames * ISTFT_HOP * SUBBANDS
        assert waveform.shape == (batch, 1, expected_length), (
            f"PQMF produced {waveform.shape[-1]} samples; expected {expected_length}"
        )
        return waveform

