from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.1
MEL_CHANNELS = 80
BASE_CHANNELS = 128
TEMPORAL_CHANNELS = 64
TEMPORAL_UPSAMPLE = 4
SUBBANDS = 4
ISTFT_N_FFT = 64
ISTFT_HOP = 16
FREQUENCY_BINS = ISTFT_N_FFT // 2 + 1
OUTPUT_HOP = TEMPORAL_UPSAMPLE * ISTFT_HOP * SUBBANDS


def normalized(module: nn.Module) -> nn.Module:
    nn.init.normal_(module.weight, 0.0, 0.01)
    return weight_norm(module)


def same_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        dilations = (1, 3, 5)
        self.dilated = nn.ModuleList(
            normalized(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=dilation,
                    padding=same_padding(kernel_size, dilation),
                )
            )
            for dilation in dilations
        )
        self.plain = nn.ModuleList(
            normalized(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    padding=same_padding(kernel_size),
                )
            )
            for _ in dilations
        )

    def forward(self, features: Tensor) -> Tensor:
        for dilated, plain in zip(self.dilated, self.plain, strict=True):
            residual = F.leaky_relu(features, LRELU_SLOPE)
            residual = dilated(residual)
            residual = plain(F.leaky_relu(residual, LRELU_SLOPE))
            features = features + residual
        return features


class MultiReceptiveField(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ResBlock1D(TEMPORAL_CHANNELS, kernel_size)
            for kernel_size in (3, 7, 11)
        )

    def forward(self, features: Tensor) -> Tensor:
        return torch.cat(tuple(block(features) for block in self.blocks), dim=1)


class MultiBandShuffleBlock(nn.Module):
    """Figure 3(b), using the paper's multi-band C rather than 2C expansion."""

    def __init__(self) -> None:
        super().__init__()
        half_channels = TEMPORAL_CHANNELS // 2
        self.expand = normalized(
            nn.Conv2d(half_channels, TEMPORAL_CHANNELS, 3, padding=1)
        )
        self.contract = normalized(
            nn.Conv2d(TEMPORAL_CHANNELS, half_channels, 3, padding=1)
        )

    def forward(self, features: Tensor) -> Tensor:
        batch, channels, frequencies, frames = features.shape
        shuffled = features.view(batch, 2, channels // 2, frequencies, frames)
        shuffled = shuffled.transpose(1, 2).reshape(
            batch,
            channels,
            frequencies,
            frames,
        )
        skip, active = shuffled.chunk(2, dim=1)
        active = self.expand(F.leaky_relu(active, LRELU_SLOPE))
        active = self.contract(F.leaky_relu(active, LRELU_SLOPE))
        return torch.cat((skip, active), dim=1)


class PQMF(nn.Module):
    def __init__(
        self,
        taps: int = 62,
        cutoff_ratio: float = 0.142,
        beta: float = 9.0,
    ) -> None:
        super().__init__()
        positions = torch.arange(taps + 1, dtype=torch.float64) - taps / 2
        prototype = cutoff_ratio * torch.sinc(cutoff_ratio * positions)
        prototype *= torch.kaiser_window(
            taps + 1,
            periodic=False,
            beta=beta,
            dtype=torch.float64,
        )
        filters = []
        for band in range(SUBBANDS):
            phase = (
                (2 * band + 1) * torch.pi / (2 * SUBBANDS) * positions
                - ((-1) ** band) * torch.pi / 4
            )
            filters.append(2 * prototype * torch.cos(phase))
        synthesis = torch.stack(filters).float().unsqueeze(0)
        upsample = torch.zeros(SUBBANDS, SUBBANDS, SUBBANDS)
        indices = torch.arange(SUBBANDS)
        upsample[indices, indices, 0] = 1
        self.register_buffer("synthesis_filter", synthesis)
        self.register_buffer("upsample_filter", upsample)
        self.padding = nn.ConstantPad1d(taps // 2, 0.0)

    def forward(self, subbands: Tensor) -> Tensor:
        upsampled = F.conv_transpose1d(
            subbands,
            self.upsample_filter * SUBBANDS,
            stride=SUBBANDS,
        )
        return F.conv1d(self.padding(upsampled), self.synthesis_filter)


class MultiBandISTFT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pqmf = PQMF()
        self.register_buffer("window", torch.hann_window(ISTFT_N_FFT))

    def _inverse(self, spectrogram: Tensor, length: int) -> Tensor:
        magnitude = torch.exp(spectrogram[:, 0].float())
        phase = torch.sin(spectrogram[:, 1].float())
        return torch.istft(
            torch.polar(magnitude, phase),
            n_fft=ISTFT_N_FFT,
            hop_length=ISTFT_HOP,
            win_length=ISTFT_N_FFT,
            window=self.window,
            center=True,
            length=length,
        )

    def forward(self, spectrogram: Tensor) -> Tensor:
        batch, channels, frequencies, frames = spectrogram.shape
        assert channels == SUBBANDS * 2, f"expected 8 spectral channels, got {channels}"
        assert frequencies == FREQUENCY_BINS, (
            f"expected {FREQUENCY_BINS} frequency bins, got {frequencies}"
        )
        bands = spectrogram.view(
            batch,
            SUBBANDS,
            2,
            frequencies,
            frames,
        )
        band_length = frames * ISTFT_HOP
        waveforms = torch.stack(
            tuple(self._inverse(bands[:, band], band_length) for band in range(SUBBANDS)),
            dim=1,
        )
        waveform = self.pqmf(waveforms)
        assert waveform.shape == (batch, 1, band_length * SUBBANDS)
        return waveform


class ISTFTNet2MB(nn.Module):
    """iSTFTNet2-MB from Kaneko et al., INTERSPEECH 2023, Section 4.3."""

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = normalized(
            nn.Conv1d(MEL_CHANNELS, BASE_CHANNELS, 7, padding=3)
        )
        self.temporal_upsample = normalized(
            nn.ConvTranspose1d(
                BASE_CHANNELS,
                TEMPORAL_CHANNELS,
                8,
                stride=TEMPORAL_UPSAMPLE,
                padding=2,
            )
        )
        self.mrf = MultiReceptiveField()
        self.frequency_entry = normalized(
            nn.Conv2d(48, TEMPORAL_CHANNELS, 3, padding=1)
        )
        self.shuffle_blocks = nn.ModuleList(
            MultiBandShuffleBlock() for _ in range(3)
        )
        self.frequency_upsamples = nn.ModuleList(
            (
                normalized(
                    nn.ConvTranspose2d(64, 32, (4, 3), (2, 1), (1, 1))
                ),
                normalized(
                    nn.ConvTranspose2d(32, 16, (4, 3), (2, 1), (1, 1))
                ),
                normalized(
                    nn.ConvTranspose2d(16, 8, (3, 3), (2, 1), (0, 1))
                ),
            )
        )
        self.istft = MultiBandISTFT()

    def forward(self, mel: Tensor) -> Tensor:
        assert mel.ndim == 3, f"expected [batch, 80, frames], got {mel.shape}"
        assert mel.shape[1] == MEL_CHANNELS, (
            f"expected {MEL_CHANNELS} mel channels, got {mel.shape[1]}"
        )
        features = self.input_projection(mel)
        features = self.temporal_upsample(
            F.leaky_relu(features, LRELU_SLOPE)
        )
        features = self.mrf(features)
        batch, channels, frames = features.shape
        features = features.view(batch, channels // 4, 4, frames)
        features = self.frequency_entry(features)
        for block in self.shuffle_blocks:
            features = block(features)
        for upsample in self.frequency_upsamples:
            features = upsample(F.leaky_relu(features, LRELU_SLOPE))
        waveform = self.istft(features)
        assert waveform.shape[-1] == mel.shape[-1] * OUTPUT_HOP
        return waveform
