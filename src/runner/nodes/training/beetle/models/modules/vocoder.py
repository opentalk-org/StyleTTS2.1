import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ...config.architecture import GeneratorConfig


class HarmonicSource(nn.Module):
    def __init__(self, sample_rate: int, output_hop: int, harmonic_count: int) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.output_hop = output_hop
        self.harmonic_count = harmonic_count
        self.merge = nn.Linear(harmonic_count + 1, 1)

    def forward(self, f0: Tensor, generator: torch.Generator) -> Tensor:
        with torch.no_grad():
            sampled = F.interpolate(
                f0.float().unsqueeze(1),
                scale_factor=self.output_hop,
                mode="nearest",
            ).transpose(1, 2)
            harmonics = torch.arange(
                1,
                self.harmonic_count + 2,
                device=f0.device,
                dtype=torch.float32,
            )
            phase_increments = sampled * harmonics.view(1, 1, -1)
            phase_increments = (phase_increments / self.sample_rate) % 1
            initial = torch.rand(
                f0.shape[0],
                self.harmonic_count + 1,
                device=f0.device,
                dtype=torch.float32,
                generator=generator,
            )
            initial[:, 0] = 0
            phase_increments[:, 0] = phase_increments[:, 0] + initial
            frame_increments = F.interpolate(
                phase_increments.transpose(1, 2),
                size=f0.shape[-1],
                mode="linear",
                align_corners=False,
            )
            frame_phase = torch.cumsum(frame_increments.transpose(1, 2), dim=1)
            phase = F.interpolate(
                (frame_phase * (2 * math.pi * self.output_hop)).transpose(1, 2),
                size=sampled.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            voiced = (sampled > 10).to(dtype=torch.float32)
            sine = torch.sin(phase) * 0.1 * voiced
            noise_scale = voiced * 0.003 + (1 - voiced) * (0.1 / 3)
            noise = torch.randn(
                sine.shape,
                device=sine.device,
                dtype=torch.float32,
                generator=generator,
            )
            excitation = sine + noise * noise_scale
        return torch.tanh(self.merge(excitation)).transpose(1, 2)


class HarmonicSourceFeatures(nn.Module):
    def __init__(self, config: GeneratorConfig, sample_rate: int) -> None:
        super().__init__()
        self.config = config
        self.source = HarmonicSource(sample_rate, config.output_hop(), config.harmonic_count)
        self.register_buffer("window", torch.hann_window(config.source_n_fft))

    @property
    def output_channels(self) -> int:
        return (self.config.source_n_fft // 2 + 1) * 2

    def forward(self, f0: Tensor, generator: torch.Generator) -> Tensor:
        waveform = self.source(f0, generator)
        padding = (self.config.source_n_fft - self.config.source_hop_length) // 2
        padded = F.pad(waveform, (padding, padding), mode="reflect")
        spectrum = torch.stft(
            padded.squeeze(1).float(),
            n_fft=self.config.source_n_fft,
            hop_length=self.config.source_hop_length,
            win_length=self.config.source_n_fft,
            window=self.window,
            center=False,
            return_complex=True,
        )
        features = torch.cat((spectrum.abs(), torch.angle(spectrum)), dim=1)
        expected = f0.shape[-1] * self.config.temporal_upsample_rate
            )
        return features


class PQMF(nn.Module):
    def __init__(
        self,
        subbands: int,
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
        for band in range(subbands):
            phase = (
                (2 * band + 1) * torch.pi / (2 * subbands) * positions
                - ((-1) ** band) * torch.pi / 4
            )
            filters.append(2 * prototype * torch.cos(phase))
        synthesis = torch.stack(filters).float().unsqueeze(0)
        upsample = torch.zeros(subbands, subbands, subbands)
        indices = torch.arange(subbands)
        upsample[indices, indices, 0] = 1
        self.register_buffer("synthesis_filter", synthesis)
        self.register_buffer("upsample_filter", upsample)
        self.padding = nn.ConstantPad1d(taps // 2, 0)
        self.subbands = subbands

    def synthesize(self, subband_waveforms: Tensor) -> Tensor:
        upsampled = F.conv_transpose1d(
            subband_waveforms,
            self.upsample_filter * self.subbands,
            stride=self.subbands,
        )
        return F.conv1d(self.padding(upsampled), self.synthesis_filter)


class MultiBandISTFT(nn.Module):
    def __init__(self, subbands: int, n_fft: int, hop_length: int) -> None:
        super().__init__()
        self.subbands = subbands
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pqmf = PQMF(subbands)
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, spectrogram: Tensor) -> Tensor:
        batch, channels, frequency_bins, frames = spectrogram.shape
        bands = spectrogram.view(batch, self.subbands, 2, frequency_bins, frames)
        band_length = frames * self.hop_length
        waveforms = []
        for band in range(self.subbands):
            magnitude = torch.exp(bands[:, band, 0].float())
            phase = bands[:, band, 1].float()
            spectrum = torch.polar(magnitude, phase)
            waveforms.append(
                torch.istft(
                    spectrum,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.n_fft,
                    window=self.window,
                    center=True,
                    length=band_length,
                )
            )
        waveform = self.pqmf.synthesize(torch.stack(waveforms, dim=1))
        expected = frames * self.hop_length * self.subbands
            )
        return waveform
