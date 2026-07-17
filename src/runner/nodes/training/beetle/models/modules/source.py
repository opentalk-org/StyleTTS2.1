import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ...config.architecture import GeneratorConfig


class HarmonicSource(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        output_hop: int,
        harmonic_count: int,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.output_hop = output_hop
        self.harmonic_count = harmonic_count
        self.merge = nn.Linear(harmonic_count + 1, 1)

    def forward(self, f0: Tensor, generator: torch.Generator) -> Tensor:
        if f0.ndim != 2:
            raise ValueError("harmonic-source F0 must have shape [B,T]")
        sampled = F.interpolate(
            f0.unsqueeze(1),
            scale_factor=self.output_hop,
            mode="nearest",
        ).transpose(1, 2)
        harmonics = torch.arange(
            1,
            self.harmonic_count + 2,
            device=f0.device,
            dtype=f0.dtype,
        )
        frequencies = sampled * harmonics.view(1, 1, -1)
        increments = frequencies / self.sample_rate * (2 * math.pi)
        initial = torch.rand(
            f0.shape[0],
            self.harmonic_count + 1,
            device=f0.device,
            dtype=f0.dtype,
            generator=generator,
        ) * (2 * math.pi)
        initial[:, 0] = 0
        phase = torch.cumsum(increments, dim=1) + initial.unsqueeze(1)
        voiced = (sampled > 10).to(dtype=f0.dtype)
        sine = torch.sin(phase) * 0.1 * voiced
        noise_scale = voiced * 0.003 + (1 - voiced) * (0.1 / 3)
        noise = torch.randn(
            sine.shape,
            device=sine.device,
            dtype=sine.dtype,
            generator=generator,
        )
        return torch.tanh(self.merge(sine + noise * noise_scale)).transpose(1, 2)


class HarmonicSourceFeatures(nn.Module):
    def __init__(self, config: GeneratorConfig, sample_rate: int) -> None:
        super().__init__()
        self.config = config
        self.source = HarmonicSource(
            sample_rate,
            config.output_hop(),
            config.harmonic_count,
        )
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
        if features.shape[-1] != expected:
            raise ValueError(
                f"harmonic features produced {features.shape[-1]} frames; expected {expected}"
            )
        return features
