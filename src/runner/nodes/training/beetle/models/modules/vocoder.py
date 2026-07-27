import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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

    def forward(
        self,
        f0: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
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
            initial_phase = torch.rand(
                f0.shape[0],
                self.harmonic_count + 1,
                device=f0.device,
                dtype=torch.float32,
                generator=generator,
            )
            initial_phase[:, 0] = 0
            phase_increments[:, 0] = (
                phase_increments[:, 0] + initial_phase
            )
            frame_increments = F.interpolate(
                phase_increments.transpose(1, 2),
                size=f0.shape[-1],
                mode="linear",
                align_corners=False,
            )
            frame_phase = torch.cumsum(
                frame_increments.transpose(1, 2),
                dim=1,
            )
            phase = F.interpolate(
                (
                    frame_phase * (2 * math.pi * self.output_hop)
                ).transpose(1, 2),
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


class ISTFT(nn.Module):
    def __init__(self, n_fft: int, hop_length: int) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

    def transform(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        spectrum = torch.stft(
            waveform.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        )
        return spectrum.abs(), torch.angle(spectrum)

    def inverse(self, magnitude: Tensor, phase: Tensor) -> Tensor:
        spectrum = torch.polar(magnitude.float(), phase.float())
        sample_count = (magnitude.shape[-1] - 1) * self.hop_length
        waveform = torch.istft(
            spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            length=sample_count,
        )
        return waveform.unsqueeze(1)
