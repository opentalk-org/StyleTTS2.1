from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from traintts.modules.decoder_blocks import SourceModuleHnNSF


SAMPLE_RATE = 24_000
SOURCE_SCALE = 300
SOURCE_NFFT = 240
SOURCE_HOP = 60
SOURCE_PADDING = (SOURCE_NFFT - SOURCE_HOP) // 2
SOURCE_CHANNELS = (SOURCE_NFFT // 2 + 1) * 2


class HarmonicSourceFeatures(nn.Module):
    harmonic_components = 9

    def __init__(self) -> None:
        super().__init__()
        self.source = SourceModuleHnNSF(
            sampling_rate=SAMPLE_RATE,
            upsample_scale=SOURCE_SCALE,
            harmonic_num=self.harmonic_components - 1,
            sine_amp=0.1,
            add_noise_std=0.003,
            voiced_threshod=10,
        )
        self.register_buffer("window", torch.hann_window(SOURCE_NFFT))

    def harmonic_waveform(self, f0: torch.Tensor) -> torch.Tensor:
        assert f0.ndim == 2, f"expected rank-2 F0, got rank {f0.ndim}"
        sampled_f0 = F.interpolate(f0.unsqueeze(1), scale_factor=SOURCE_SCALE, mode="nearest")
        merged, _, _ = self.source(sampled_f0.transpose(1, 2))
        waveform = merged.transpose(1, 2)
        expected_length = f0.shape[-1] * SOURCE_SCALE
        assert waveform.shape == (f0.shape[0], 1, expected_length), (
            f"harmonic source produced {waveform.shape[-1]} samples; expected {expected_length}"
        )
        return waveform

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        waveform = self.harmonic_waveform(f0)
        padded = F.pad(waveform, (SOURCE_PADDING, SOURCE_PADDING), mode="reflect")
        spectrum = torch.stft(
            padded.squeeze(1).float(),
            n_fft=SOURCE_NFFT,
            hop_length=SOURCE_HOP,
            win_length=SOURCE_NFFT,
            window=self.window,
            center=False,
            return_complex=True,
        )
        features = torch.cat([spectrum.abs(), torch.angle(spectrum)], dim=1)
        expected_frames = f0.shape[-1] * (SOURCE_SCALE // SOURCE_HOP)
        assert features.shape == (f0.shape[0], SOURCE_CHANNELS, expected_frames), (
            f"harmonic STFT produced {features.shape}; expected "
            f"({f0.shape[0]}, {SOURCE_CHANNELS}, {expected_frames})"
        )
        return features
