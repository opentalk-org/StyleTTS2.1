import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config.architecture import GeneratorConfig
from .modules.convolution import (
    FrequencyShuffleBlock,
    FrequencyUpsample,
    ResBlock1D,
)
from .modules.istft import MultiBandISTFT
from .modules.source import HarmonicSourceFeatures


class Generator(nn.Module):
    def __init__(self, config: GeneratorConfig, sample_rate: int) -> None:
        super().__init__()
        self.config = config
        self.sample_rate = sample_rate
        padding = (
            config.temporal_upsample_kernel_size
            - config.temporal_upsample_rate
            + 1
        ) // 2
        output_padding = (
            config.temporal_upsample_rate
            + 2 * padding
            - config.temporal_upsample_kernel_size
        )
        if output_padding < 0 or output_padding >= config.temporal_upsample_rate:
            raise ValueError("temporal upsampling cannot produce an exact integer rate")
        self.input_projection = nn.Conv1d(
            config.input_channels,
            config.frame_channels,
            7,
            padding=3,
        )
        self.temporal_upsample = nn.ConvTranspose1d(
            config.frame_channels,
            config.temporal_channels,
            config.temporal_upsample_kernel_size,
            stride=config.temporal_upsample_rate,
            padding=padding,
            output_padding=output_padding,
        )
        self.resblocks = nn.ModuleList(
            ResBlock1D(config.temporal_channels, kernel, dilations)
            for kernel, dilations in zip(
                config.resblock_kernel_sizes,
                config.resblock_dilations,
                strict=True,
            )
        )
        concatenated = config.temporal_channels * len(self.resblocks)
        if concatenated % config.initial_frequency_bins:
            raise ValueError("MRF channels must divide into initial frequency bins")
        self.frequency_entry = nn.Conv2d(
            concatenated // config.initial_frequency_bins,
            config.temporal_channels,
            3,
            padding=1,
        )
        self.frequency_shuffles = nn.ModuleList(
            FrequencyShuffleBlock(config.temporal_channels) for _ in range(3)
        )
        first_kernel, second_kernel, third_kernel = config.frequency_upsample_kernel_sizes
        self.frequency_up_1 = FrequencyUpsample(
            config.temporal_channels,
            config.temporal_channels // 2,
            first_kernel,
        )
        self.frequency_up_2 = FrequencyUpsample(
            config.temporal_channels // 2,
            config.temporal_channels // 4,
            second_kernel,
        )
        self.frequency_up_3 = FrequencyUpsample(
            config.temporal_channels // 4,
            config.subbands * 2,
            third_kernel,
        )
        self.harmonic_features = HarmonicSourceFeatures(config, sample_rate)
        self.source_projection = nn.Conv1d(
            self.harmonic_features.output_channels,
            config.temporal_channels,
            1,
        )
        self.source_residual = ResBlock1D(
            config.temporal_channels,
            11,
            (1, 3, 5),
        )
        self.istft = MultiBandISTFT(
            config.subbands,
            config.istft_n_fft,
            config.istft_hop_length,
        )

    def forward(
        self,
        features: Tensor,
        f0: Tensor,
        mask: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        if features.ndim != 3 or features.shape[1] != self.config.input_channels:
            raise ValueError("generator features must have configured [B,C,T] geometry")
        batch_size, _, frames = features.shape
        if f0.shape != (batch_size, frames):
            raise ValueError("generator F0 must match frame features")
        if mask.shape != (batch_size, 1, frames):
            raise ValueError("generator mask must have shape [B,1,T]")
        frame_mask = mask.to(dtype=features.dtype)
        projected = self.input_projection(features * frame_mask) * frame_mask
        temporal = self.temporal_upsample(F.leaky_relu(projected, 0.1))
        expected_frames = frames * self.config.temporal_upsample_rate
        if temporal.shape[-1] != expected_frames:
            raise ValueError("temporal upsampling did not produce configured rate")
        source = self.source_projection(
            self.harmonic_features(f0 * frame_mask[:, 0], generator)
        )
        source = self.source_residual(source)
        temporal_mask = F.interpolate(frame_mask, size=expected_frames, mode="nearest")
        temporal = (temporal + source) * temporal_mask
        spectrum = self._subband_spectrogram(temporal)
        waveform = self.istft(spectrum)
        sample_mask = frame_mask.repeat_interleave(self.config.output_hop(), dim=-1)
        return waveform * sample_mask

    def _subband_spectrogram(self, temporal: Tensor) -> Tensor:
        features = torch.cat(
            tuple(block(temporal) for block in self.resblocks),
            dim=1,
        )
        batch, channels, frames = features.shape
        features = features.view(
            batch,
            channels // self.config.initial_frequency_bins,
            self.config.initial_frequency_bins,
            frames,
        )
        features = self.frequency_entry(features)
        for shuffle in self.frequency_shuffles:
            features = shuffle(features)
        features = self.frequency_up_1(features)
        features = self.frequency_up_2(features)
        features = self.frequency_up_3(features)
        frequency_bins = self.config.istft_n_fft // 2 + 1
        if features.shape[2] != frequency_bins:
            raise ValueError("frequency path did not reach configured iSTFT bins")
        return features
