import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from ...config.architecture import GeneratorConfig
from .convolution import (
    FrequencyShuffleBlock,
    FrequencyUpsample,
    ResBlock1D,
    normalized_weight_norm,
)
from .vocoder import MultiBandISTFT


class Generator(nn.Module):
    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__()
        self.config = config
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
        self.input_projection = normalized_weight_norm(
            nn.Conv1d(
                config.input_channels,
                config.frame_channels,
                7,
                padding=3,
            )
        )
        self.temporal_upsample = normalized_weight_norm(
            nn.ConvTranspose1d(
                config.frame_channels,
                config.temporal_channels,
                config.temporal_upsample_kernel_size,
                stride=config.temporal_upsample_rate,
                padding=padding,
                output_padding=output_padding,
            )
        )
        self.resblocks = nn.ModuleList(
            ResBlock1D(config.temporal_channels, kernel, dilations)
            for kernel, dilations in zip(
                config.resblock_kernel_sizes,
                config.resblock_dilations,
                strict=True,
            )
        )
        concatenated_channels = config.temporal_channels * len(self.resblocks)
        self.frequency_entry = normalized_weight_norm(
            nn.Conv2d(
                concatenated_channels // config.initial_frequency_bins,
                config.temporal_channels,
                3,
                padding=1,
            )
        )
        source_channels = config.istft_n_fft + 2
        self.upsamples = nn.ModuleList()
        self.source_convolutions = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        stage_count = len(config.upsample_rates)
        for stage, (rate, kernel) in enumerate(
            zip(
                config.upsample_rates,
                config.upsample_kernel_sizes,
                strict=True,
            )
        ):
            input_channels = config.upsample_initial_channel // (2**stage)
            channels = config.upsample_initial_channel // (2 ** (stage + 1))
            self.upsamples.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        input_channels,
                        channels,
                        kernel,
                        rate,
                        padding=(kernel - rate) // 2,
                    )
                )
            )
            final_stage = stage == stage_count - 1
            remaining_rate = math.prod(config.upsample_rates[stage + 1 :])
            source_kernel = 1 if final_stage else remaining_rate * 2
            source_stride = 1 if final_stage else remaining_rate
            source_padding = 0 if final_stage else (remaining_rate + 1) // 2
            self.source_convolutions.append(
                nn.Conv1d(
                    source_channels,
                    channels,
                    source_kernel,
                    stride=source_stride,
                    padding=source_padding,
                )
            )
            hidden_channels = (
                channels // config.final_stage_resblock_bottleneck
                if final_stage
                else channels
            )
            source_resblock_kernel = 11 if final_stage else 7
            self.source_resblocks.append(
                HiFTResidualBlock(
                    channels,
                    hidden_channels,
                    source_resblock_kernel,
                    (1, 3, 5),
                )
            )
            self.resblocks.extend(
                HiFTResidualBlock(
                    channels,
                    hidden_channels,
                    resblock_kernel,
                    dilations,
                )
                for resblock_kernel, dilations in zip(
                    config.resblock_kernel_sizes,
                    config.resblock_dilations,
                    strict=True,
                )
            )
        self.output_projection = weight_norm(
            nn.Conv1d(channels, config.istft_n_fft + 2, 7, padding=3)
        )
        kernels = config.frequency_upsample_kernel_sizes
        paddings = config.frequency_upsample_paddings
        self.frequency_upsamples = nn.ModuleList(
            (
                FrequencyUpsample(
                    config.temporal_channels,
                    config.temporal_channels // 2,
                    kernels[0],
                    paddings[0],
                ),
                FrequencyUpsample(
                    config.temporal_channels // 2,
                    config.temporal_channels // 4,
                    kernels[1],
                    paddings[1],
                ),
                FrequencyUpsample(
                    config.temporal_channels // 4,
                    config.subbands * 2,
                    kernels[2],
                    paddings[2],
                ),
            )
        )
        self.istft = MultiBandISTFT(
            config.subbands,
            config.istft_n_fft,
            config.istft_hop_length,
        )
        self.upsamples.apply(_initialize_convolution)
        self.output_projection.apply(_initialize_convolution)

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        frame_mask = mask.to(dtype=features.dtype)
        projected = self.input_projection(features * frame_mask)
        temporal = self.temporal_upsample(F.leaky_relu(projected, 0.1))
        temporal_mask = F.interpolate(
            frame_mask,
            size=features.shape[-1] * self.config.temporal_upsample_rate,
            mode="nearest",
        )
        spectrum = self._subband_spectrogram(temporal * temporal_mask)
        waveform = self.istft(spectrum)
        sample_mask = frame_mask.repeat_interleave(
            self.config.output_hop(),
            dim=-1,
        )
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
        for upsample in self.frequency_upsamples:
            features = upsample(features)
        expected_bins = self.config.istft_n_fft // 2 + 1
        assert features.shape[2] == expected_bins, (
            f"generator produced {features.shape[2]} bins; expected {expected_bins}"
        )
        return features
