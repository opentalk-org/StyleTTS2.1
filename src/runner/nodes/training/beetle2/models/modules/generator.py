import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from ...config.architecture import GeneratorConfig
from .vocoder import HarmonicSource, ISTFT


class SnakeActivation(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, features: Tensor) -> Tensor:
        scaled = self.alpha * features
        return features + torch.sin(scaled).square() / self.alpha


class HiFTResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilations: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.first = nn.ModuleList(
            weight_norm(
                nn.Conv1d(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation=dilation,
                    padding=(kernel_size * dilation - dilation) // 2,
                )
            )
            for dilation in dilations
        )
        self.second = nn.ModuleList(
            weight_norm(
                nn.Conv1d(
                    hidden_channels,
                    channels,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                )
            )
            for _ in dilations
        )
        self.first.apply(_initialize_convolution)
        self.second.apply(_initialize_convolution)
        self.first_activations = nn.ModuleList(
            SnakeActivation(channels) for _ in dilations
        )
        self.second_activations = nn.ModuleList(
            SnakeActivation(hidden_channels) for _ in dilations
        )

    def forward(self, features: Tensor) -> Tensor:
        for first, second, activation1, activation2 in zip(
            self.first,
            self.second,
            self.first_activations,
            self.second_activations,
            strict=True,
        ):
            residual = first(activation1(features))
            residual = second(activation2(residual))
            features = features + residual
        return features


class Generator(nn.Module):
    def __init__(self, config: GeneratorConfig, sample_rate: int) -> None:
        super().__init__()
        self.config = config
        self.input_projection = weight_norm(
            nn.Conv1d(
                config.input_channels,
                config.upsample_initial_channel,
                7,
                padding=3,
            )
        )
        self.harmonic_source = HarmonicSource(
            sample_rate,
            config.output_hop(),
            config.harmonic_count,
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
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.istft = ISTFT(
            config.istft_n_fft,
            config.istft_hop_length,
        )
        self.upsamples.apply(_initialize_convolution)
        self.output_projection.apply(_initialize_convolution)

    def forward(
        self,
        features: Tensor,
        f0: Tensor,
        mask: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        frame_mask = mask.to(dtype=features.dtype)
        harmonic_waveform = self.harmonic_source(
            f0 * frame_mask[:, 0],
            generator,
        )
        source_magnitude, source_phase = self.istft.transform(
            harmonic_waveform[:, 0]
        )
        harmonic = torch.cat((source_magnitude, source_phase), dim=1)
        features = self.input_projection(features * frame_mask)
        kernel_count = len(self.config.resblock_kernel_sizes)
        for stage, upsample in enumerate(self.upsamples):
            features = F.leaky_relu(features, 0.1)
            source = self.source_convolutions[stage](harmonic)
            source = self.source_resblocks[stage](source)
            features = upsample(features)
            if stage == len(self.upsamples) - 1:
                features = self.reflection_pad(features)
            features = features + source.to(dtype=features.dtype)
            paths = None
            for path in range(kernel_count):
                residual = self.resblocks[stage * kernel_count + path](features)
                paths = residual if paths is None else paths + residual
            features = paths / kernel_count
        spectrum = self.output_projection(F.leaky_relu(features))
        frequency_bins = self.config.istft_n_fft // 2 + 1
        magnitude = torch.exp(spectrum[:, :frequency_bins].float())
        phase = torch.sin(spectrum[:, frequency_bins:].float())
        waveform = self.istft.inverse(magnitude, phase)
        sample_mask = frame_mask.repeat_interleave(
            self.config.output_hop(),
            dim=-1,
        )
        return waveform * sample_mask


def _initialize_convolution(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.normal_(module.weight, 0.0, 0.01)
