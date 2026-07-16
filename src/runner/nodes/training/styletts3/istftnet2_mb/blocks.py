from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


LRELU_SLOPE = 0.1


def pad_same(kernel: int, dilation: int = 1) -> int:
    return (kernel * dilation - dilation) // 2


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel: int, dilations: tuple[int, ...] = (1, 3, 5)) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=pad_same(kernel, dilation))
            for dilation in dilations
        )
        self.convs2 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel, padding=pad_same(kernel))
            for _ in dilations
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = conv2(F.leaky_relu(conv1(F.leaky_relu(features, LRELU_SLOPE)), LRELU_SLOPE))
            features = features + residual
        return features


class MRF1DConcat(nn.Module):
    def __init__(self, channels: int, kernels: tuple[int, ...] = (3, 7, 11)) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(ResBlock1D(channels, kernel) for kernel in kernels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([block(features) for block in self.blocks], dim=1)


class ShuffleBlock2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        assert channels % 2 == 0, "ShuffleBlock channels must be even"
        self.channels = channels
        half = channels // 2
        self.convs = nn.Sequential(
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv2d(half, channels, 3, padding=1),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv2d(channels, half, 3, padding=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, frequency, time = features.shape
        shuffled = features.view(batch, 2, channels // 2, frequency, time)
        shuffled = shuffled.transpose(1, 2).reshape(batch, channels, frequency, time)
        skip, active = shuffled[:, : self.channels // 2], shuffled[:, self.channels // 2 :]
        return torch.cat([skip, self.convs(active)], dim=1)


class FrequencyUpsample2D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, frequency_kernel: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            input_channels,
            output_channels,
            (frequency_kernel, 3),
            stride=(2, 1),
            padding=(1, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(F.leaky_relu(features, LRELU_SLOPE))

