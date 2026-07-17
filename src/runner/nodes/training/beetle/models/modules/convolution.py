import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


def same_padding(kernel_size: int, dilation: int) -> int:
    return (kernel_size * dilation - dilation) // 2


class MaskedResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gated = weight_norm(
            nn.Conv1d(
                channels,
                channels * 2,
                kernel_size,
                dilation=dilation,
                padding=same_padding(kernel_size, dilation),
            )
        )
        self.project = weight_norm(nn.Conv1d(channels, channels, 1))
        self.dropout = dropout

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        hidden = self.gated(features * mask)
        tanh_part, sigmoid_part = hidden.chunk(2, dim=1)
        hidden = torch.tanh(tanh_part) * torch.sigmoid(sigmoid_part)
        hidden = F.dropout(hidden, self.dropout, self.training)
        return (features + self.project(hidden)) * mask


class DilatedResidualStack(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, ...],
        cycles: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            MaskedResidualBlock(channels, kernel_size, dilation, dropout)
            for _ in range(cycles)
            for dilation in dilations
        )

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        if mask.ndim != 3 or mask.shape[1] != 1:
            raise ValueError("temporal mask must have shape [B,1,T]")
        for block in self.blocks:
            features = block(features, mask)
        return features * mask


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: tuple[int, ...]) -> None:
        super().__init__()
        self.dilated = nn.ModuleList(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=dilation,
                padding=same_padding(kernel_size, dilation),
            )
            for dilation in dilations
        )
        self.plain = nn.ModuleList(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=same_padding(kernel_size, 1),
            )
            for _ in dilations
        )

    def forward(self, features: Tensor) -> Tensor:
        for first, second in zip(self.dilated, self.plain, strict=True):
            residual = second(F.leaky_relu(first(F.leaky_relu(features, 0.1)), 0.1))
            features = features + residual
        return features


class FrequencyShuffleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError("frequency shuffle channels must be even")
        half = channels // 2
        self.channels = channels
        self.convolutions = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv2d(half, channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels, half, 3, padding=1),
        )

    def forward(self, features: Tensor) -> Tensor:
        batch, channels, frequency, frames = features.shape
        shuffled = features.view(batch, 2, channels // 2, frequency, frames)
        shuffled = shuffled.transpose(1, 2).reshape(batch, channels, frequency, frames)
        skip, active = shuffled.chunk(2, dim=1)
        return torch.cat((skip, self.convolutions(active)), dim=1)


class FrequencyUpsample(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.convolution = nn.ConvTranspose2d(
            input_channels,
            output_channels,
            (kernel_size, 3),
            stride=(2, 1),
            padding=(1, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.convolution(F.leaky_relu(features, 0.1))
