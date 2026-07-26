import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


def normalized_weight_norm(module: nn.Module) -> nn.Module:
    nn.init.normal_(module.weight, mean=0.0, std=0.01)
    return weight_norm(module)


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
        for block in self.blocks:
            features = block(features, mask)
        return features * mask


class ResBlock1D(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            normalized_weight_norm(
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
        self.convs2 = nn.ModuleList(
            normalized_weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    padding=same_padding(kernel_size, 1),
                )
            )
            for _ in dilations
        )

    def forward(self, features: Tensor) -> Tensor:
        for first, second in zip(
            self.convs1,
            self.convs2,
            strict=True,
        ):
            residual = first(F.leaky_relu(features, 0.1))
            residual = second(F.leaky_relu(residual, 0.1))
            features = features + residual
        return features


class FrequencyShuffleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        half_channels = channels // 2
        self.convs = nn.Sequential(
            nn.LeakyReLU(0.1),
            normalized_weight_norm(
                nn.Conv2d(half_channels, channels, 3, padding=1)
            ),
            nn.LeakyReLU(0.1),
            normalized_weight_norm(
                nn.Conv2d(channels, half_channels, 3, padding=1)
            ),
        )

    def forward(self, features: Tensor) -> Tensor:
        batch, channels, frequency, frames = features.shape
        shuffled = features.view(batch, 2, channels // 2, frequency, frames)
        shuffled = shuffled.transpose(1, 2).reshape(
            batch,
            channels,
            frequency,
            frames,
        )
        skip, active = shuffled.chunk(2, dim=1)
        return torch.cat((skip, self.convs(active)), dim=1)


class FrequencyUpsample(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        frequency_kernel_size: int,
        frequency_padding: int,
    ) -> None:
        super().__init__()
        self.convolution = normalized_weight_norm(
            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                (frequency_kernel_size, 3),
                stride=(2, 1),
                padding=(frequency_padding, 1),
            )
        )

    def forward(self, features: Tensor) -> Tensor:
        activated = F.leaky_relu(features, negative_slope=0.1)
        return self.convolution(activated)
