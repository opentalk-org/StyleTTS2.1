from math import sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


def same_padding(kernel_size: int, dilation: int) -> int:
    return (kernel_size * dilation - dilation) // 2


class MaskedInstanceNorm1d(nn.Module):
    def __init__(self, channels: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))
        self.epsilon = epsilon

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("instance normalization features must have shape [B,C,T]")
        if mask.shape != (features.shape[0], 1, features.shape[2]):
            raise ValueError("instance normalization mask must have shape [B,1,T]")
        numeric_mask = mask.to(dtype=features.dtype)
        valid_count = numeric_mask.sum(dim=2, keepdim=True).clamp_min(1)
        mean = (features * numeric_mask).sum(dim=2, keepdim=True) / valid_count
        centered = features - mean
        variance = (centered.square() * numeric_mask).sum(
            dim=2,
            keepdim=True,
        ) / valid_count
        normalized = centered * torch.rsqrt(variance + self.epsilon)
        return (normalized * self.weight + self.bias) * numeric_mask


class StyleFreeResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dropout: float,
        upsample: bool = False,
    ) -> None:
        super().__init__()
        self.upsample = upsample
        self.norm1 = MaskedInstanceNorm1d(input_channels)
        self.norm2 = MaskedInstanceNorm1d(output_channels)
        self.conv1 = weight_norm(nn.Conv1d(input_channels, output_channels, 3, padding=1))
        self.conv2 = weight_norm(nn.Conv1d(output_channels, output_channels, 3, padding=1))
        self.dropout = nn.Dropout(dropout)
        self.resample = (
            weight_norm(
                nn.ConvTranspose1d(
                    input_channels,
                    input_channels,
                    kernel_size=3,
                    stride=2,
                    groups=input_channels,
                    padding=1,
                    output_padding=1,
                )
            )
            if upsample
            else nn.Identity()
        )
        self.shortcut_projection = (
            weight_norm(nn.Conv1d(input_channels, output_channels, 1, bias=False))
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(
        self,
        features: Tensor,
        input_mask: Tensor,
        output_mask: Tensor,
    ) -> Tensor:
        expected_frames = features.shape[-1] * (2 if self.upsample else 1)
        if input_mask.shape != (features.shape[0], 1, features.shape[-1]):
            raise ValueError("residual input mask must match feature frames")
        if output_mask.shape != (features.shape[0], 1, expected_frames):
            raise ValueError("residual output mask must match block output frames")
        numeric_output_mask = output_mask.to(dtype=features.dtype)

        residual = F.leaky_relu(self.norm1(features, input_mask), negative_slope=0.2)
        residual = self.resample(residual)
        if self.upsample:
            residual = residual * numeric_output_mask
        residual = self.conv1(self.dropout(residual)) * numeric_output_mask
        residual = F.leaky_relu(
            self.norm2(residual, output_mask),
            negative_slope=0.2,
        )
        residual = self.conv2(self.dropout(residual)) * numeric_output_mask

        shortcut = features * input_mask.to(dtype=features.dtype)
        if self.upsample:
            shortcut = F.interpolate(shortcut, scale_factor=2, mode="nearest")
        shortcut = self.shortcut_projection(shortcut) * numeric_output_mask
        return (residual + shortcut) * (1 / sqrt(2)) * numeric_output_mask


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
