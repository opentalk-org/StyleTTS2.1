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
