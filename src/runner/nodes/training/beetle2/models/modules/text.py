from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ...config.architecture import ContextConfig, PhonemeConfig
from .conditioning import MaskedAttentivePool1d
from .convolution import DilatedResidualStack


@dataclass(frozen=True)
class PhonemeEncoding:
    tokens: Tensor
    pooled: Tensor
    mask: Tensor


class PhonemeEncoder(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_channels, output_channels)

    def forward(self, encoded: Tensor, mask: Tensor) -> PhonemeEncoding:
        numeric_mask = mask.unsqueeze(2).to(dtype=encoded.dtype)
        tokens = self.projection(encoded) * numeric_mask
        tokens = tokens.transpose(1, 2)
        channel_mask = numeric_mask.transpose(1, 2)
        pooled = tokens.sum(dim=2) / channel_mask.sum(dim=2).clamp_min(1)
        return PhonemeEncoding(tokens=tokens, pooled=pooled, mask=mask.unsqueeze(1))


class PhonemeCnnEncoder(nn.Module):
    def __init__(self, config: PhonemeConfig) -> None:
        super().__init__()
        self.input = nn.Conv1d(
            config.projection_channels,
            config.cnn_hidden_channels,
            1,
        )
        dilations = tuple(2 ** (index % 5) for index in range(config.cnn_layers))
        self.stack = DilatedResidualStack(
            config.cnn_hidden_channels,
            config.cnn_kernel_size,
            dilations,
            cycles=1,
            dropout=config.dropout,
        )

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        numeric_mask = mask.to(dtype=tokens.dtype)
        features = self.input(tokens * numeric_mask) * numeric_mask
        return self.stack(features, numeric_mask)


class LatentPhonemeEncoder(PhonemeCnnEncoder):
    """Projects phoneme tokens for latent generation."""


class DurationPhonemeEncoder(PhonemeCnnEncoder):
    """Projects phoneme tokens for duration likelihood."""


class ContextPhonemeEncoder(nn.Module):
    def __init__(self, input_channels: int, config: ContextConfig) -> None:
        super().__init__()
        self.input = nn.Conv1d(input_channels, config.hidden_channels, 1)
        dilations = tuple(2 ** index for index in range(config.layers))
        self.stack = DilatedResidualStack(
            config.hidden_channels,
            config.kernel_size,
            dilations,
            cycles=1,
            dropout=config.dropout,
        )
        self.pool = MaskedAttentivePool1d(
            config.hidden_channels,
            config.hidden_channels,
            config.output_channels,
        )

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        numeric_mask = mask.to(dtype=tokens.dtype)
        features = self.input(tokens * numeric_mask) * numeric_mask
        features = self.stack(features, numeric_mask)
        return self.pool(features, mask.to(dtype=torch.bool))


class ContextAudioEncoder(nn.Module):
    def __init__(self, input_channels: int, config: ContextConfig) -> None:
        super().__init__()
        self.input = nn.Conv1d(input_channels, config.hidden_channels, 1)
        dilations = tuple(2 ** (index % 5) for index in range(config.layers))
        self.stack = DilatedResidualStack(
            config.hidden_channels,
            config.kernel_size,
            dilations,
            cycles=1,
            dropout=config.dropout,
        )
        self.pool = MaskedAttentivePool1d(
            config.hidden_channels,
            config.hidden_channels,
            config.output_channels,
        )

    def forward(self, latent: Tensor, mask: Tensor) -> Tensor:
        numeric_mask = mask.to(dtype=latent.dtype)
        features = self.input(latent * numeric_mask) * numeric_mask
        features = self.stack(features, numeric_mask)
        return self.pool(features, mask.to(dtype=torch.bool))
