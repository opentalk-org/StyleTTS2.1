from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config.architecture import EmbeddingEncoderConfig
from .modules.convolution import DilatedResidualStack
from .modules.pooling import MaskedAttentivePool1d


class EmbeddingEncoder(nn.Module):
    def __init__(self, config: EmbeddingEncoderConfig) -> None:
        super().__init__()
        self.input = nn.Conv1d(config.input_channels, config.hidden_channels, 1)
        dilations = tuple(2 ** (index % 5) for index in range(config.layers))
        self.stack = DilatedResidualStack(
            config.hidden_channels,
            kernel_size=3,
            dilations=dilations,
            cycles=1,
            dropout=0.0,
        )
        self.pool = MaskedAttentivePool1d(
            config.hidden_channels,
            config.attention_channels,
            config.embedding_channels,
        )

    def forward(self, latent: Tensor, mask: Tensor) -> Tensor:
        numeric_mask = mask.to(dtype=latent.dtype)
        features = self.input(latent * numeric_mask) * numeric_mask
        features = self.stack(features, numeric_mask)
        return F.normalize(self.pool(features, mask.to(dtype=torch.bool)), dim=1)


class StyleEncoder(EmbeddingEncoder):
    """Encodes prosodic and acoustic style."""


class VoiceEncoder(EmbeddingEncoder):
    """Encodes speaker identity."""


class _ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(context, embedding: Tensor, scale: float) -> Tensor:
        context.scale = scale
        return embedding.view_as(embedding)

    @staticmethod
    def backward(context, gradient: Tensor) -> tuple[Tensor, None]:
        return -context.scale * gradient, None


class StyleSpeakerClassifier(nn.Module):
    def __init__(self, embedding_channels: int, speaker_classes: int) -> None:
        super().__init__()
        self.output = nn.Linear(embedding_channels, speaker_classes)

    def forward(self, embedding: Tensor, reversal_scale: float) -> Tensor:
        return self.output(_ReverseGradient.apply(embedding, reversal_scale))


@dataclass(frozen=True)
class AcousticStatistics:
    f0_mean: Tensor
    f0_std: Tensor
    n_mean: Tensor
    n_std: Tensor


class StyleStatisticsHead(nn.Module):
    def __init__(self, embedding_channels: int) -> None:
        super().__init__()
        self.output = nn.Linear(embedding_channels, 4)

    def forward(self, embedding: Tensor) -> AcousticStatistics:
        f0_mean, f0_scale, n_mean, n_scale = self.output(embedding).unbind(dim=1)
        return AcousticStatistics(
            f0_mean=f0_mean,
            f0_std=F.softplus(f0_scale) + 1e-5,
            n_mean=n_mean,
            n_std=F.softplus(n_scale) + 1e-5,
        )
