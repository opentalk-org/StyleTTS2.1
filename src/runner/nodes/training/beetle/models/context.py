import torch
from torch import Tensor, nn

from ..config.architecture import ContextConfig
from .modules.convolution import DilatedResidualStack
from .modules.pooling import MaskedAttentivePool1d


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
            config.output_channels,
        )

    def forward(self, latent: Tensor, mask: Tensor) -> Tensor:
        numeric_mask = mask.to(dtype=latent.dtype)
        features = self.input(latent * numeric_mask) * numeric_mask
        features = self.stack(features, numeric_mask)
        return self.pool(features, mask.to(dtype=torch.bool))
