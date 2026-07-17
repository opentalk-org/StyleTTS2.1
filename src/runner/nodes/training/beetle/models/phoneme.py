from dataclasses import dataclass

from torch import Tensor, nn
from transformers import AlbertModel

from ..config.architecture import PhonemeConfig
from .modules.convolution import DilatedResidualStack


@dataclass(frozen=True)
class PhonemeEncoding:
    tokens: Tensor
    pooled: Tensor
    mask: Tensor


class PhonemeEncoder(nn.Module):
    def __init__(self, albert: AlbertModel, output_channels: int) -> None:
        super().__init__()
        self.albert = albert
        self.projection = nn.Conv1d(albert.config.hidden_size, output_channels, 1)

    def forward(self, input_ids: Tensor, mask: Tensor) -> PhonemeEncoding:
        if input_ids.shape != mask.shape:
            raise ValueError("phoneme ids and mask must have equal shapes")
        encoded = self.albert(
            input_ids=input_ids,
            attention_mask=mask,
            return_dict=True,
        ).last_hidden_state.transpose(1, 2)
        numeric_mask = mask.unsqueeze(1).to(dtype=encoded.dtype)
        tokens = self.projection(encoded) * numeric_mask
        pooled = tokens.sum(dim=2) / numeric_mask.sum(dim=2).clamp_min(1)
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


class ContextPhonemeEncoder(PhonemeCnnEncoder):
    """Projects neighboring text for boundary conditioning."""
