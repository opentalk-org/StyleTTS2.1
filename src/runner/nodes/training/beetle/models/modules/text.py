from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import BertModel, PreTrainedModel

from ...config.architecture import ContextConfig, PhonemeConfig
from .conditioning import MaskedAttentivePool1d
from .convolution import DilatedResidualStack


@dataclass(frozen=True)
class PhonemeEncoding:
    tokens: Tensor
    pooled: Tensor
    mask: Tensor


class PhonemeEncoder(nn.Module):
    def __init__(self, bert: PreTrainedModel, output_channels: int) -> None:
        super().__init__()
        self.bert = bert
        self.projection = nn.Conv1d(bert.config.hidden_size, output_channels, 1)

    def forward(self, input_ids: Tensor, mask: Tensor) -> PhonemeEncoding:
        # PL-BERT has learned positions for 512 tokens, while a recording may be
        # much longer. Independent windows preserve every phoneme without
        # inventing positional embeddings outside the pretrained range.
        window_size = self.bert.config.max_position_embeddings
        windows = []
        for start in range(0, input_ids.shape[1], window_size):
            end = start + window_size
            windows.append(
                self.bert(
                    input_ids=input_ids[:, start:end],
                    attention_mask=mask[:, start:end],
                    return_dict=True,
                ).last_hidden_state
            )
        encoded = torch.cat(windows, dim=1).transpose(1, 2)
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


@dataclass(frozen=True)
class PromptEncoding:
    style: Tensor
    voice: Tensor


class TextEncoder(nn.Module):
    def __init__(self, bert: BertModel, output_channels: int) -> None:
        super().__init__()
        self.bert = bert
        self.style_projection = nn.Linear(bert.config.hidden_size, output_channels)
        self.voice_projection = nn.Linear(bert.config.hidden_size, output_channels)

    def forward(self, input_ids: Tensor, mask: Tensor) -> PromptEncoding:
        tokens = self.bert(
            input_ids=input_ids,
            attention_mask=mask,
            return_dict=True,
        ).last_hidden_state
        numeric_mask = mask.unsqueeze(2).to(dtype=tokens.dtype)
        pooled = (tokens * numeric_mask).sum(dim=1)
        pooled = pooled / numeric_mask.sum(dim=1).clamp_min(1)
        return PromptEncoding(
            style=self.style_projection(pooled),
            voice=self.voice_projection(pooled),
        )
