from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ...config.architecture import ConditionDropoutConfig
from .conditioning_inputs import (
    CONDITION_SOURCE_NAMES,
    ConditionChannels,
    ConditionInputs,
    ConditionKeep,
)

__all__ = [
    "AdaLNZero1d",
    "CONDITION_SOURCE_NAMES",
    "ConditionBank",
    "ConditionChannels",
    "ConditionInputs",
    "ConditionKeep",
    "MaskedAttentivePool1d",
    "ProjectedConditions",
]


@dataclass(frozen=True)
class ProjectedConditions:
    phoneme: Tensor
    style: Tensor
    voice: Tensor
    pooled_phoneme: Tensor
    pre_text: Tensor
    post_text: Tensor
    pre_audio: Tensor
    post_audio: Tensor
    language: Tensor

    def slice_from(self, start: int) -> "ProjectedConditions":
        return ProjectedConditions(
            phoneme=self.phoneme[start:],
            style=self.style[start:],
            voice=self.voice[start:],
            pooled_phoneme=self.pooled_phoneme[start:],
            pre_text=self.pre_text[start:],
            post_text=self.post_text[start:],
            pre_audio=self.pre_audio[start:],
            post_audio=self.post_audio[start:],
            language=self.language[start:],
        )

    def combined(self) -> Tensor:
        return (
            self.phoneme
            + self.style
            + self.voice
            + self.pooled_phoneme
            + self.pre_text
            + self.post_text
            + self.pre_audio
            + self.post_audio
            + self.language
        )

    def concatenated(self) -> Tensor:
        return torch.cat(
            (
                self.phoneme,
                self.style,
                self.voice,
                self.pooled_phoneme,
                self.pre_text,
                self.post_text,
                self.pre_audio,
                self.post_audio,
                self.language,
            ),
            dim=1,
        )


class ConditionProjector(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv1d(input_channels, output_channels, 1)

    def forward(self, tokens: Tensor, keep: Tensor) -> Tensor:
        token_mask = tokens.abs().sum(dim=1, keepdim=True) > 0
        return self.projection(tokens) * token_mask * keep


class ConditionBank(nn.Module):
    def __init__(self, input_channels: ConditionChannels, common_channels: int) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict(
            {
                name: ConditionProjector(channels, common_channels)
                for name, channels in (
                    ("phoneme", input_channels.phoneme),
                    ("style", input_channels.style),
                    ("voice", input_channels.voice),
                    ("pooled_phoneme", input_channels.pooled_phoneme),
                    ("pre_text", input_channels.pre_text),
                    ("post_text", input_channels.post_text),
                    ("pre_audio", input_channels.pre_audio),
                    ("post_audio", input_channels.post_audio),
                    ("language", input_channels.language),
                )
            }
        )

    @staticmethod
    def _sample_keep(
        batch_size: int,
        device: torch.device,
        probability: float,
        generator: torch.Generator,
    ) -> Tensor:
        return (
            torch.rand(
                batch_size,
                1,
                1,
                device=device,
                generator=generator,
            )
            >= probability
        )

    def sample_keep(
        self,
        batch_size: int,
        device: torch.device,
        probabilities: ConditionDropoutConfig,
        generator: torch.Generator,
    ) -> ConditionKeep:
        return ConditionKeep(
            phoneme=self._sample_keep(
                batch_size, device, probabilities.phoneme_embedding, generator
            ),
            style=self._sample_keep(
                batch_size, device, probabilities.style, generator
            ),
            voice=self._sample_keep(
                batch_size, device, probabilities.voice, generator
            ),
            pooled_phoneme=self._sample_keep(
                batch_size, device, probabilities.pooled_phoneme, generator
            ),
            pre_text=self._sample_keep(
                batch_size, device, probabilities.pre_text, generator
            ),
            post_text=self._sample_keep(
                batch_size, device, probabilities.post_text, generator
            ),
            pre_audio=self._sample_keep(
                batch_size, device, probabilities.pre_audio, generator
            ),
            post_audio=self._sample_keep(
                batch_size, device, probabilities.post_audio, generator
            ),
            language=self._sample_keep(
                batch_size, device, probabilities.language, generator
            ),
        )

    def forward(
        self,
        inputs: ConditionInputs,
        keep: ConditionKeep,
    ) -> ProjectedConditions:
        return ProjectedConditions(
            **{
                name: self.projectors[name](
                    getattr(inputs, name),
                    getattr(keep, name),
                )
                for name in CONDITION_SOURCE_NAMES
            }
        )


class AdaLNZero1d(nn.Module):
    def __init__(self, channels: int, condition_channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.modulation = nn.Conv1d(condition_channels, channels * 3, 1)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, features: Tensor, condition: Tensor, mask: Tensor) -> Tensor:
        numeric_mask = mask.to(dtype=features.dtype)
        normalized = F.layer_norm(
            features.transpose(1, 2),
            (self.channels,),
        ).transpose(1, 2)
        scale, shift, gate = self.modulation(condition * numeric_mask).chunk(3, dim=1)
        residual = (normalized * (1 + scale) + shift) * torch.tanh(gate)
        return (features + residual) * numeric_mask


class MaskedAttentivePool1d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        attention_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_channels, attention_channels, 1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, 1, 1),
        )
        self.output = nn.Linear(input_channels * 2, output_channels)

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        torch._assert_async(
            torch.all(mask.sum(dim=2) > 0),
            "attentive pooling requires a valid token per item",
        )
        logits = self.attention(features * mask).masked_fill(~mask, -torch.inf)
        weights = torch.softmax(logits, dim=2)
        mean = (features * weights).sum(dim=2)
        variance = ((features - mean.unsqueeze(2)).square() * weights).sum(dim=2)
        statistics = torch.cat((mean, torch.sqrt(variance.clamp_min(1e-5))), dim=1)
        return self.output(statistics)


def align_phoneme_tokens(
    tokens: Tensor,
    hard_alignment: Tensor,
) -> tuple[Tensor, Tensor]:
    numeric_alignment = hard_alignment.to(dtype=tokens.dtype)
    aligned_mask = hard_alignment.to(dtype=torch.bool).any(dim=1, keepdim=True)
    aligned = torch.bmm(tokens, numeric_alignment)
    return aligned * aligned_mask.to(dtype=aligned.dtype), aligned_mask
