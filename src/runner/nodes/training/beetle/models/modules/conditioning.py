from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ...config.architecture import ConditionDropoutConfig


@dataclass(frozen=True)
class ConditionInputs:
    phoneme: Tensor
    style: Tensor
    voice: Tensor
    pooled_phoneme: Tensor
    pre_text: Tensor
    post_text: Tensor
    pre_audio: Tensor
    post_audio: Tensor

    @classmethod
    def from_shared(cls, value: Tensor) -> "ConditionInputs":
        return cls(value, value, value, value, value, value, value, value)


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
        )


class ConditionProjector(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv1d(input_channels, output_channels, 1)

    def forward(self, tokens: Tensor, keep: Tensor) -> Tensor:
        if tokens.ndim != 3 or keep.shape != (tokens.shape[0], 1, 1):
            raise ValueError("condition tokens require [B,C,T] and [B,1,1] keep mask")
        token_mask = tokens.abs().sum(dim=1, keepdim=True) > 0
        return self.projection(tokens) * token_mask * keep


class ConditionBank(nn.Module):
    def __init__(self, input_channels: int, common_channels: int) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict(
            {
                name: ConditionProjector(input_channels, common_channels)
                for name in (
                    "phoneme",
                    "style",
                    "voice",
                    "pooled_phoneme",
                    "pre_text",
                    "post_text",
                    "pre_audio",
                    "post_audio",
                )
            }
        )

    def _project(
        self,
        name: str,
        tokens: Tensor,
        probability: float,
        generator: torch.Generator,
    ) -> Tensor:
        keep = torch.rand(
            tokens.shape[0],
            1,
            1,
            device=tokens.device,
            generator=generator,
        ) >= probability
        return self.projectors[name](tokens, keep)

    def forward(
        self,
        inputs: ConditionInputs,
        probabilities: ConditionDropoutConfig,
        generator: torch.Generator,
    ) -> ProjectedConditions:
        return ProjectedConditions(
            phoneme=self._project(
                "phoneme",
                inputs.phoneme,
                probabilities.phoneme_embedding,
                generator,
            ),
            style=self._project("style", inputs.style, probabilities.style, generator),
            voice=self._project("voice", inputs.voice, probabilities.voice, generator),
            pooled_phoneme=self._project(
                "pooled_phoneme",
                inputs.pooled_phoneme,
                probabilities.pooled_phoneme,
                generator,
            ),
            pre_text=self._project(
                "pre_text", inputs.pre_text, probabilities.pre_text, generator
            ),
            post_text=self._project(
                "post_text", inputs.post_text, probabilities.post_text, generator
            ),
            pre_audio=self._project(
                "pre_audio", inputs.pre_audio, probabilities.pre_audio, generator
            ),
            post_audio=self._project(
                "post_audio", inputs.post_audio, probabilities.post_audio, generator
            ),
        )


class AdaLNZero1d(nn.Module):
    def __init__(self, channels: int, condition_channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.modulation = nn.Conv1d(condition_channels, channels * 3, 1)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, features: Tensor, condition: Tensor, mask: Tensor) -> Tensor:
        if features.ndim != 3 or mask.shape != (features.shape[0], 1, features.shape[2]):
            raise ValueError("AdaLN requires [B,C,T] features and [B,1,T] mask")
        if condition.shape[0] != features.shape[0] or condition.shape[2] != features.shape[2]:
            raise ValueError("AdaLN condition must match batch and token dimensions")
        numeric_mask = mask.to(dtype=features.dtype)
        normalized = F.layer_norm(
            features.transpose(1, 2),
            (self.channels,),
        ).transpose(1, 2)
        scale, shift, gate = self.modulation(condition * numeric_mask).chunk(3, dim=1)
        residual = (normalized * (1 + scale) + shift) * torch.tanh(gate)
        return (features + residual) * numeric_mask
