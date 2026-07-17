from dataclasses import dataclass

import torch
from torch import Tensor


CONDITION_SOURCE_NAMES = (
    "phoneme",
    "style",
    "voice",
    "pooled_phoneme",
    "pre_text",
    "post_text",
    "pre_audio",
    "post_audio",
    "language",
)


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
    language: Tensor

    @classmethod
    def from_shared(cls, value: Tensor) -> "ConditionInputs":
        return cls(value, value, value, value, value, value, value, value, value)

    def dropped_concatenated(self, keep: "ConditionKeep") -> Tensor:
        return torch.cat(
            tuple(
                getattr(self, name) * getattr(keep, name)
                for name in CONDITION_SOURCE_NAMES
            ),
            dim=1,
        )


@dataclass(frozen=True)
class ConditionKeep:
    phoneme: Tensor
    style: Tensor
    voice: Tensor
    pooled_phoneme: Tensor
    pre_text: Tensor
    post_text: Tensor
    pre_audio: Tensor
    post_audio: Tensor
    language: Tensor


@dataclass(frozen=True)
class ConditionChannels:
    phoneme: int
    style: int
    voice: int
    pooled_phoneme: int
    pre_text: int
    post_text: int
    pre_audio: int
    post_audio: int
    language: int

    @classmethod
    def from_shared(cls, channels: int) -> "ConditionChannels":
        return cls(
            channels,
            channels,
            channels,
            channels,
            channels,
            channels,
            channels,
            channels,
            channels,
        )

    def total(self) -> int:
        return sum(getattr(self, name) for name in CONDITION_SOURCE_NAMES)


@dataclass(frozen=True)
class ConditionVectors:
    style: Tensor
    voice: Tensor
    pooled_phoneme: Tensor
    pre_text: Tensor
    post_text: Tensor
    pre_audio: Tensor
    post_audio: Tensor
    language: Tensor

    def at_rate(self, phoneme: Tensor) -> ConditionInputs:
        if phoneme.ndim != 3:
            raise ValueError("phoneme condition must have shape [B,C,T]")
        tokens = phoneme.shape[2]
        vectors = (
            self.style,
            self.voice,
            self.pooled_phoneme,
            self.pre_text,
            self.post_text,
            self.pre_audio,
            self.post_audio,
            self.language,
        )
        if any(value.ndim != 2 or value.shape[0] != phoneme.shape[0] for value in vectors):
            raise ValueError("condition vectors must have shape [B,C]")
        expanded = tuple(value.unsqueeze(2).expand(-1, -1, tokens) for value in vectors)
        return ConditionInputs(phoneme, *expanded)
