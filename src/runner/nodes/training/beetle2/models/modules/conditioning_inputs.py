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
