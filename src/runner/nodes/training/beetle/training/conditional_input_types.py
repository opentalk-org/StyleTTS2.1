from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from ..data.records import BeetleBatch
from ..models.modules.aligner import AlignerOutput
from ..models.modules.audio import AudioPosterior
from ..models.modules.conditioning import (
    ConditionBank,
    ConditionKeep,
    ConditionVectors,
    ProjectedConditions,
)
from ..models.modules.text import PhonemeEncoding
from .conditional_features import ConditionalAcousticTargets


class SpeakerIndex(Protocol):
    def resolve(
        self,
        speaker_ids: tuple[str | None, ...],
        device: torch.device,
    ) -> Tensor: ...


@dataclass(frozen=True)
class CoreConditionalInput:
    acoustic_targets: ConditionalAcousticTargets
    posterior: AudioPosterior
    alignment: AlignerOutput
    phoneme: PhonemeEncoding
    aligned_tokens: Tensor
    vectors: ConditionVectors
    keep: ConditionKeep


def build_rate_conditions(
    bank: ConditionBank,
    vectors: ConditionVectors,
    duration_phoneme: Tensor,
    latent_phoneme: Tensor,
    keep: ConditionKeep,
) -> tuple[Tensor, ProjectedConditions]:
    duration = vectors.at_rate(duration_phoneme).dropped_concatenated(keep)
    latent = bank(vectors.at_rate(latent_phoneme), keep)
    return duration, latent


def keep_all_conditions(batch_size: int, device: torch.device) -> ConditionKeep:
    if batch_size <= 0:
        raise ValueError("condition batch size must be positive")
    keep = torch.ones(batch_size, 1, 1, dtype=torch.bool, device=device)
    return ConditionKeep(keep, keep, keep, keep, keep, keep, keep, keep, keep)


def require_batch(batch: object) -> BeetleBatch:
    if not isinstance(batch, BeetleBatch):
        raise TypeError("conditional input builder requires a BeetleBatch")
    return batch
