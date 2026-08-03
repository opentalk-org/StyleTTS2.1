from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import Tensor

from ...studio.val_sample_export import ValidationSampleArtifacts


@dataclass(frozen=True)
class TrainingBatch:
    waves: tuple[np.ndarray, ...]
    speaker_ids: Tensor
    texts: Tensor
    input_lengths: Tensor
    reference_texts: Tensor
    reference_lengths: Tensor
    mels: Tensor
    mel_lengths: Tensor
    reference_mels: Tensor
    reference_mel_lengths: Tensor

    def to(self, device: torch.device) -> "TrainingBatch":
        cpu_fields = {
            "input_lengths",
            "reference_lengths",
            "mel_lengths",
            "reference_mel_lengths",
        }
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = (
                value.to(device, non_blocking=True)
                if isinstance(value, Tensor) and field.name not in cpu_fields
                else value
            )
        return TrainingBatch(**values)


@dataclass(frozen=True)
class ValidationResult:
    metrics: dict[str, Tensor | float]
    samples: list[ValidationSampleArtifacts]
