from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import Tensor

from ...studio.val_sample_export import ValidationSampleArtifacts


@dataclass(frozen=True)
class TrainingBatch:
    waves: tuple[np.ndarray, ...]
    audio_durations: tuple[float, ...]
    bucket_fetch_seconds: float
    bucket_fetch_bytes: int
    bucket_request_seconds: float
    bucket_request_count: int
    bucket_error_count: int
    speaker_ids: Tensor
    language_ids: Tensor
    modality_ids: Tensor
    texts: Tensor
    input_lengths: Tensor
    mels: Tensor
    mel_lengths: Tensor

    def to(self, device: torch.device) -> "TrainingBatch":
        cpu_fields = {
            "input_lengths",
            "mel_lengths",
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
