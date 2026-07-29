from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class TrainingBatch:
    waves: tuple[np.ndarray, ...]
    texts: Tensor
    input_lengths: Tensor
    reference_texts: Tensor
    reference_lengths: Tensor
    mels: Tensor
    mel_lengths: Tensor
    reference_mels: Tensor

    def to(self, device: torch.device) -> "TrainingBatch":
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = (
                value.to(device, non_blocking=True)
                if isinstance(value, Tensor)
                else value
            )
        return TrainingBatch(**values)


@dataclass(frozen=True)
class ValidationResult:
    metrics: dict[str, Tensor | float]
    samples: list[dict[str, str]]
