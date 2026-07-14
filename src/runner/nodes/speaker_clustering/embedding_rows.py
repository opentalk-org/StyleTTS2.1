from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PreparedSpeakerBatch:
    """A duration-sorted ECAPA input batch and its caller-order mapping."""

    waveforms: torch.Tensor
    relative_lengths: torch.Tensor
    original_indices: torch.Tensor
