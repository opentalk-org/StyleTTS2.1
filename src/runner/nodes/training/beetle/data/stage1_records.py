from dataclasses import dataclass

import torch
from torch import Tensor

from .records import IndexedSegment, SegmentKey


@dataclass(frozen=True, order=True)
class Stage1WindowPlan:
    key: SegmentKey
    latent_start: int
    window_index: int

    def __post_init__(self) -> None:
        if self.latent_start < 0 or self.window_index < 0:
            raise ValueError("Stage 1 window positions must be non-negative")


@dataclass(frozen=True)
class Stage1PlannedBatch:
    windows: tuple[Stage1WindowPlan, ...]

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("Stage 1 planned batch must contain windows")


@dataclass(frozen=True)
class Stage1WindowGeometry:
    sample_rate: int
    hop_length: int
    posterior_rate: int
    latent_frames: int
    context_mel_frames: int

    def __post_init__(self) -> None:
        positive = (
            self.sample_rate,
            self.hop_length,
            self.posterior_rate,
            self.latent_frames,
        )
        if min(positive) <= 0:
            raise ValueError("Stage 1 window geometry must be positive")
        if (
            self.context_mel_frames < 0
            or self.context_mel_frames % self.posterior_rate
        ):
            raise ValueError("Stage 1 context must align to posterior frames")

    @property
    def target_mel_frames(self) -> int:
        return self.latent_frames * self.posterior_rate

    @property
    def target_samples(self) -> int:
        return self.target_mel_frames * self.hop_length

    @property
    def encoder_mel_frames(self) -> int:
        return self.target_mel_frames + 2 * self.context_mel_frames

    @property
    def posterior_start(self) -> int:
        return self.context_mel_frames // self.posterior_rate

    def mel_frames(self, item: IndexedSegment) -> int:
        start = round(item.start * item.sample_rate)
        end = round(item.end * item.sample_rate)
        source_samples = end - start
        output_samples = (
            source_samples * self.sample_rate + item.sample_rate - 1
        ) // item.sample_rate
        return output_samples // self.hop_length + 1

    def plans(self, item: IndexedSegment) -> tuple[Stage1WindowPlan, ...]:
        latent_count = self.mel_frames(item) // self.posterior_rate
        if latent_count < self.latent_frames:
            return (Stage1WindowPlan(item.key, 0, 0),)
        final_start = latent_count - self.latent_frames
        starts = list(range(0, final_start + 1, self.latent_frames))
        if starts[-1] != final_start:
            starts.append(final_start)
        return tuple(
            Stage1WindowPlan(item.key, start, index)
            for index, start in enumerate(starts)
        )


@dataclass(frozen=True)
class FetchedStage1Source:
    item: IndexedSegment
    waveform: Tensor
    mel: Tensor


@dataclass(frozen=True)
class FetchedStage1Batch:
    plans: tuple[Stage1WindowPlan, ...]
    sources: tuple[FetchedStage1Source, ...]


@dataclass(frozen=True)
class Stage1Batch:
    encoder_mel: Tensor
    encoder_mask: Tensor
    target_mel: Tensor
    frame_mask: Tensor
    waveform: Tensor
    sample_keys: tuple[SegmentKey, ...]
    window_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        batch = len(self.sample_keys)
        if self.encoder_mel.shape[0] != batch or self.target_mel.shape[0] != batch:
            raise ValueError("Stage 1 tensor batch must match sample keys")
        if self.waveform.shape[0] != batch or len(self.window_indices) != batch:
            raise ValueError("Stage 1 targets must match sample keys")
        if (
            self.encoder_mask.dtype is not torch.bool
            or self.frame_mask.dtype is not torch.bool
        ):
            raise TypeError("Stage 1 masks must be Boolean")

    def to(self, device: torch.device) -> "Stage1Batch":
        return Stage1Batch(
            encoder_mel=self.encoder_mel.to(device, non_blocking=True),
            encoder_mask=self.encoder_mask.to(device, non_blocking=True),
            target_mel=self.target_mel.to(device, non_blocking=True),
            frame_mask=self.frame_mask.to(device, non_blocking=True),
            waveform=self.waveform.to(device, non_blocking=True),
            sample_keys=self.sample_keys,
            window_indices=self.window_indices,
        )
