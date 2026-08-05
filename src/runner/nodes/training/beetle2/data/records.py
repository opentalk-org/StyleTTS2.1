from dataclasses import dataclass, fields
from uuid import UUID

import torch
from torch import Tensor

from shared.db.audio.ranges import WavClip


@dataclass(frozen=True, order=True)
class SegmentKey:
    audio_file_id: UUID
    segment_index: int
    segment_id: str

@dataclass(frozen=True)
class WordBoundary:
    word: str
    start: float
    end: float

@dataclass(frozen=True)
class IndexedSegment:
    key: SegmentKey
    audio_name: str
    start: float
    end: float
    audio_duration: float
    sample_rate: int
    estimated_bytes: int
    language: str | None
    speaker_id: str | None
    has_text: bool
    has_phonemes: bool
    phoneme_word_count: int
    words: tuple[WordBoundary, ...]
    style_prompt: str | None
    voice_prompt: str | None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def mid_sentence_eligible(self) -> bool:
        return len(self.words) >= 2 and len(self.words) == self.phoneme_word_count


@dataclass(frozen=True)
class CutRange:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class PlannedExample:
    key: SegmentKey
    target: CutRange
    target_word_start: int
    target_word_end: int
    sentence: bool
    seed: int

@dataclass(frozen=True)
class EmbeddingViewPlan:
    key: SegmentKey
    audio: CutRange
    seed: int
    distance_seconds: float

@dataclass(frozen=True)
class EmbeddingGroupPlan:
    group_id: str
    views: tuple[EmbeddingViewPlan, ...]

@dataclass(frozen=True)
class PlannedBatch:
    examples: tuple[PlannedExample, ...]
    voice_groups: tuple[EmbeddingGroupPlan, ...]
    style_groups: tuple[EmbeddingGroupPlan, ...]
    voice_condition_indices: tuple[int, ...]
    voice_auxiliary_view_count: int

@dataclass(frozen=True)
class FetchedExample:
    plan: PlannedExample
    text: str
    phonemes: str
    target_clip: WavClip
    style_prompt: str | None
    voice_prompt: str | None
    speaker_id: str | None
    language: str | None


@dataclass(frozen=True)
class FetchedEmbeddingView:
    plan: EmbeddingViewPlan
    clip: WavClip


@dataclass(frozen=True)
class FetchedEmbeddingGroup:
    group_id: str
    views: tuple[FetchedEmbeddingView, ...]


@dataclass(frozen=True)
class FetchedBatch:
    examples: tuple[FetchedExample, ...]
    voice_groups: tuple[FetchedEmbeddingGroup, ...]
    style_groups: tuple[FetchedEmbeddingGroup, ...]
    voice_condition_indices: tuple[int, ...]
    voice_auxiliary_view_count: int


@dataclass(frozen=True)
class BeetleBatch:
    waveform: Tensor
    mel: Tensor
    jdc_mel: Tensor
    phoneme_ids: Tensor
    text_input_ids: Tensor
    language_ids: Tensor
    alignments: Tensor
    durations: Tensor
    style_prompt_ids: Tensor
    voice_prompt_ids: Tensor
    style_views: Tensor
    voice_views: Tensor
    voice_condition_indices: Tensor
    voice_auxiliary_view_count: int
    waveform_lengths: Tensor
    frame_lengths: Tensor
    phoneme_lengths: Tensor
    text_lengths: Tensor
    style_prompt_lengths: Tensor
    voice_prompt_lengths: Tensor
    style_view_lengths: Tensor
    voice_view_lengths: Tensor
    frame_mask: Tensor
    phoneme_mask: Tensor
    text_mask: Tensor
    style_prompt_available: Tensor
    voice_prompt_available: Tensor
    style_distances: Tensor
    view_seeds: Tensor
    sample_keys: tuple[SegmentKey, ...]
    speaker_ids: tuple[str | None, ...]
    recording_ids: tuple[UUID, ...]
    style_group_ids: tuple[str, ...]
    voice_group_ids: tuple[str, ...]

    def to(self, device: torch.device) -> "BeetleBatch":
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.to(device) if isinstance(value, Tensor) else value
        return BeetleBatch(**values)
