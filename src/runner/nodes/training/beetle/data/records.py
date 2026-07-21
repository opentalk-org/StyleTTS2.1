from dataclasses import dataclass, fields
from uuid import UUID

import torch
from torch import Tensor


@dataclass(frozen=True, order=True)
class SegmentKey:
    audio_file_id: UUID
    segment_index: int
    segment_id: str

    def __post_init__(self) -> None:
        if self.segment_index < 0:
            raise ValueError("segment_index must be non-negative")
        if not self.segment_id:
            raise ValueError("segment_id must not be empty")


@dataclass(frozen=True)
class WordBoundary:
    word: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if not self.word:
            raise ValueError("aligned word must not be empty")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("aligned word range is invalid")


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

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid segment range: {self.key}")
        if self.end > self.audio_duration:
            raise ValueError(f"segment exceeds audio duration: {self.key}")
        if self.sample_rate <= 0:
            raise ValueError(f"segment sample rate is invalid: {self.key}")
        if self.estimated_bytes <= 0:
            raise ValueError(f"estimated bytes must be positive: {self.key}")

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

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("cut range must be positive and ordered")

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class ContextRange:
    key: SegmentKey
    audio: CutRange
    word_start: int
    word_end: int
    speaker_id: str | None

    def __post_init__(self) -> None:
        if self.word_start < 0 or self.word_end < self.word_start:
            raise ValueError("context word range is invalid")


@dataclass(frozen=True)
class PlannedExample:
    key: SegmentKey
    target: CutRange
    target_word_start: int
    target_word_end: int
    pre_context: ContextRange | None
    post_context: ContextRange | None
    sentence: bool
    seed: int

    def __post_init__(self) -> None:
        if self.target_word_start < 0 or self.target_word_end < self.target_word_start:
            raise ValueError("target word range is invalid")


@dataclass(frozen=True)
class EmbeddingViewPlan:
    key: SegmentKey
    audio: CutRange
    seed: int
    distance_seconds: float

    def __post_init__(self) -> None:
        if self.distance_seconds < 0:
            raise ValueError("embedding-view distance must be non-negative")


@dataclass(frozen=True)
class EmbeddingGroupPlan:
    group_id: str
    views: tuple[EmbeddingViewPlan, ...]

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("embedding group ID must not be empty")
        if len(self.views) < 2:
            raise ValueError("embedding groups require at least two views")


@dataclass(frozen=True)
class PlannedBatch:
    examples: tuple[PlannedExample, ...]
    voice_groups: tuple[EmbeddingGroupPlan, ...]
    style_groups: tuple[EmbeddingGroupPlan, ...]

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("planned batch must contain reconstruction examples")


@dataclass(frozen=True)
class DecodedExample:
    plan: PlannedExample
    waveform: Tensor
    mel: Tensor
    phoneme_ids: Tensor
    text_input_ids: Tensor
    alignment: Tensor
    durations: Tensor
    pre_audio: Tensor
    post_audio: Tensor
    pre_text_ids: Tensor
    post_text_ids: Tensor
    style_view: Tensor
    voice_view: Tensor
    speaker_id: str
    recording_id: UUID


@dataclass(frozen=True)
class BeetleBatch:
    waveform: Tensor
    mel: Tensor
    phoneme_ids: Tensor
    text_input_ids: Tensor
    language_ids: Tensor
    alignments: Tensor
    durations: Tensor
    pre_audio: Tensor
    post_audio: Tensor
    pre_text_ids: Tensor
    post_text_ids: Tensor
    style_prompt_ids: Tensor
    voice_prompt_ids: Tensor
    style_views: Tensor
    voice_views: Tensor
    waveform_lengths: Tensor
    frame_lengths: Tensor
    phoneme_lengths: Tensor
    text_lengths: Tensor
    pre_audio_lengths: Tensor
    post_audio_lengths: Tensor
    pre_text_lengths: Tensor
    post_text_lengths: Tensor
    style_prompt_lengths: Tensor
    voice_prompt_lengths: Tensor
    style_view_lengths: Tensor
    voice_view_lengths: Tensor
    frame_mask: Tensor
    phoneme_mask: Tensor
    text_mask: Tensor
    pre_audio_available: Tensor
    post_audio_available: Tensor
    pre_text_available: Tensor
    post_text_available: Tensor
    style_prompt_available: Tensor
    voice_prompt_available: Tensor
    style_distances: Tensor
    view_seeds: Tensor
    sample_keys: tuple[SegmentKey, ...]
    speaker_ids: tuple[str | None, ...]
    recording_ids: tuple[UUID, ...]
    style_group_ids: tuple[str, ...]
    voice_group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        batch_size, channels, samples = self.waveform.shape
        if channels != 1:
            raise ValueError("waveform must be mono")
        if samples != self.mel.shape[-1] * 300:
            raise ValueError("waveform samples must equal mel frames times 300")
        if len(self.sample_keys) != batch_size:
            raise ValueError("sample key count must match batch size")
        if self.frame_mask.dtype != torch.bool:
            raise TypeError("frame_mask must be bool")
        if self.phoneme_ids.dtype != torch.long:
            raise TypeError("phoneme_ids must be int64")
        if self.language_ids.shape != (batch_size,):
            raise ValueError("language_ids must have shape [B]")
        if self.language_ids.dtype != torch.long:
            raise TypeError("language_ids must be int64")

    def to(self, device: torch.device) -> "BeetleBatch":
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.to(device) if isinstance(value, Tensor) else value
        return BeetleBatch(**values)

    @classmethod
    def synthetic(cls, batch_size: int, frames: int, samples: int) -> "BeetleBatch":
        phonemes = 3
        texts = 4
        audio_context = 300
        zeros = torch.zeros
        ones = torch.ones
        keys = tuple(SegmentKey(UUID(int=index + 1), 0, str(index)) for index in range(batch_size))
        lengths = torch.full((batch_size,), frames, dtype=torch.long)
        return cls(
            waveform=zeros(batch_size, 1, samples),
            mel=zeros(batch_size, 80, frames),
            phoneme_ids=zeros(batch_size, phonemes, dtype=torch.long),
            text_input_ids=zeros(batch_size, texts, dtype=torch.long),
            language_ids=zeros(batch_size, dtype=torch.long),
            alignments=zeros(batch_size, phonemes, frames),
            durations=zeros(batch_size, phonemes),
            pre_audio=zeros(batch_size, 1, audio_context),
            post_audio=zeros(batch_size, 1, audio_context),
            pre_text_ids=zeros(batch_size, texts, dtype=torch.long),
            post_text_ids=zeros(batch_size, texts, dtype=torch.long),
            style_prompt_ids=zeros(batch_size, texts, dtype=torch.long),
            voice_prompt_ids=zeros(batch_size, texts, dtype=torch.long),
            style_views=zeros(batch_size, 2, 1, samples),
            voice_views=zeros(batch_size, 2, 1, samples),
            waveform_lengths=torch.full((batch_size,), samples, dtype=torch.long),
            frame_lengths=lengths,
            phoneme_lengths=torch.full((batch_size,), phonemes, dtype=torch.long),
            text_lengths=torch.full((batch_size,), texts, dtype=torch.long),
            pre_audio_lengths=torch.full((batch_size,), audio_context, dtype=torch.long),
            post_audio_lengths=torch.full((batch_size,), audio_context, dtype=torch.long),
            pre_text_lengths=torch.full((batch_size,), texts, dtype=torch.long),
            post_text_lengths=torch.full((batch_size,), texts, dtype=torch.long),
            style_prompt_lengths=torch.full((batch_size,), texts, dtype=torch.long),
            voice_prompt_lengths=torch.full((batch_size,), texts, dtype=torch.long),
            style_view_lengths=torch.full((batch_size, 2), samples, dtype=torch.long),
            voice_view_lengths=torch.full((batch_size, 2), samples, dtype=torch.long),
            frame_mask=ones(batch_size, 1, frames, dtype=torch.bool),
            phoneme_mask=ones(batch_size, phonemes, dtype=torch.bool),
            text_mask=ones(batch_size, texts, dtype=torch.bool),
            pre_audio_available=ones(batch_size, dtype=torch.bool),
            post_audio_available=ones(batch_size, dtype=torch.bool),
            pre_text_available=ones(batch_size, dtype=torch.bool),
            post_text_available=ones(batch_size, dtype=torch.bool),
            style_prompt_available=ones(batch_size, dtype=torch.bool),
            voice_prompt_available=ones(batch_size, dtype=torch.bool),
            style_distances=zeros(batch_size, 2),
            view_seeds=torch.arange(batch_size, dtype=torch.long),
            sample_keys=keys,
            speaker_ids=tuple(f"voice-{index}" for index in range(batch_size)),
            recording_ids=tuple(key.audio_file_id for key in keys),
            style_group_ids=tuple(f"recording-{index}" for index in range(batch_size)),
            voice_group_ids=tuple(f"voice-{index}" for index in range(batch_size)),
        )
