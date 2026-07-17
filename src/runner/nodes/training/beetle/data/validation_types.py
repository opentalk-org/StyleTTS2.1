from dataclasses import dataclass
from uuid import UUID

from torch import Tensor

from shared.db.audio.ranges.wav import WavClip

from .audio import ProcessedAudio
from .records import BeetleBatch


@dataclass(frozen=True)
class ValidationSegment:
    segment_id: str
    start: float
    end: float
    text: str
    phonemes: str
    voice_id: str | None

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("validation segment ID must not be empty")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("validation segment range is invalid")


@dataclass(frozen=True)
class StoredValidationAudio:
    audio_file_id: UUID
    duration: float
    language: str | None
    style_prompt: str | None
    voice_prompt: str | None
    virtual: bool
    storage_kind: str
    segments: tuple[ValidationSegment, ...]
    clip: WavClip


@dataclass(frozen=True)
class PreparedValidationAudio:
    stored: StoredValidationAudio
    processed: ProcessedAudio


@dataclass(frozen=True)
class ValidationSource:
    stage_number: int
    items: tuple[PreparedValidationAudio, ...]


@dataclass(frozen=True)
class ValidationRecording:
    audio_file_id: UUID
    batch: BeetleBatch
    waveform: Tensor
