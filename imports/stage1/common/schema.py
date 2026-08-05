from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AlignmentWord(StrictModel):
    word: str
    start: float
    end: float

    @model_validator(mode="after")
    def validate_range(self) -> "AlignmentWord":
        if self.start < 0.0 or self.end < self.start:
            raise ValueError("alignment word must have an ordered non-negative range")
        return self


class SegmentRecord(StrictModel):
    start: float
    end: float
    text: str
    source: Literal["dataset"]
    score: float | None
    accuracy: float | None
    alignment: list[AlignmentWord]
    phon: str = ""

    @model_validator(mode="after")
    def validate_range(self) -> "SegmentRecord":
        if self.start < 0.0 or self.end < self.start:
            raise ValueError("segment must have an ordered non-negative range")
        return self


class AudioRecord(StrictModel):
    path: str
    source_id: str
    duration: float
    language: str
    speaker_id: str | None
    style_prompt: str | None
    voice_prompt: str | None
    score: float | None
    accuracy: float | None
    segments: list[SegmentRecord]
    metadata: dict[str, Any]

    @model_validator(mode="after")
    def validate_audio(self) -> "AudioRecord":
        if self.duration <= 0.0:
            raise ValueError("audio duration must be positive")
        if not self.path.startswith("wavs/") or not self.path.endswith(".wav"):
            raise ValueError("audio path must reference a WAV under wavs/")
        if not self.metadata:
            raise ValueError("audio metadata must retain source provenance")
        if any(segment.end > self.duration + 0.001 for segment in self.segments):
            raise ValueError("segment exceeds audio duration")
        return self


class DatasetRecord(StrictModel):
    name: str
    language_limits_hours: dict[str, float]
    source_url: str


class DatasetManifest(StrictModel):
    dataset: DatasetRecord
    audio_files: list[AudioRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_languages(self) -> "DatasetManifest":
        languages = {record.language for record in self.audio_files}
        unknown = sorted(languages.difference(self.dataset.language_limits_hours))
        if unknown:
            raise ValueError(f"audio languages have no declared target: {unknown}")
        return self
