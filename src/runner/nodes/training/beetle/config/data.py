from uuid import UUID

from pydantic import Field, model_validator

from .architecture import StrictConfigModel


class DatabaseSelection(StrictConfigModel):
    dataset_id: UUID
    audio_file_ids: tuple[UUID, ...]


class PrefetchConfig(StrictConfigModel):
    page_size: int = Field(gt=0)
    planned_batches: int = Field(gt=0)
    decoded_bytes: int = Field(gt=0)
    maximum_full_read_bytes: int = Field(gt=0)
    worker_count: int = Field(gt=0)


class GroupSamplingConfig(StrictConfigModel):
    duration_boundaries_seconds: tuple[float, ...] = Field(min_length=1)
    voices_per_batch: int = Field(gt=1)
    utterances_per_voice: int = Field(gt=1)
    recordings_per_batch: int = Field(gt=1)
    cuts_per_recording: int = Field(gt=1)


class AugmentationConfig(StrictConfigModel):
    time_stretch_min: float = Field(gt=0)
    time_stretch_max: float = Field(gt=0)
    pitch_shift_min_semitones: float
    pitch_shift_max_semitones: float
    gain_min_db: float
    gain_max_db: float

    @model_validator(mode="after")
    def validate_ranges(self) -> "AugmentationConfig":
        if self.time_stretch_min > self.time_stretch_max:
            raise ValueError("time-stretch minimum must not exceed maximum")
        if self.pitch_shift_min_semitones > self.pitch_shift_max_semitones:
            raise ValueError("pitch-shift minimum must not exceed maximum")
        if self.gain_min_db > self.gain_max_db:
            raise ValueError("gain minimum must not exceed maximum")
        return self


class DataConfig(StrictConfigModel):
    selection: DatabaseSelection
    minimum_seconds: float = Field(ge=1, le=45)
    maximum_seconds: float = Field(ge=1, le=45)
    sentence_probability: float = Field(gt=0, lt=1)
    prefetch: PrefetchConfig
    grouping: GroupSamplingConfig
    augmentation: AugmentationConfig

    @model_validator(mode="after")
    def validate_durations(self) -> "DataConfig":
        if self.minimum_seconds > self.maximum_seconds:
            raise ValueError("minimum_seconds must not exceed maximum_seconds")
        boundaries = self.grouping.duration_boundaries_seconds
        if tuple(sorted(boundaries)) != boundaries:
            raise ValueError("duration boundaries must be sorted")
        if boundaries[-1] > self.maximum_seconds:
            raise ValueError("duration boundary exceeds maximum_seconds")
        return self
