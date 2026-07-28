from uuid import UUID

from pydantic import Field, model_validator

from .architecture import StrictConfigModel


class DatabaseSelection(StrictConfigModel):
    dataset_id: UUID
    audio_file_ids: tuple[UUID, ...]


class PrefetchConfig(StrictConfigModel):
    page_size: int = Field(gt=0)
    window_size: int = Field(gt=0)
    decoded_bytes: int = Field(gt=0)
    preprocessing_threads: int = Field(gt=0)


class GroupSamplingConfig(StrictConfigModel):
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
    maximum_seconds: float = Field(ge=1, le=45)
    prefetch: PrefetchConfig
    grouping: GroupSamplingConfig
    augmentation: AugmentationConfig
