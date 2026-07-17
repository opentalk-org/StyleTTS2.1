from uuid import UUID

from pydantic import Field, model_validator

from .architecture import StrictConfigModel


class ValidationConfig(StrictConfigModel):
    audio_file_ids: tuple[UUID, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_audio_files(self) -> "ValidationConfig":
        if len(set(self.audio_file_ids)) != len(self.audio_file_ids):
            raise ValueError("validation audio_file_ids must be unique")
        return self
