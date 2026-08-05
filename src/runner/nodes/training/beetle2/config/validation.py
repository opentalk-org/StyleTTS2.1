from uuid import UUID

from pydantic import Field

from .architecture import StrictConfigModel


class ValidationConfig(StrictConfigModel):
    sample_count: int = Field(gt=0)
    audio_file_ids: tuple[UUID, ...]
