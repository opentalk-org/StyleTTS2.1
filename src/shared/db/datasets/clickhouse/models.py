from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DatasetRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    updated_at: datetime
    name: str


class DatasetMembership(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: UUID
    audio_file_id: UUID
    updated_at: datetime
    created_at: datetime
