from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from shared.db.clickhouse.types import utc_datetime


class DatasetRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    updated_at: datetime
    name: str

    _updated_at_utc = field_validator("updated_at")(utc_datetime)


class DatasetMembership(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: UUID
    audio_file_id: UUID
    updated_at: datetime
    created_at: datetime

    _timestamps_utc = field_validator("updated_at", "created_at")(utc_datetime)
