from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from shared.db.clickhouse.types import utc_datetime


class MosComparisonRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    updated_at: datetime
    audio_a_id: UUID
    audio_b_id: UUID
    preferred_audio_id: UUID
    score_a: float
    score_b: float
    created_at: datetime

    _timestamps_utc = field_validator("updated_at", "created_at")(utc_datetime)


class MosPairIds(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: UUID
    audio_a_id: UUID
    audio_b_id: UUID
