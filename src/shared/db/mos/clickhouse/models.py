from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class MosComparisonRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    updated_at: datetime
    dataset_id: UUID
    audio_a_id: UUID
    audio_b_id: UUID
    preferred_audio_id: UUID
    score_a: float
    score_b: float
    created_at: datetime


class MosPairIds(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: UUID
    audio_a_id: UUID
    audio_b_id: UUID
