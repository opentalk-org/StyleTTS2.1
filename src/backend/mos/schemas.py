from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class MosAudioRead(BaseModel):
    id: UUID
    name: str
    duration: float
    score: float | None
    speaker: str


class MosPairRead(BaseModel):
    dataset_id: UUID
    audio_a: MosAudioRead
    audio_b: MosAudioRead


class MosRatingRead(BaseModel):
    id: UUID
    dataset_id: UUID
    audio_a_id: UUID
    audio_b_id: UUID
    preferred_audio_id: UUID
    score_a: float
    score_b: float
    created_at: datetime
