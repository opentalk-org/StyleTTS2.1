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
    previous_score_a: float | None
    previous_score_b: float | None
    created_at: datetime


class MosRatingDetailRead(MosRatingRead):
    audio_a: MosAudioRead
    audio_b: MosAudioRead
    can_modify: bool


class MosRatingPage(BaseModel):
    rows: list[MosRatingDetailRead]
    total: int
    limit: int
    offset: int
