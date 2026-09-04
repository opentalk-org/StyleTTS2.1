from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MosRatingCreate(BaseModel):
    audio_a_id: UUID
    audio_b_id: UUID
    preferred_audio_id: UUID
    score_a: float = Field(allow_inf_nan=False)
    score_b: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        if self.audio_a_id == self.audio_b_id:
            raise ValueError("MOS rating requires two distinct audio files")
        if self.preferred_audio_id not in (self.audio_a_id, self.audio_b_id):
            raise ValueError("MOS preferred audio must be a member of the pair")
        return self


class MosComparisonRead(BaseModel):
    id: UUID
    audio_a_id: UUID
    audio_b_id: UUID
    preferred_audio_id: UUID
    score_a: float
    score_b: float
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class MosRatingUpdate(BaseModel):
    preferred_audio_id: UUID
    score_a: float = Field(allow_inf_nan=False)
    score_b: float = Field(allow_inf_nan=False)


@dataclass(frozen=True)
class MosPair:
    dataset_id: UUID
    audio_a_id: UUID
    audio_b_id: UUID
