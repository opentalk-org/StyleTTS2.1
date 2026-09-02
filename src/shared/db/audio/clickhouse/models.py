from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class AudioFileRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    updated_at: datetime
    name: str
    bucket_file_id: UUID | None
    byte_offset: int
    duration: float
    byte_length: int
    score: float | None
    language: str | None
    style_prompt: str | None
    voice_prompt: str | None
    virtual: bool
    storage_ref: dict[str, Any] | None
    metadata: dict[str, Any]


class AudioFileUpdate(BaseModel):
    name: str
    bucket_file_id: UUID | None
    byte_offset: int
    duration: float
    byte_length: int
    score: float | None
    language: str | None
    style_prompt: str | None
    voice_prompt: str | None
    virtual: bool
    storage_ref: dict[str, Any] | None
    metadata: dict[str, Any]
    updated_at: datetime


class AudioSegmentRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    audio_file_id: UUID
    updated_at: datetime
    position: int
    start_seconds: float
    end_seconds: float
    text: str
    phon: str
    kind: str
    accuracy: float | None
    speaker_id: str | None
    metadata: dict[str, Any]
    alignment: list[dict[str, Any]] | None
